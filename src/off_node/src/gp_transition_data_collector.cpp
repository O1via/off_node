#include <ros/ros.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <mavros_msgs/AttitudeTarget.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/PositionTarget.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace {
constexpr double kPi = 3.14159265358979323846;

inline double deg2rad(double deg) { return deg * kPi / 180.0; }

inline double clamp(double x, double lo, double hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

inline std::string joinPath(const std::string& dir, const std::string& name) {
  if (dir.empty()) return name;
  if (!dir.empty() && dir.back() == '/') return dir + name;
  return dir + "/" + name;
}

bool ensureDirectory(const std::string& path) {
  if (path.empty()) return false;

  std::string cur;
  if (path[0] == '/') cur = "/";

  std::stringstream ss(path);
  std::string part;
  while (std::getline(ss, part, '/')) {
    if (part.empty()) continue;
    if (!cur.empty() && cur.back() != '/') cur.push_back('/');
    cur += part;

    struct stat st;
    if (::stat(cur.c_str(), &st) == 0) {
      if (!S_ISDIR(st.st_mode)) return false;
      continue;
    }
    if (::mkdir(cur.c_str(), 0755) != 0 && errno != EEXIST) return false;
  }
  return true;
}

void quatToRpy(const geometry_msgs::Quaternion& q_msg,
               double& roll,
               double& pitch,
               double& yaw) {
  tf2::Quaternion q;
  tf2::fromMsg(q_msg, q);
  tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);
}
}  // namespace

struct TransitionPoint {
  ros::Time stamp;
  std::array<double, 8> x{};  // [pn, pe, vn, ve, pd, vd, phi, theta]
  std::array<double, 3> u{};  // [dT, phi_cmd, theta_cmd]

  double dT_cmd{0.0};
  double phi_cmd{0.0};
  double theta_cmd{0.0};
  double thrust_norm{0.0};
  double yaw_cmd{0.0};
  std::string profile{"circle"};
  double speed_xy{0.0};

  double vx_cmd{0.0};
  double vy_cmd{0.0};
  double vz_cmd{0.0};
  double x_ref{0.0};
  double y_ref{0.0};
  double z_ref{0.0};

  double u_from_fcu_target{1.0};
  double u_age_sec{0.0};
};

class GpTransitionDataCollector {
 public:
  explicit GpTransitionDataCollector(ros::NodeHandle& nh) : nh_(nh), pnh_("~") {
    // Core run config
    pnh_.param("rate_hz", rate_hz_, 10.0);
    pnh_.param("duration_sec", duration_sec_, 300.0);
    pnh_.param("max_dt_sec", max_dt_sec_, 0.5);
    pnh_.param("skip_initial_sec", skip_initial_sec_, 0.0);

    // Single consolidated output
    pnh_.param<std::string>("output_dir", output_dir_, std::string("/home/zxy/off_node/src/data_gp"));
    pnh_.param<std::string>("file_prefix", file_prefix_, std::string("gp_transitions"));

    // Vehicle / input mapping
    pnh_.param("mass_kg", mass_kg_, 1.5);
    pnh_.param("g", g_, 9.81);
    pnh_.param("hover_thrust_norm", hover_thrust_norm_, 0.60);
    pnh_.param("thrust_to_dT_scale", thrust_to_dT_scale_, -1.0);

    // FCU input freshness
    pnh_.param("target_att_topic", target_att_topic_, std::string("mavros/setpoint_raw/target_attitude"));
    pnh_.param("require_fcu_target_for_u", require_fcu_target_for_u_, true);
    pnh_.param("fcu_target_timeout_sec", fcu_target_timeout_sec_, 0.25);

    // Time alignment
    pnh_.param("align_state_to_input_stamp", align_state_to_input_stamp_, true);
    pnh_.param("align_state_max_gap_sec", align_state_max_gap_sec_, 0.08);
    pnh_.param("state_history_sec", state_history_sec_, 2.0);

    // Hover estimation stage
    pnh_.param("hover_estimation_sec", hover_estimation_sec_, 30.0);
    pnh_.param("hover_estimation_apply", hover_estimation_apply_, true);
    pnh_.param("hover_est_max_tilt_deg", hover_est_max_tilt_deg_, 8.0);
    pnh_.param("hover_est_max_vxy_mps", hover_est_max_vxy_mps_, 0.25);
    pnh_.param("hover_est_max_vz_mps", hover_est_max_vz_mps_, 0.12);
    pnh_.param("hover_est_min_samples", hover_est_min_samples_, 40);

    // PX4 velocity control only
    pnh_.param("auto_offboard_arm", auto_offboard_arm_, true);
    pnh_.param("yaw_deg", yaw_deg_, 0.0);
    pnh_.param("takeoff_z_m", takeoff_z_m_, 5.0);

    // Circle tracking only
    pnh_.param("circle_radius_m", circle_radius_m_, 8.0);
    pnh_.param("target_speed_mps", target_speed_mps_, 2.1);
    pnh_.param("track_pos_kp_xy", track_pos_kp_xy_, 0.20);
    pnh_.param("track_pos_kp_z", track_pos_kp_z_, 0.60);
    pnh_.param("track_pos_kd_z", track_pos_kd_z_, 0.15);
    pnh_.param("max_xy_speed_cmd_mps", max_xy_speed_cmd_mps_, 2.5);
    pnh_.param("max_z_speed_cmd_mps", max_z_speed_cmd_mps_, 1.2);
    pnh_.param("z_osc_enable", z_osc_enable_, true);
    pnh_.param("z_osc_amp_m", z_osc_amp_m_, 2.5);
    pnh_.param("z_osc_cycles_per_lap", z_osc_cycles_per_lap_, 0.5);

    // Topics
    pnh_.param<std::string>("pose_topic", pose_topic_, std::string("mavros/local_position/pose"));
    pnh_.param<std::string>("vel_topic", vel_topic_, std::string("mavros/local_position/velocity_local"));
    pnh_.param<std::string>("state_topic", state_topic_, std::string("mavros/state"));
    pnh_.param<std::string>("vel_sp_topic", vel_sp_topic_, std::string("mavros/setpoint_raw/local"));

    state_sub_ = nh_.subscribe<mavros_msgs::State>(state_topic_, 20, &GpTransitionDataCollector::stateCb, this);
    pose_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>(pose_topic_, 100, &GpTransitionDataCollector::poseCb, this);
    vel_sub_ = nh_.subscribe<geometry_msgs::TwistStamped>(vel_topic_, 100, &GpTransitionDataCollector::velCb, this);
    target_att_sub_ = nh_.subscribe<mavros_msgs::AttitudeTarget>(target_att_topic_, 100,
                                                                 &GpTransitionDataCollector::targetAttCb, this);

    local_sp_pub_ = nh_.advertise<mavros_msgs::PositionTarget>(vel_sp_topic_, 100);
    arming_client_ = nh_.serviceClient<mavros_msgs::CommandBool>("mavros/cmd/arming");
    set_mode_client_ = nh_.serviceClient<mavros_msgs::SetMode>("mavros/set_mode");

    ROS_INFO("[gp_collect] simplified mode: single episode + circle only");
    ROS_INFO("[gp_collect] control mode: pure velocity setpoint (position ignored by type_mask)");
    ROS_INFO("[gp_collect] target_speed=%.3f m/s, circle_radius=%.3f m", target_speed_mps_, circle_radius_m_);
    ROS_INFO("[gp_collect] z_osc: enable=%s amp=%.3f m cycles_per_lap=%.3f", z_osc_enable_ ? "true" : "false",
             z_osc_amp_m_, z_osc_cycles_per_lap_);
    ROS_INFO("[gp_collect] hover_estimation_sec=%.2f, apply=%s", hover_estimation_sec_,
             hover_estimation_apply_ ? "true" : "false");
    ROS_INFO("[gp_collect] align_state_to_input_stamp=%s, max_gap=%.3f", align_state_to_input_stamp_ ? "true" : "false",
             align_state_max_gap_sec_);
  }

  bool run() {
    if (!prepareOutput()) {
      ROS_ERROR("[gp_collect] failed to prepare output csv");
      return false;
    }

    ros::Rate rate(rate_hz_);

    ROS_INFO("[gp_collect] waiting FCU connection + pose/velocity...");
    while (ros::ok() && (!state_received_ || !current_state_.connected || !pose_received_ || !vel_received_)) {
      ros::spinOnce();
      rate.sleep();
    }

    // Capture circle center once at start.
    ref_center_x_enu_ = pose_msg_.pose.position.x;
    ref_center_y_enu_ = pose_msg_.pose.position.y;
    ref_center_z_enu_ = std::max(0.2, takeoff_z_m_);

    // Warm up setpoints before switching OFFBOARD
    ROS_INFO("[gp_collect] priming setpoints...");
    for (int i = 0; ros::ok() && i < static_cast<int>(2.0 * rate_hz_); ++i) {
      ros::Time now = ros::Time::now();
      publishVelocitySetpoint(hoverSetpoint(), now);
      ros::spinOnce();
      rate.sleep();
    }

    ros::Time last_request = ros::Time::now();
    const ros::Time t0 = ros::Time::now();
    const double data_start_sec = std::max(0.0, hover_estimation_sec_) + std::max(0.0, skip_initial_sec_);
    ROS_INFO("[gp_collect] hover phase=%.2fs, data_start=%.2fs", hover_estimation_sec_, data_start_sec);

    TransitionPoint prev;
    bool has_prev = false;
    bool hover_finalized = false;
    resetHoverEstimator();

    std::size_t rows = 0;
    std::size_t dropped = 0;

    while (ros::ok()) {
      ros::spinOnce();
      const ros::Time now = ros::Time::now();
      const double t = (now - t0).toSec();
      if (t >= duration_sec_) break;

      if (auto_offboard_arm_) {
        trySetModeAndArm(now, last_request);
      }

      const bool in_hover_phase = (t < hover_estimation_sec_);
      if (in_hover_phase) {
        publishVelocitySetpoint(hoverSetpoint(), now);
        updateHoverEstimator(now);
      } else {
        if (!hover_finalized) {
          finalizeHoverEstimator();
          hover_finalized = true;
        }
        publishVelocitySetpoint(circleSetpoint(t - hover_estimation_sec_), now);
      }

      TransitionPoint cur;
      if (!buildCurrentPoint(now, cur)) {
        dropped++;
        rate.sleep();
        continue;
      }

      if (has_prev) {
        const double dt = (cur.stamp - prev.stamp).toSec();
        if (dt > 1e-5 && dt <= max_dt_sec_) {
          if (t >= data_start_sec) {
            csv_ << formatRow(prev, cur, dt);
            rows++;
          } else {
            dropped++;
          }
        } else {
          dropped++;
        }
      }

      prev = cur;
      has_prev = true;
      rate.sleep();
    }

    if (!hover_finalized && hover_estimation_sec_ > 0.0) finalizeHoverEstimator();

    csv_.flush();
    csv_.close();

    ROS_INFO("[gp_collect] done. rows=%zu, dropped=%zu", rows, dropped);
    ROS_INFO("[gp_collect] output: %s", output_csv_path_.c_str());
    return true;
  }

 private:
  struct VelocitySetpointRef {
    double x_ref_enu{0.0};
    double y_ref_enu{0.0};
    double z_ref_enu{0.0};
    double vx_cmd_enu{0.0};
    double vy_cmd_enu{0.0};
    double vz_cmd_enu{0.0};
    double yaw_rad{0.0};
    std::string profile{"circle"};
  };

  std::string csvHeader() const {
    return std::string("episode,t_t,t_tp1,dt,") +
           "pn_t,pe_t,vn_t,ve_t,pd_t,vd_t,phi_t,theta_t," +
           "u0_t,u1_t,u2_t," +
           "dT_cmd_t,phi_cmd_t,theta_cmd_t," +
           "pn_tp1,pe_tp1,vn_tp1,ve_tp1,pd_tp1,vd_tp1,phi_tp1,theta_tp1," +
           "thrust_norm_t,yaw_cmd_t,profile_t,speed_xy_t,target_speed_ep,cmd_scale_t," +
           "vx_cmd_enu_t,vy_cmd_enu_t,vz_cmd_enu_t,x_ref_enu_t,y_ref_enu_t,z_ref_enu_t," +
           "u_from_fcu_target_t,u_age_sec_t,ut_source_t\n";
  }

  bool prepareOutput() {
    if (!ensureDirectory(output_dir_)) return false;
    output_csv_path_ = joinPath(output_dir_, file_prefix_ + std::string("_all.csv"));
    csv_.open(output_csv_path_.c_str(), std::ios::out | std::ios::trunc);
    if (!csv_.is_open()) return false;
    csv_ << std::fixed << std::setprecision(9);
    csv_ << csvHeader();
    return true;
  }

  void trimStateHistory(const ros::Time& now) {
    const double keep_sec = std::max(0.5, state_history_sec_);
    while (!pose_hist_.empty() && (now - pose_hist_.front().header.stamp).toSec() > keep_sec) {
      pose_hist_.pop_front();
    }
    while (!vel_hist_.empty() && (now - vel_hist_.front().header.stamp).toSec() > keep_sec) {
      vel_hist_.pop_front();
    }
  }

  bool lookupNearestPose(const ros::Time& t, geometry_msgs::PoseStamped& out) const {
    if (pose_hist_.empty()) return false;
    std::size_t best_i = 0;
    double best_abs = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < pose_hist_.size(); ++i) {
      const double d = std::abs((pose_hist_[i].header.stamp - t).toSec());
      if (d < best_abs) {
        best_abs = d;
        best_i = i;
      }
    }
    if (best_abs > align_state_max_gap_sec_) return false;
    out = pose_hist_[best_i];
    return true;
  }

  bool lookupNearestVel(const ros::Time& t, geometry_msgs::TwistStamped& out) const {
    if (vel_hist_.empty()) return false;
    std::size_t best_i = 0;
    double best_abs = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < vel_hist_.size(); ++i) {
      const double d = std::abs((vel_hist_[i].header.stamp - t).toSec());
      if (d < best_abs) {
        best_abs = d;
        best_i = i;
      }
    }
    if (best_abs > align_state_max_gap_sec_) return false;
    out = vel_hist_[best_i];
    return true;
  }

  bool resolveSampleState(const ros::Time& sample_stamp,
                          geometry_msgs::PoseStamped& pose_out,
                          geometry_msgs::TwistStamped& vel_out) const {
    if (!align_state_to_input_stamp_) {
      pose_out = pose_msg_;
      vel_out = vel_msg_;
      return true;
    }
    if (!lookupNearestPose(sample_stamp, pose_out)) return false;
    if (!lookupNearestVel(sample_stamp, vel_out)) return false;
    return true;
  }

  bool fcuTargetReady(const ros::Time& now) const {
    if (!fcu_target_received_ || !fcu_target_have_att_ || !fcu_target_have_thrust_) return false;
    if (fcu_target_timeout_sec_ <= 0.0) return true;
    return (now - fcu_target_stamp_).toSec() <= fcu_target_timeout_sec_;
  }

  double thrustNormToDT(double thrust_norm) const {
    if (thrust_to_dT_scale_ > 0.0) {
      return (thrust_norm - hover_thrust_norm_) * thrust_to_dT_scale_;
    }
    const double safe_hover = std::max(1e-3, hover_thrust_norm_);
    return (thrust_norm - hover_thrust_norm_) * (mass_kg_ * g_ / safe_hover);
  }

  bool buildCurrentPoint(const ros::Time& now, TransitionPoint& pt) {
    if (!pose_received_ || !vel_received_) return false;
    if (!fcuTargetReady(now)) {
      if (require_fcu_target_for_u_) {
        ROS_WARN_THROTTLE(2.0, "[gp_collect] waiting fresh FCU target on %s", target_att_topic_.c_str());
        return false;
      }
    }

    const double phi_cmd = fcu_target_roll_;
    const double theta_cmd = fcu_target_pitch_;
    const double yaw_cmd = fcu_target_yaw_;
    const double thrust_norm = fcu_target_thrust_norm_;
    const double dT = thrustNormToDT(thrust_norm);

    ros::Time sample_stamp = align_state_to_input_stamp_ ? fcu_target_stamp_ : now;

    geometry_msgs::PoseStamped pose_s;
    geometry_msgs::TwistStamped vel_s;
    if (!resolveSampleState(sample_stamp, pose_s, vel_s)) {
      ROS_WARN_THROTTLE(2.0, "[gp_collect] state alignment failed (stamp=%.3f)", sample_stamp.toSec());
      return false;
    }

    double roll = 0.0, pitch = 0.0, yaw = 0.0;
    quatToRpy(pose_s.pose.orientation, roll, pitch, yaw);

    const double x_enu = pose_s.pose.position.x;
    const double y_enu = pose_s.pose.position.y;
    const double z_enu = pose_s.pose.position.z;
    const double vx_enu = vel_s.twist.linear.x;
    const double vy_enu = vel_s.twist.linear.y;
    const double vz_enu = vel_s.twist.linear.z;

    pt.stamp = sample_stamp;
    pt.x[0] = y_enu;
    pt.x[1] = x_enu;
    pt.x[2] = vy_enu;
    pt.x[3] = vx_enu;
    pt.x[4] = -z_enu;
    pt.x[5] = -vz_enu;
    pt.x[6] = roll;
    pt.x[7] = pitch;

    pt.u[0] = dT;
    pt.u[1] = phi_cmd;
    pt.u[2] = theta_cmd;

    pt.dT_cmd = dT;
    pt.phi_cmd = phi_cmd;
    pt.theta_cmd = theta_cmd;
    pt.thrust_norm = thrust_norm;
    pt.yaw_cmd = yaw_cmd;
    pt.profile = current_profile_;
    pt.speed_xy = std::sqrt(vx_enu * vx_enu + vy_enu * vy_enu);
    pt.vx_cmd = last_vx_cmd_enu_;
    pt.vy_cmd = last_vy_cmd_enu_;
    pt.vz_cmd = last_vz_cmd_enu_;
    pt.x_ref = last_x_ref_enu_;
    pt.y_ref = last_y_ref_enu_;
    pt.z_ref = last_z_ref_enu_;
    pt.u_from_fcu_target = 1.0;
    pt.u_age_sec = (sample_stamp - fcu_target_stamp_).toSec();
    return true;
  }

  std::string formatRow(const TransitionPoint& a, const TransitionPoint& b, double dt) const {
    std::ostringstream os;
    os << std::fixed << std::setprecision(9);
    os << 1 << ',' << a.stamp.toSec() << ',' << b.stamp.toSec() << ',' << dt;
    for (double v : a.x) os << ',' << v;
    for (double v : a.u) os << ',' << v;
    os << ',' << a.dT_cmd << ',' << a.phi_cmd << ',' << a.theta_cmd;
    for (double v : b.x) os << ',' << v;
    os << ',' << a.thrust_norm << ',' << a.yaw_cmd << ',' << a.profile;
    os << ',' << a.speed_xy << ',' << target_speed_mps_ << ',' << 1.0;
    os << ',' << a.vx_cmd << ',' << a.vy_cmd << ',' << a.vz_cmd;
    os << ',' << a.x_ref << ',' << a.y_ref << ',' << a.z_ref;
    os << ',' << a.u_from_fcu_target << ',' << a.u_age_sec << ",fcu_target_attitude\n";
    return os.str();
  }

  VelocitySetpointRef hoverSetpoint() const {
    VelocitySetpointRef sp;
    sp.x_ref_enu = ref_center_x_enu_;
    sp.y_ref_enu = ref_center_y_enu_;
    sp.z_ref_enu = ref_center_z_enu_;
    sp.vx_cmd_enu = 0.0;
    sp.vy_cmd_enu = 0.0;
    sp.vz_cmd_enu = clamp(track_pos_kp_z_ * (sp.z_ref_enu - pose_msg_.pose.position.z) -
                              track_pos_kd_z_ * vel_msg_.twist.linear.z,
                          -max_z_speed_cmd_mps_, max_z_speed_cmd_mps_);
    sp.yaw_rad = deg2rad(yaw_deg_);
    sp.profile = "hover_calib";
    return sp;
  }

  VelocitySetpointRef circleSetpoint(double t_motion) const {
    VelocitySetpointRef sp;
    const double r = std::max(0.5, circle_radius_m_);
    const double v_target = std::max(0.1, target_speed_mps_);
    const double w = v_target / r;

    const double z_meas = pose_msg_.pose.position.z;
    const double vz_meas = vel_msg_.twist.linear.z;

    const double x_ref = ref_center_x_enu_ + r * std::cos(w * t_motion);
    const double y_ref = ref_center_y_enu_ + r * std::sin(w * t_motion);
    double z_ref = ref_center_z_enu_;
    double vz_ff = 0.0;
    const double wz = std::max(0.0, z_osc_cycles_per_lap_) * w;
    if (z_osc_enable_ && z_osc_amp_m_ > 1e-6 && wz > 1e-6) {
      z_ref = ref_center_z_enu_ + z_osc_amp_m_ * std::sin(wz * t_motion);
      vz_ff = z_osc_amp_m_ * wz * std::cos(wz * t_motion);
    }

    const double vx_ff = -r * w * std::sin(w * t_motion);
    const double vy_ff = +r * w * std::cos(w * t_motion);

    const double ez = z_ref - z_meas;

    // Pure velocity mode: XY uses feedforward only.
    double vx_cmd = vx_ff;
    double vy_cmd = vy_ff;
    double vz_cmd = vz_ff + track_pos_kp_z_ * ez - track_pos_kd_z_ * vz_meas;

    const double vxy = std::sqrt(vx_cmd * vx_cmd + vy_cmd * vy_cmd);
    const double max_vxy = std::max(0.2, max_xy_speed_cmd_mps_);
    if (vxy > max_vxy) {
      const double s = max_vxy / vxy;
      vx_cmd *= s;
      vy_cmd *= s;
    }
    vz_cmd = clamp(vz_cmd, -max_z_speed_cmd_mps_, max_z_speed_cmd_mps_);

    sp.x_ref_enu = x_ref;
    sp.y_ref_enu = y_ref;
    sp.z_ref_enu = z_ref;
    sp.vx_cmd_enu = vx_cmd;
    sp.vy_cmd_enu = vy_cmd;
    sp.vz_cmd_enu = vz_cmd;
    sp.yaw_rad = deg2rad(yaw_deg_);
    sp.profile = "circle";
    return sp;
  }

  void publishVelocitySetpoint(const VelocitySetpointRef& sp, const ros::Time& now) {
    mavros_msgs::PositionTarget msg;
    msg.header.stamp = now;
    msg.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
    // Pure velocity offboard: ignore XYZ position, keep velocity+yaw.
    msg.type_mask = mavros_msgs::PositionTarget::IGNORE_PX |
                    mavros_msgs::PositionTarget::IGNORE_PY |
                    mavros_msgs::PositionTarget::IGNORE_PZ |
                    mavros_msgs::PositionTarget::IGNORE_AFX |
                    mavros_msgs::PositionTarget::IGNORE_AFY |
                    mavros_msgs::PositionTarget::IGNORE_AFZ |
                    mavros_msgs::PositionTarget::IGNORE_YAW_RATE;
    msg.position.x = sp.x_ref_enu;
    msg.position.y = sp.y_ref_enu;
    msg.position.z = sp.z_ref_enu;
    msg.velocity.x = sp.vx_cmd_enu;
    msg.velocity.y = sp.vy_cmd_enu;
    msg.velocity.z = sp.vz_cmd_enu;
    msg.yaw = sp.yaw_rad;
    msg.yaw_rate = 0.0;
    local_sp_pub_.publish(msg);

    current_profile_ = sp.profile;
    last_vx_cmd_enu_ = sp.vx_cmd_enu;
    last_vy_cmd_enu_ = sp.vy_cmd_enu;
    last_vz_cmd_enu_ = sp.vz_cmd_enu;
    last_x_ref_enu_ = sp.x_ref_enu;
    last_y_ref_enu_ = sp.y_ref_enu;
    last_z_ref_enu_ = sp.z_ref_enu;
  }

  void trySetModeAndArm(const ros::Time& now, ros::Time& last_request) {
    if (current_state_.mode != "OFFBOARD" && (now - last_request > ros::Duration(1.0))) {
      mavros_msgs::SetMode set_mode;
      set_mode.request.custom_mode = "OFFBOARD";
      if (set_mode_client_.call(set_mode) && set_mode.response.mode_sent) {
        ROS_INFO_THROTTLE(2.0, "[gp_collect] OFFBOARD enabled");
      }
      last_request = now;
      return;
    }

    if (!current_state_.armed && (now - last_request > ros::Duration(1.0))) {
      mavros_msgs::CommandBool arm_cmd;
      arm_cmd.request.value = true;
      if (arming_client_.call(arm_cmd) && arm_cmd.response.success) {
        ROS_INFO_THROTTLE(2.0, "[gp_collect] Vehicle armed");
      }
      last_request = now;
    }
  }

  void resetHoverEstimator() {
    hover_est_samples_.clear();
    hover_est_reject_tilt_ = 0;
    hover_est_reject_speed_ = 0;
    hover_est_reject_fcu_ = 0;
  }

  void updateHoverEstimator(const ros::Time& now) {
    if (!pose_received_ || !vel_received_) return;
    if (!fcuTargetReady(now)) {
      hover_est_reject_fcu_++;
      return;
    }

    double roll = 0.0, pitch = 0.0, yaw = 0.0;
    quatToRpy(pose_msg_.pose.orientation, roll, pitch, yaw);
    const double tilt = std::sqrt(roll * roll + pitch * pitch);
    if (tilt > deg2rad(std::max(1.0, hover_est_max_tilt_deg_))) {
      hover_est_reject_tilt_++;
      return;
    }

    const double vx = vel_msg_.twist.linear.x;
    const double vy = vel_msg_.twist.linear.y;
    const double vz = vel_msg_.twist.linear.z;
    const double vxy = std::sqrt(vx * vx + vy * vy);
    if (vxy > hover_est_max_vxy_mps_ || std::abs(vz) > hover_est_max_vz_mps_) {
      hover_est_reject_speed_++;
      return;
    }

    const double est = fcu_target_thrust_norm_ * std::cos(roll) * std::cos(pitch);
    if (std::isfinite(est) && est > 0.05 && est < 0.98) hover_est_samples_.push_back(est);
  }

  void finalizeHoverEstimator() {
    if (hover_est_samples_.empty()) {
      ROS_WARN("[gp_collect] hover estimator: no valid samples; keep hover_thrust_norm=%.6f", hover_thrust_norm_);
      return;
    }

    std::vector<double> v = hover_est_samples_;
    std::sort(v.begin(), v.end());
    const double median = v[v.size() / 2];
    const std::size_t i10 = static_cast<std::size_t>(0.10 * static_cast<double>(v.size() - 1));
    const std::size_t i90 = static_cast<std::size_t>(0.90 * static_cast<double>(v.size() - 1));
    const double p10 = v[i10];
    const double p90 = v[i90];

    if (hover_estimation_apply_ && static_cast<int>(v.size()) >= hover_est_min_samples_) {
      const double old = hover_thrust_norm_;
      hover_thrust_norm_ = clamp(median, 0.05, 0.95);
      ROS_INFO("[gp_collect] hover estimator applied: old=%.6f -> new=%.6f (n=%zu, p10=%.6f, p90=%.6f, rej_tilt=%zu, rej_speed=%zu, rej_fcu=%zu)",
               old, hover_thrust_norm_, v.size(), p10, p90,
               hover_est_reject_tilt_, hover_est_reject_speed_, hover_est_reject_fcu_);
    } else {
      ROS_WARN("[gp_collect] hover estimator not applied (n=%zu, min=%d, apply=%s); keep hover_thrust_norm=%.6f (median=%.6f)",
               v.size(), hover_est_min_samples_, hover_estimation_apply_ ? "true" : "false",
               hover_thrust_norm_, median);
    }
  }

  // callbacks
  void stateCb(const mavros_msgs::State::ConstPtr& msg) {
    current_state_ = *msg;
    state_received_ = true;
  }

  void poseCb(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    pose_msg_ = *msg;
    if (pose_msg_.header.stamp.isZero()) pose_msg_.header.stamp = ros::Time::now();
    pose_hist_.push_back(pose_msg_);
    trimStateHistory(pose_msg_.header.stamp);
    pose_received_ = true;
  }

  void velCb(const geometry_msgs::TwistStamped::ConstPtr& msg) {
    vel_msg_ = *msg;
    if (vel_msg_.header.stamp.isZero()) vel_msg_.header.stamp = ros::Time::now();
    vel_hist_.push_back(vel_msg_);
    trimStateHistory(vel_msg_.header.stamp);
    vel_received_ = true;
  }

  void targetAttCb(const mavros_msgs::AttitudeTarget::ConstPtr& msg) {
    fcu_target_stamp_ = msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;

    if ((msg->type_mask & mavros_msgs::AttitudeTarget::IGNORE_ATTITUDE) == 0U) {
      quatToRpy(msg->orientation, fcu_target_roll_, fcu_target_pitch_, fcu_target_yaw_);
      fcu_target_have_att_ = true;
    }
    if ((msg->type_mask & mavros_msgs::AttitudeTarget::IGNORE_THRUST) == 0U) {
      fcu_target_thrust_norm_ = msg->thrust;
      fcu_target_have_thrust_ = true;
    }
    fcu_target_received_ = fcu_target_have_att_ && fcu_target_have_thrust_;
  }

 private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Subscriber state_sub_;
  ros::Subscriber pose_sub_;
  ros::Subscriber vel_sub_;
  ros::Subscriber target_att_sub_;

  ros::Publisher local_sp_pub_;
  ros::ServiceClient arming_client_;
  ros::ServiceClient set_mode_client_;

  mavros_msgs::State current_state_;
  geometry_msgs::PoseStamped pose_msg_;
  geometry_msgs::TwistStamped vel_msg_;
  std::deque<geometry_msgs::PoseStamped> pose_hist_;
  std::deque<geometry_msgs::TwistStamped> vel_hist_;

  bool state_received_{false};
  bool pose_received_{false};
  bool vel_received_{false};

  ros::Time fcu_target_stamp_;
  bool fcu_target_received_{false};
  bool fcu_target_have_att_{false};
  bool fcu_target_have_thrust_{false};
  double fcu_target_roll_{0.0};
  double fcu_target_pitch_{0.0};
  double fcu_target_yaw_{0.0};
  double fcu_target_thrust_norm_{0.0};

  std::string current_profile_{"hover_calib"};
  double last_vx_cmd_enu_{0.0};
  double last_vy_cmd_enu_{0.0};
  double last_vz_cmd_enu_{0.0};
  double last_x_ref_enu_{0.0};
  double last_y_ref_enu_{0.0};
  double last_z_ref_enu_{0.0};

  // Params
  double rate_hz_{10.0};
  double duration_sec_{300.0};
  double max_dt_sec_{0.5};
  double skip_initial_sec_{0.0};

  std::string output_dir_;
  std::string file_prefix_;
  std::string output_csv_path_;
  std::ofstream csv_;

  double mass_kg_{1.5};
  double g_{9.81};
  double hover_thrust_norm_{0.60};
  double thrust_to_dT_scale_{-1.0};

  std::string target_att_topic_;
  bool require_fcu_target_for_u_{true};
  double fcu_target_timeout_sec_{0.25};

  bool align_state_to_input_stamp_{true};
  double align_state_max_gap_sec_{0.08};
  double state_history_sec_{2.0};

  double hover_estimation_sec_{30.0};
  bool hover_estimation_apply_{true};
  double hover_est_max_tilt_deg_{8.0};
  double hover_est_max_vxy_mps_{0.25};
  double hover_est_max_vz_mps_{0.12};
  int hover_est_min_samples_{40};
  std::vector<double> hover_est_samples_;
  std::size_t hover_est_reject_tilt_{0};
  std::size_t hover_est_reject_speed_{0};
  std::size_t hover_est_reject_fcu_{0};

  bool auto_offboard_arm_{true};
  double yaw_deg_{0.0};
  double takeoff_z_m_{2.0};

  double circle_radius_m_{4.0};
  double target_speed_mps_{2.0};
  double track_pos_kp_xy_{0.20};
  double track_pos_kp_z_{0.60};
  double track_pos_kd_z_{0.15};
  double max_xy_speed_cmd_mps_{2.5};
  double max_z_speed_cmd_mps_{1.2};
  bool z_osc_enable_{true};
  double z_osc_amp_m_{0.60};
  double z_osc_cycles_per_lap_{2.0};

  std::string pose_topic_;
  std::string vel_topic_;
  std::string state_topic_;
  std::string vel_sp_topic_;

  double ref_center_x_enu_{0.0};
  double ref_center_y_enu_{0.0};
  double ref_center_z_enu_{2.0};
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "gp_transition_data_collector");
  ros::NodeHandle nh;

  GpTransitionDataCollector node(nh);
  const bool ok = node.run();
  return ok ? 0 : 1;
}
