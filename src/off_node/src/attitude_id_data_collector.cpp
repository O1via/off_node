#include <ros/ros.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>

#include <mavros_msgs/AttitudeTarget.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>

#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Float64.h>

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

struct CommandRPY {
  double roll_rad{0.0};
  double pitch_rad{0.0};
  double yaw_rad{0.0};
};

class AttitudeIdCollector {
 public:
  AttitudeIdCollector(ros::NodeHandle &nh)
      : nh_(nh), pnh_("~"), imu_received_(false), state_received_(false), pose_received_(false) {
    pnh_.param("duration_sec", duration_sec_, 60.0);
    pnh_.param("rate_hz", rate_hz_, 120.0);
    pnh_.param("step_amp_deg", step_amp_deg_, 5.0);
    pnh_.param("step_hold_sec", step_hold_sec_, 2.0);
    pnh_.param("sine_amp_deg", sine_amp_deg_, 4.0);
    pnh_.param("sine_freq_hz", sine_freq_hz_, 0.35);
    pnh_.param("yaw_deg", yaw_deg_, 0.0);
    pnh_.param("thrust", thrust_, 0.60);  // 作为悬停基准推力
    pnh_.param("takeoff_thrust", takeoff_thrust_, 0.72);
    pnh_.param("takeoff_z_m", takeoff_z_m_, 1.2);
    pnh_.param("takeoff_timeout_sec", takeoff_timeout_sec_, 12.0);
    pnh_.param("z_kp", z_kp_, 0.22);
    pnh_.param("z_kd", z_kd_, 0.10);
    pnh_.param("thrust_min", thrust_min_, 0.45);
    pnh_.param("thrust_max", thrust_max_, 0.90);
    pnh_.param("enable_tilt_comp", enable_tilt_comp_, true);
    pnh_.param("auto_offboard_arm", auto_offboard_arm_, true);
    pnh_.param<std::string>("csv_path", csv_path_, std::string("/home/zxy/off_node/src/data_system_identification/iris_attitude_id_60s.csv"));

    state_sub_ = nh_.subscribe<mavros_msgs::State>("mavros/state", 20,
                                                   &AttitudeIdCollector::stateCb, this);
    imu_sub_ = nh_.subscribe<sensor_msgs::Imu>("mavros/imu/data", 200,
                                               &AttitudeIdCollector::imuCb, this);
    local_pose_sub_ = nh_.subscribe<geometry_msgs::PoseStamped>(
      "mavros/local_position/pose", 50, &AttitudeIdCollector::localPoseCb, this);

    att_sp_pub_ = nh_.advertise<mavros_msgs::AttitudeTarget>(
        "mavros/setpoint_raw/attitude", 50);
    phi_pub_ = nh_.advertise<std_msgs::Float64>("attitude_id/phi", 50);
    theta_pub_ = nh_.advertise<std_msgs::Float64>("attitude_id/theta", 50);
    phi_cmd_pub_ = nh_.advertise<std_msgs::Float64>("attitude_id/phi_cmd", 50);
    theta_cmd_pub_ = nh_.advertise<std_msgs::Float64>("attitude_id/theta_cmd", 50);

    arming_client_ = nh_.serviceClient<mavros_msgs::CommandBool>("mavros/cmd/arming");
    set_mode_client_ = nh_.serviceClient<mavros_msgs::SetMode>("mavros/set_mode");
  }

  bool run() {
    ros::Rate rate(rate_hz_);

    ROS_INFO("[attitude_id] Waiting for FCU connection...");
    while (ros::ok() && (!state_received_ || !current_state_.connected)) {
      ros::spinOnce();
      rate.sleep();
    }

    if (!openCsv()) {
      return false;
    }

    // PX4 OFFBOARD 进入前需先持续发送 setpoint。
    CommandRPY zero_cmd;
    zero_cmd.yaw_rad = deg2rad(yaw_deg_);
    ROS_INFO("[attitude_id] Priming setpoints before OFFBOARD...");
    for (int i = 0; ros::ok() && i < static_cast<int>(2.0 * rate_hz_); ++i) {
      publishAttitudeSetpoint(zero_cmd);
      ros::spinOnce();
      rate.sleep();
    }

    ros::Time last_request = ros::Time::now();
    ros::Time pre_t0 = ros::Time::now();
    ros::Time t0;
    bool excitation_started = false;

    ROS_INFO("[attitude_id] Waiting takeoff to z>=%.2f m (timeout %.1f s)...",
             takeoff_z_m_, takeoff_timeout_sec_);

    while (ros::ok()) {
      ros::Time now = ros::Time::now();

      if (auto_offboard_arm_) {
        trySetModeAndArm(now, last_request);
      }

      if (!excitation_started) {
        CommandRPY cmd_takeoff;
        cmd_takeoff.yaw_rad = deg2rad(yaw_deg_);
        publishAttitudeSetpoint(cmd_takeoff, takeoff_thrust_);
        publishPlotTopics(cmd_takeoff);

        bool reached_alt = pose_received_ && (z_meas_ >= takeoff_z_m_);
        bool timeout = (now - pre_t0).toSec() >= takeoff_timeout_sec_;
        if (reached_alt || timeout) {
          excitation_started = true;
          t0 = ros::Time::now();
          z_ref_ = pose_received_ ? z_meas_ : takeoff_z_m_;
          prev_z_meas_ = z_meas_;
          prev_z_t_ = now;
          ROS_INFO("[attitude_id] Start excitation for %.1f s, output: %s",
                   duration_sec_, csv_path_.c_str());
          ROS_INFO("[attitude_id] Altitude-hold active: z_ref=%.2f m, thrust_base=%.2f",
                   z_ref_, thrust_);
        }

        ros::spinOnce();
        rate.sleep();
        continue;
      }

      double t = (now - t0).toSec();

      CommandRPY cmd = commandAt(t);
      double thrust_cmd = computeThrustForExcitation(now, cmd);
      publishAttitudeSetpoint(cmd, thrust_cmd);
      publishPlotTopics(cmd);
      writeCsvRow(t, cmd);

      if (t >= duration_sec_) {
        break;
      }

      ros::spinOnce();
      rate.sleep();
    }

    csv_.flush();
    csv_.close();
    ROS_INFO("[attitude_id] Finished. CSV saved to: %s", csv_path_.c_str());
    return true;
  }

 private:
  static double deg2rad(double deg) { return deg * M_PI / 180.0; }

  static double clamp(double x, double lo, double hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
  }

  void stateCb(const mavros_msgs::State::ConstPtr &msg) {
    current_state_ = *msg;
    state_received_ = true;
  }

  void imuCb(const sensor_msgs::Imu::ConstPtr &msg) {
    tf2::Quaternion q;
    tf2::fromMsg(msg->orientation, q);
    tf2::Matrix3x3(q).getRPY(roll_meas_, pitch_meas_, yaw_meas_);
    imu_received_ = true;
  }

  void localPoseCb(const geometry_msgs::PoseStamped::ConstPtr &msg) {
    z_meas_ = msg->pose.position.z;
    pose_received_ = true;
  }

  double computeThrustForExcitation(const ros::Time &now, const CommandRPY &cmd) {
    if (!pose_received_) {
      return thrust_;
    }

    double dt = (now - prev_z_t_).toSec();
    if (dt <= 1e-4) {
      dt = 1.0 / rate_hz_;
    }

    double z_dot = (z_meas_ - prev_z_meas_) / dt;
    prev_z_meas_ = z_meas_;
    prev_z_t_ = now;

    double e_z = z_ref_ - z_meas_;
    double u = thrust_ + z_kp_ * e_z - z_kd_ * z_dot;

    // 姿态倾斜时，竖直分量下降，做简易补偿。
    if (enable_tilt_comp_) {
      double c = std::cos(cmd.roll_rad) * std::cos(cmd.pitch_rad);
      c = std::max(0.55, c);
      u /= c;
    }

    return clamp(u, thrust_min_, thrust_max_);
  }

  bool openCsv() {
    csv_.open(csv_path_.c_str(), std::ios::out | std::ios::trunc);
    if (!csv_.is_open()) {
      ROS_ERROR("[attitude_id] Failed to open csv file: %s", csv_path_.c_str());
      return false;
    }

    csv_ << "t,phi,theta,phi_cmd,theta_cmd\n";
    csv_ << std::fixed << std::setprecision(6);
    return true;
  }

  void trySetModeAndArm(const ros::Time &now, ros::Time &last_request) {
    if (current_state_.mode != "OFFBOARD" &&
        (now - last_request > ros::Duration(1.0))) {
      mavros_msgs::SetMode set_mode;
      set_mode.request.custom_mode = "OFFBOARD";
      if (set_mode_client_.call(set_mode) && set_mode.response.mode_sent) {
        ROS_INFO_THROTTLE(2.0, "[attitude_id] OFFBOARD enabled");
      }
      last_request = now;
      return;
    }

    if (!current_state_.armed && (now - last_request > ros::Duration(1.0))) {
      mavros_msgs::CommandBool arm_cmd;
      arm_cmd.request.value = true;
      if (arming_client_.call(arm_cmd) && arm_cmd.response.success) {
        ROS_INFO_THROTTLE(2.0, "[attitude_id] Vehicle armed");
      }
      last_request = now;
    }
  }

  CommandRPY commandAt(double t) const {
    CommandRPY cmd;
    cmd.yaw_rad = deg2rad(yaw_deg_);

    const double amp = deg2rad(step_amp_deg_);
    const double hold = step_hold_sec_;

    // 0~5s: 稳定悬停
    if (t < 5.0) {
      cmd.roll_rad = 0.0;
      cmd.pitch_rad = 0.0;
      return cmd;
    }

    // 5~29s: roll 阶跃激励（pitch=0）
    if (t < 29.0) {
      int idx = static_cast<int>((t - 5.0) / hold) % 4;
      if (idx == 0) cmd.roll_rad = +amp;
      if (idx == 1) cmd.roll_rad = 0.0;
      if (idx == 2) cmd.roll_rad = -amp;
      if (idx == 3) cmd.roll_rad = 0.0;
      cmd.pitch_rad = 0.0;
      return cmd;
    }

    // 29~53s: pitch 阶跃激励（roll=0）
    if (t < 53.0) {
      int idx = static_cast<int>((t - 29.0) / hold) % 4;
      if (idx == 0) cmd.pitch_rad = +amp;
      if (idx == 1) cmd.pitch_rad = 0.0;
      if (idx == 2) cmd.pitch_rad = -amp;
      if (idx == 3) cmd.pitch_rad = 0.0;
      cmd.roll_rad = 0.0;
      return cmd;
    }

    // 53~60s: 小幅正弦，补充频域激励
    double tt = t - 53.0;
    double s_amp = deg2rad(sine_amp_deg_);
    double w = 2.0 * M_PI * sine_freq_hz_;
    cmd.roll_rad = s_amp * std::sin(w * tt);
    cmd.pitch_rad = s_amp * std::sin(w * tt + M_PI_2);
    return cmd;
  }

  void publishAttitudeSetpoint(const CommandRPY &cmd, double thrust_cmd = -1.0) {
    mavros_msgs::AttitudeTarget msg;
    msg.header.stamp = ros::Time::now();
    msg.type_mask = mavros_msgs::AttitudeTarget::IGNORE_ROLL_RATE |
                    mavros_msgs::AttitudeTarget::IGNORE_PITCH_RATE |
                    mavros_msgs::AttitudeTarget::IGNORE_YAW_RATE;

    tf2::Quaternion q;
    q.setRPY(cmd.roll_rad, cmd.pitch_rad, cmd.yaw_rad);
    q.normalize();
    msg.orientation = tf2::toMsg(q);
    msg.thrust = (thrust_cmd >= 0.0) ? thrust_cmd : thrust_;

    att_sp_pub_.publish(msg);
  }

  void publishPlotTopics(const CommandRPY &cmd) {
    std_msgs::Float64 phi_msg;
    std_msgs::Float64 theta_msg;
    std_msgs::Float64 phi_cmd_msg;
    std_msgs::Float64 theta_cmd_msg;

    phi_msg.data = roll_meas_;
    theta_msg.data = pitch_meas_;
    phi_cmd_msg.data = cmd.roll_rad;
    theta_cmd_msg.data = cmd.pitch_rad;

    phi_pub_.publish(phi_msg);
    theta_pub_.publish(theta_msg);
    phi_cmd_pub_.publish(phi_cmd_msg);
    theta_cmd_pub_.publish(theta_cmd_msg);
  }

  void writeCsvRow(double t, const CommandRPY &cmd) {
    if (!csv_.is_open()) {
      return;
    }

    if (!imu_received_) {
      // IMU 未到时先写 0，保证时间序列连续
      csv_ << t << ',' << 0.0 << ',' << 0.0 << ',' << cmd.roll_rad << ','
           << cmd.pitch_rad << '\n';
      return;
    }

    csv_ << t << ',' << roll_meas_ << ',' << pitch_meas_ << ',' << cmd.roll_rad
         << ',' << cmd.pitch_rad << '\n';
  }

 private:
  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;

  ros::Subscriber state_sub_;
  ros::Subscriber imu_sub_;
  ros::Subscriber local_pose_sub_;

  ros::Publisher att_sp_pub_;
  ros::Publisher phi_pub_;
  ros::Publisher theta_pub_;
  ros::Publisher phi_cmd_pub_;
  ros::Publisher theta_cmd_pub_;

  ros::ServiceClient arming_client_;
  ros::ServiceClient set_mode_client_;

  mavros_msgs::State current_state_;
  bool imu_received_;
  bool state_received_;

  double roll_meas_{0.0};
  double pitch_meas_{0.0};
  double yaw_meas_{0.0};
  double z_meas_{0.0};

  double duration_sec_{60.0};
  double rate_hz_{120.0};
  double step_amp_deg_{7.0};
  double step_hold_sec_{1.5};
  double sine_amp_deg_{4.0};
  double sine_freq_hz_{0.35};
  double yaw_deg_{0.0};
  double thrust_{0.60};
  double takeoff_thrust_{0.72};
  double takeoff_z_m_{1.2};
  double takeoff_timeout_sec_{12.0};
  double z_kp_{0.22};
  double z_kd_{0.10};
  double thrust_min_{0.45};
  double thrust_max_{0.90};
  bool enable_tilt_comp_{true};
  bool auto_offboard_arm_{true};

  bool pose_received_;
  double z_ref_{1.2};
  double prev_z_meas_{0.0};
  ros::Time prev_z_t_;

  std::string csv_path_;
  std::ofstream csv_;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "attitude_id_data_collector");
  ros::NodeHandle nh;

  AttitudeIdCollector node(nh);
  bool ok = node.run();
  return ok ? 0 : 1;
}
