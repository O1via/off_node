#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/PositionTarget.h>
#include <mavros_msgs/AttitudeTarget.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <cmath>
#include <string>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/WrenchStamped.h>
#include <mavros_msgs/ActuatorControl.h>

double roll, pitch, yaw;
tf::Quaternion angle;

mavros_msgs::State current_state;
geometry_msgs::PoseStamped local_position;
geometry_msgs::TwistStamped current_velocity;
geometry_msgs::Vector3 current_angular_velocity; // from IMU

// Inertia parameters (kg*m^2) - set reasonable defaults, can be tuned via ROS params later
double I_x = 0.02; // roll inertia
double I_y = 0.02; // pitch inertia
double I_z = 0.04; // yaw inertia

// PID控制器参数
const double Kp_x = 1;  // 位置比例增益
const double Ki_x = 0.05; // 积分增益
const double Kd_x = 0.5;  // 速度微分增益

const double Kp_y = 1;
const double Ki_y = 0.05;
const double Kd_y = 0.5;

const double Kp_z = 6;
const double Ki_z = 0.5;
const double Kd_z = 2;

static double integral_x = 0, integral_y = 0, integral_z = 0;
static double prev_error_x = 0, prev_error_y = 0, prev_error_z = 0;

// PID 参数（与原始值等效）
class PID3 {
public:
    PID3(double p, double i, double d) : Kp(p), Ki(i), Kd(d), integral(0), prev_error(0) {}

    double update(double error, double dt) {
        integral += error * dt;
        double derivative = (error - prev_error) / dt;
        prev_error = error;
        return Kp * error + Ki * integral + Kd * derivative;
    }

private:
    double Kp, Ki, Kd;
    double integral, prev_error;
};

PID3 pid_x(1.0, 0.05, 0.5);
PID3 pid_y(1.0, 0.05, 0.5);
PID3 pid_z(6.0, 0.5, 2.0);

// 角度->角速度 比例增益（用于从角度误差生成期望角速度）
const double Kp_angle_roll = 4.0;
const double Kp_angle_pitch = 4.0;
const double Kp_angle_yaw = 1.5;

// 对角速度进行 PID 控制（输出直接作为 body_rate 命令）
// Note: 期望角速度由角度误差乘以上面的 Kp_angle_* 得到（不读取 PX4 发布的 body_rate）
PID3 pid_rate_roll(3.5, 0.01, 0.15);
PID3 pid_rate_pitch(3.5, 0.01, 0.15);
PID3 pid_rate_yaw(1.2, 0.005, 0.08);

// desired attitude (read from mavros official topic)
static mavros_msgs::AttitudeTarget desired_att;
static bool have_desired_att = false;

void desired_att_cb(const mavros_msgs::AttitudeTarget::ConstPtr& msg) {
    desired_att = *msg;
    have_desired_att = true;
}

// Attitude PID gains (angle Kp, integral, and rate D)
const double Kp_att_roll = 6.0;
const double Ki_att_roll = 0.01;
const double Kd_rate_roll = 0.1;
const double Kp_att_pitch = 6.0;
const double Ki_att_pitch = 0.01;
const double Kd_rate_pitch = 0.1;
const double Kp_att_yaw = 2.5;
const double Ki_att_yaw = 0.005;
const double Kd_rate_yaw = 0.05;

double integral_att_x = 0.0, integral_att_y = 0.0, integral_att_z = 0.0;
double prev_error_att_x = 0.0, prev_error_att_y = 0.0, prev_error_att_z = 0.0;

void state_cb(const mavros_msgs::State::ConstPtr& msg) {
    current_state = *msg;
}

void local_pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    local_position = *msg;
    tf::quaternionMsgToTF(local_position.pose.orientation, angle);
    tf::Matrix3x3(angle).getRPY(roll, pitch, yaw);

    // ROS_INFO("X: %f",local_position.pose.position.x);
    // ROS_INFO("Y: %f",local_position.pose.position.y);
    // ROS_INFO("Z: %f",local_position.pose.position.z);
//    ROS_INFO("roll: %f",roll);
//    ROS_INFO("pitch: %f",pitch);
//    ROS_INFO("yaw: %f",yaw);
}

void velocity_cb(const geometry_msgs::TwistStamped::ConstPtr& msg) {
    current_velocity = *msg;
}

void imu_cb(const sensor_msgs::Imu::ConstPtr& msg) { current_angular_velocity = msg->angular_velocity; }

// --- helper utilities ---
static inline double clamp(double v, double lo, double hi) {
    return (v > hi) ? hi : ((v < lo) ? lo : v);
}

// compute small-angle attitude error vector from q_des * q_cur^{-1}
static void quat_error_vector(const tf::Quaternion &q_des, const tf::Quaternion &q_cur,
                              double &ex, double &ey, double &ez) {
    tf::Quaternion q_err = q_des * q_cur.inverse();
    if (q_err.w() < 0.0) q_err = tf::Quaternion(-q_err.x(), -q_err.y(), -q_err.z(), -q_err.w());
    ex = q_err.x(); ey = q_err.y(); ez = q_err.z();
}

static void publish_torque(ros::Publisher &pub, double tx, double ty, double tz) {
    mavros_msgs::ActuatorControl msg;
    msg.header.stamp = ros::Time::now();
    msg.group_mix = 0;
    for (int i = 0; i < 8; ++i) msg.controls[i] = 0.0f;
    msg.controls[0] = static_cast<float>(tx);
    msg.controls[1] = static_cast<float>(ty);
    msg.controls[2] = static_cast<float>(tz);
    pub.publish(msg);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "off_node_pid");
    ros::NodeHandle nh;
    ros::Time::waitForValid(); // 确保ros::Time已初始化

    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>
            ("mavros/state", 10, state_cb);
    ros::Subscriber local_pose = nh.subscribe<geometry_msgs::PoseStamped>
            ("mavros/local_position/pose", 10, local_pose_cb);
    ros::Subscriber velocity_sub = nh.subscribe<geometry_msgs::TwistStamped>
            ("mavros/local_position/velocity_local", 10, velocity_cb);
    ros::Subscriber imu_sub = nh.subscribe<sensor_msgs::Imu>("mavros/imu/data", 10, imu_cb);
    ros::Subscriber desired_att_sub = nh.subscribe<mavros_msgs::AttitudeTarget>
            ("mavros/setpoint_attitude/attitude", 10, desired_att_cb);
    
    // 发布加速度命令（ENU，与 MAVROS/ROS 约定一致）
    ros::Publisher local_accel_pub = nh.advertise<geometry_msgs::Vector3Stamped>("mavros/setpoint_accel/accel", 10);
    
    // NOTE: we do not publish locally-computed desired attitude here.
    // The node reads desired attitude from MAVROS (subscribed above) and
    // computes torque outputs which are published to the actuator topic.
    
    // Publisher for torque setpoint (use MAVROS ActuatorControl, standard topic)
    ros::Publisher torque_pub = nh.advertise<mavros_msgs::ActuatorControl>("mavros/actuator_control", 10);

    ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>
            ("mavros/cmd/arming");
    ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>
            ("mavros/set_mode");

    ros::Rate rate(20.0);

    ros::Time prev_time = ros::Time::now();

    // 等待飞控连接
    while(ros::ok() && !current_state.connected) {
        ros::spinOnce();
        rate.sleep();
    }

    // 目标位置
    geometry_msgs::PoseStamped target_pose;
    target_pose.pose.position.x = 1.0;
    target_pose.pose.position.y = 1.0;
    target_pose.pose.position.z = 2.0;

    // 初始化加速度命令消息（geometry_msgs::Vector3Stamped，ENU）
    geometry_msgs::Vector3Stamped accel_msg;
    accel_msg.header.frame_id = "map"; // 与 local_position.pose.header.frame_id 保持一致
    accel_msg.vector.x = 0.0;
    accel_msg.vector.y = 0.0;
    accel_msg.vector.z = 0.0;

    // 发送初始命令建立连接
    for(int i = 100; ros::ok() && i > 0; --i) {
        accel_msg.header.stamp = ros::Time::now();
        local_accel_pub.publish(accel_msg);
        ros::spinOnce();
        rate.sleep();
    }

    mavros_msgs::SetMode offb_set_mode;
    offb_set_mode.request.custom_mode = "OFFBOARD";

    mavros_msgs::CommandBool arm_cmd;
    arm_cmd.request.value = true;

    ros::Time last_request = ros::Time::now();

    while(ros::ok()) {
        if(current_state.mode != "OFFBOARD" &&
           (ros::Time::now() - last_request > ros::Duration(5.0))) {
            if(set_mode_client.call(offb_set_mode) &&
               offb_set_mode.response.mode_sent) {
                ROS_INFO("Offboard enabled");
            }
            last_request = ros::Time::now();
        } else {
            if(!current_state.armed &&
               (ros::Time::now() - last_request > ros::Duration(5.0))) {
                if(arming_client.call(arm_cmd) &&
                   arm_cmd.response.success) {
                    ROS_INFO("Vehicle armed");
                }
                last_request = ros::Time::now();
            }
        }

        ros::Time now = ros::Time::now();
        double dt = (now - prev_time).toSec();
        if(dt == 0) dt = 0.05;
        prev_time = now;
        // PID控制器计算加速度
        double error_x = target_pose.pose.position.x - local_position.pose.position.x;
        double error_y = target_pose.pose.position.y - local_position.pose.position.y;
        double error_z = target_pose.pose.position.z - local_position.pose.position.z;
        double error_vx = 0 - current_velocity.twist.linear.x;
        double error_vy = 0 - current_velocity.twist.linear.y;
        double error_vz = 0 - current_velocity.twist.linear.z;
        // 积分项累加
        integral_x += error_x * dt;
        integral_y += error_y * dt;
        integral_z += error_z * dt;
        // 微分项
        double derivative_x = (error_x - prev_error_x) / dt;
        double derivative_y = (error_y - prev_error_y) / dt;
        double derivative_z = (error_z - prev_error_z) / dt;
        prev_error_x = error_x;
        prev_error_y = error_y;
        prev_error_z = error_z;
    accel_msg.header.stamp = ros::Time::now();
    accel_msg.vector.x = Kp_x * error_x + Ki_x * integral_x + Kd_x * error_vx;
    accel_msg.vector.y = Kp_y * error_y + Ki_y * integral_y + Kd_y * error_vy;
    accel_msg.vector.z = Kp_z * error_z + Ki_z * integral_z + Kd_z * error_vz;
    // 注意：此处使用 ENU（z 向上），PID 计算的加速度直接发布，不再在这里对重力做手工补偿。
    // 发布加速度命令（geometry_msgs::AccelStamped，ENU）
    local_accel_pub.publish(accel_msg);

        // 我们不在此处根据期望加速度本地计算期望姿态。
        // 位置环只发布期望加速度；姿态环将使用 MAVROS 提供的期望姿态作为参考，由下方代码
        // 使用四元数误差计算并发布三轴力矩。
        if (have_desired_att) {
            // 使用四元数计算姿态误差：q_error = q_desired * q_current^{-1}
            tf::Quaternion q_desired(desired_att.orientation.x, desired_att.orientation.y, desired_att.orientation.z, desired_att.orientation.w);
            tf::Quaternion q_current;
            tf::quaternionMsgToTF(local_position.pose.orientation, q_current);
            tf::Quaternion q_error = q_desired * q_current.inverse();

            // 保证取最短旋转：若 w < 0, 取负值（等价的四元数）
            if (q_error.w() < 0.0) {
                q_error.setX(-q_error.x());
                q_error.setY(-q_error.y());
                q_error.setZ(-q_error.z());
                q_error.setW(-q_error.w());
            }

            // 使用误差四元数的矢量部分作为小角度误差表示（对于小角度近似，vec(q_error) ≈ 0.5 * angle * axis）
            double err_x = q_error.x();
            double err_y = q_error.y();
            double err_z = q_error.z();

            // 积分项更新并限幅
            integral_att_x += err_x * dt;
            integral_att_y += err_y * dt;
            integral_att_z += err_z * dt;
            const double max_integral_att = 0.5;
            if (integral_att_x > max_integral_att) integral_att_x = max_integral_att;
            if (integral_att_x < -max_integral_att) integral_att_x = -max_integral_att;
            if (integral_att_y > max_integral_att) integral_att_y = max_integral_att;
            if (integral_att_y < -max_integral_att) integral_att_y = -max_integral_att;
            if (integral_att_z > max_integral_att) integral_att_z = max_integral_att;
            if (integral_att_z < -max_integral_att) integral_att_z = -max_integral_att;

            // 将姿态误差映射为期望角速度：des_rate = Kp_angle * err_vector
            double des_rate_x = Kp_angle_roll * err_x;
            double des_rate_y = Kp_angle_pitch * err_y;
            double des_rate_z = Kp_angle_yaw * err_z;

            // 角速度误差（使用 IMU 角速度作为测量值）
            double rate_err_x = des_rate_x - current_angular_velocity.x;
            double rate_err_y = des_rate_y - current_angular_velocity.y;
            double rate_err_z = des_rate_z - current_angular_velocity.z;

            // 基于姿态误差 + 角速度误差的 PID 输出为三轴力矩 (N·m)
            double torque_x = Kp_att_roll * err_x + Ki_att_roll * integral_att_x + Kd_rate_roll * rate_err_x;
            double torque_y = Kp_att_pitch * err_y + Ki_att_pitch * integral_att_y + Kd_rate_pitch * rate_err_y;
            double torque_z = Kp_att_yaw * err_z + Ki_att_yaw * integral_att_z + Kd_rate_yaw * rate_err_z;

            // 限制 torque 输出范围，避免过大力矩
            const double max_torque = 1.0; // [N*m], 视平台而定可调整
            if (torque_x > max_torque) torque_x = max_torque;
            if (torque_x < -max_torque) torque_x = -max_torque;
            if (torque_y > max_torque) torque_y = max_torque;
            if (torque_y < -max_torque) torque_y = -max_torque;
            if (torque_z > max_torque) torque_z = max_torque;
            if (torque_z < -max_torque) torque_z = -max_torque;

            // 发布三轴力矩到 MAVROS actuator 控制话题
            mavros_msgs::ActuatorControl torque_msg2;
            torque_msg2.header.stamp = ros::Time::now();
            torque_msg2.group_mix = 0;
            for (int i = 0; i < 8; ++i) torque_msg2.controls[i] = 0.0f;
            torque_msg2.controls[0] = static_cast<float>(torque_x);
            torque_msg2.controls[1] = static_cast<float>(torque_y);
            torque_msg2.controls[2] = static_cast<float>(torque_z);
            torque_pub.publish(torque_msg2);
        }

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}