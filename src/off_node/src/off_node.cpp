//#include "tf/LinearMath/Matrix3x3.h"
//#include "tf/LinearMath/Quaternion.h"
//#include "tf/transform_datatypes.h"
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/AccelStamped.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <tf/transform_listener.h>
#include <nav_msgs/Odometry.h>
#include <string>

double roll,pitch,yaw;
tf::Quaternion angle;

mavros_msgs::State current_state;
void state_cb(const mavros_msgs::State::ConstPtr& msg)
{
    current_state = *msg;
}

geometry_msgs::PoseStamped local_position;
void local_pose_cb(const geometry_msgs::PoseStamped::ConstPtr& msg)
{
    local_position = *msg;
    tf::quaternionMsgToTF(local_position.pose.orientation, angle);
    tf::Matrix3x3(angle).getRPY(roll, pitch, yaw);

    ROS_INFO("X: %f",local_position.pose.position.x);
    ROS_INFO("Y: %f",local_position.pose.position.y);
    ROS_INFO("Z: %f",local_position.pose.position.z);
    ROS_INFO("roll: %f",roll);
    ROS_INFO("pitch: %f",pitch);
    ROS_INFO("yaw: %f",yaw);
}

// PID控制器结构体
struct PID {
    double kp, ki, kd;
    double integral;
    double prev_error;
    PID(double p, double i, double d) : kp(p), ki(i), kd(d), integral(0), prev_error(0) {}
    double update(double error, double dt) {
        integral += error * dt;
        double derivative = (error - prev_error) / dt;
        prev_error = error;
        return kp * error + ki * integral + kd * derivative;
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "offb_node");
    ros::NodeHandle nh;

    ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>
            ("mavros/state", 10, state_cb);
    ros::Subscriber local_pose = nh.subscribe<geometry_msgs::PoseStamped>
            ("mavros/local_position/pose",10,local_pose_cb);
    ros::Publisher accel_pub = nh.advertise<geometry_msgs::AccelStamped>("mavros/setpoint_accel/accel", 10);
    ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>
            ("mavros/cmd/arming");
    ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>
            ("mavros/set_mode");

    //the setpoint publishing rate MUST be faster than 2Hz
    ros::Rate rate(20.0);

    // wait for FCU connection
    while(ros::ok() && !current_state.connected){
        ros::spinOnce();
        rate.sleep();
    }

    // 目标位置
    double target_x = 1.0;
    double target_y = 1.0;
    double target_z = 2.0;
    // PID控制器初始化
    PID pid_x(1.0, 0.0, 0.2); // 参数可调
    PID pid_y(1.0, 0.0, 0.2);
    PID pid_z(1.0, 0.0, 0.2);

    mavros_msgs::SetMode offb_set_mode;
    offb_set_mode.request.custom_mode = "OFFBOARD";

    mavros_msgs::CommandBool arm_cmd;
    arm_cmd.request.value = true;

    ros::Time last_request = ros::Time::now();
    ros::Time prev_time = ros::Time::now();

    while(ros::ok()){
        ros::Time now = ros::Time::now();
        double dt = (now - prev_time).toSec();
        if(dt == 0) dt = 0.05;
        prev_time = now;

        if( current_state.mode != "OFFBOARD" &&
            (now - last_request > ros::Duration(5.0))){
            if( set_mode_client.call(offb_set_mode) &&
                offb_set_mode.response.mode_sent){
                ROS_INFO("Offboard enabled");
            }
            last_request = now;
        } else {
            if( !current_state.armed &&
                (now - last_request > ros::Duration(5.0))){
                if( arming_client.call(arm_cmd) &&
                    arm_cmd.response.success){
                    ROS_INFO("Vehicle armed");
                }
                last_request = now;
            }
        }

        // 计算误差
        double error_x = target_x - local_position.pose.position.x;
        double error_y = target_y - local_position.pose.position.y;
        double error_z = target_z - local_position.pose.position.z;
        // PID计算期望加速度
        double accel_x = pid_x.update(error_x, dt);
        double accel_y = pid_y.update(error_y, dt);
        double accel_z = pid_z.update(error_z, dt);
        geometry_msgs::AccelStamped accel_msg;
        accel_msg.header.stamp = ros::Time::now();
        accel_msg.accel.linear.x = accel_x;
        accel_msg.accel.linear.y = accel_y;
        accel_msg.accel.linear.z = accel_z;
        accel_msg.accel.angular.x = 0;
        accel_msg.accel.angular.y = 0;
        accel_msg.accel.angular.z = 0;
        accel_pub.publish(accel_msg);

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}

