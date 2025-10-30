#include <ros/ros.h>
#include <pcl/common/transforms.h>
#include <glog/logging.h>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <sys/stat.h>

#include "lio_ndt/global_defination/global_defination.h.in"
#include "lio_ndt/subscriber/cloud_subscriber.hpp"
#include "lio_ndt/subscriber/imu_subscriber.hpp"
#include "lio_ndt/tf_listener/tf_listener.hpp"
#include "lio_ndt/publisher/cloud_publisher.hpp"
#include "lio_ndt/publisher/odometry_publisher.hpp"
#include "lio_ndt/front_end/front_end.hpp"

using namespace lio_ndt;

// 创建保存目录
void CreateDirIfNotExists(const std::string &dir_path) {
    struct stat info;
    if (stat(dir_path.c_str(), &info) != 0) {
        std::string cmd = "mkdir -p " + dir_path;
        system(cmd.c_str());
        ROS_INFO_STREAM("Created directory: " << dir_path);
    }
}

int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = 1;
    FLAGS_logtostderr = 1;

    ros::init(argc, argv, "front_end_node");
    ros::NodeHandle nh;

    //========================= 订阅与发布 =========================//
    std::shared_ptr<CloudSubscriber> cloud_sub_ptr = std::make_shared<CloudSubscriber>(nh, "/rslidar_points", 100000);
    std::shared_ptr<IMUSubscriber> imu_sub_ptr = std::make_shared<IMUSubscriber>(nh, "/imu/data_raw", 1000000);

    std::shared_ptr<TFListener> imu_to_lidar_ptr = std::make_shared<TFListener>(nh, "velo_link", "imu_link");

    std::shared_ptr<CloudPublisher> cloud_pub_ptr = std::make_shared<CloudPublisher>(nh, "current_scan", 100, "world");
    std::shared_ptr<CloudPublisher> local_map_pub_ptr = std::make_shared<CloudPublisher>(nh, "local_map", 100, "world");
    std::shared_ptr<CloudPublisher> global_map_pub_ptr = std::make_shared<CloudPublisher>(nh, "global_map", 100, "world");
    std::shared_ptr<OdometryPublisher> laser_odom_pub_ptr = std::make_shared<OdometryPublisher>(nh, "laser_odom", "world", "lidar", 100);

    //========================= 前端初始化 =========================//
    std::shared_ptr<FrontEnd> front_end_ptr = std::make_shared<FrontEnd>();

    //========================= 轨迹保存 =========================//
    std::string result_dir = "/home/gec/shiyan2zuizhong";
    CreateDirIfNotExists(result_dir);

    std::string traj_file_path = result_dir + "/trajectory_lidar_gn_icp.txt";
    std::ofstream traj_file(traj_file_path);
    traj_file << std::fixed << std::setprecision(6);
    if (!traj_file.is_open()) {
        ROS_ERROR_STREAM("Failed to open file: " << traj_file_path);
        return -1;
    }
    ROS_INFO_STREAM("Trajectory will be saved to: " << traj_file_path);

    //========================= 缓冲区 =========================//
    std::deque<CloudData> cloud_data_buff;
    std::deque<IMUData> imu_data_buff;

    Eigen::Matrix4f imu_to_lidar = Eigen::Matrix4f::Identity();
    bool transform_received = false;
    bool front_end_pose_inited = false;

    CloudData::CLOUD_PTR local_map_ptr(new CloudData::CLOUD());
    CloudData::CLOUD_PTR global_map_ptr(new CloudData::CLOUD());
    CloudData::CLOUD_PTR current_scan_ptr(new CloudData::CLOUD());

    ros::Rate rate(100);

    while (ros::ok()) {
        ros::spinOnce();

        cloud_sub_ptr->ParseData(cloud_data_buff);
        imu_sub_ptr->ParseData(imu_data_buff);

        // 获取 TF
        if (!transform_received) {
            if (imu_to_lidar_ptr->LookupData(imu_to_lidar)) {
                transform_received = true;
            }
        }

        if (cloud_data_buff.empty()) {
            rate.sleep();
            continue;
        }

        CloudData cloud_data = cloud_data_buff.front();
        cloud_data_buff.pop_front();

        if (!front_end_pose_inited) {
            front_end_pose_inited = true;
            front_end_ptr->SetInitPose(Eigen::Matrix4f::Identity());
            ROS_INFO("FrontEnd initialized with Identity pose.");
        }

        // 执行前端更新
        Eigen::Matrix4f laser_matrix = front_end_ptr->Update(cloud_data);
        laser_odom_pub_ptr->Publish(laser_matrix);

        // 保存轨迹
        Eigen::Quaternionf q(laser_matrix.block<3, 3>(0, 0));
        traj_file << cloud_data.time << " "
                  << laser_matrix(0, 3) << " " << laser_matrix(1, 3) << " " << laser_matrix(2, 3) << " "
                  << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
        traj_file.flush();

        // 发布点云地图
        front_end_ptr->GetCurrentScan(current_scan_ptr);
        cloud_pub_ptr->Publish(current_scan_ptr);

        if (front_end_ptr->GetNewLocalMap(local_map_ptr))
            local_map_pub_ptr->Publish(local_map_ptr);

        rate.sleep();
    }

    traj_file.close();
    ROS_INFO("Trajectory saved successfully!");
    return 0;
}
