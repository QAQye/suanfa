#include <ros/ros.h>
#include <pcl/common/transforms.h>
#include <glog/logging.h>
#include <fstream>
#include <iomanip>
#include <cstdlib> // system()
#include <sys/stat.h> // mkdir()

#include "lio_ndt/global_defination/global_defination.h.in"
#include "lio_ndt/subscriber/cloud_subscriber.hpp"
#include "lio_ndt/subscriber/imu_subscriber.hpp"
#include "lio_ndt/subscriber/gnss_subscriber.hpp"
#include "lio_ndt/tf_listener/tf_listener.hpp"
#include "lio_ndt/publisher/cloud_publisher.hpp"
#include "lio_ndt/publisher/odometry_publisher.hpp"
#include "lio_ndt/front_end/front_end.hpp"

using namespace lio_ndt;

// 辅助函数：创建结果保存目录（兼容老系统）
void CreateDirIfNotExists(const std::string &dir_path)
{
    struct stat info;
    if (stat(dir_path.c_str(), &info) != 0)
    {
        std::string cmd = "mkdir -p " + dir_path;
        system(cmd.c_str());
        ROS_INFO_STREAM("Created directory: " << dir_path);
    }
    else
    {
        ROS_INFO_STREAM("Directory already exists: " << dir_path);
    }
}

int main(int argc, char *argv[])
{
    google::InitGoogleLogging(argv[0]);
    // FLAGS_log_dir = WORK_SPACE_PATH + "/Log";
    FLAGS_alsologtostderr = 1;
    FLAGS_logtostderr = 1;

    ros::init(argc, argv, "front_end_node");
    ros::NodeHandle nh;

    //========================= 订阅与发布 =========================//
    std::shared_ptr<CloudSubscriber> cloud_sub_ptr = std::make_shared<CloudSubscriber>(nh, "/kitti/velo/pointcloud", 100000);
    std::shared_ptr<IMUSubscriber> imu_sub_ptr = std::make_shared<IMUSubscriber>(nh, "/kitti/oxts/imu", 1000000);
    std::shared_ptr<GNSSSubscriber> gnss_sub_ptr = std::make_shared<GNSSSubscriber>(nh, "/kitti/oxts/gps/fix", 1000000);

    std::shared_ptr<TFListener> imu_to_lidar_ptr = std::make_shared<TFListener>(nh, "velo_link", "imu_link");

    std::shared_ptr<CloudPublisher> cloud_pub_ptr = std::make_shared<CloudPublisher>(nh, "current_scan", 100, "world");
    std::shared_ptr<CloudPublisher> local_map_pub_ptr = std::make_shared<CloudPublisher>(nh, "local_map", 100, "world");
    std::shared_ptr<CloudPublisher> global_map_pub_ptr = std::make_shared<CloudPublisher>(nh, "global_map", 100, "world");
    std::shared_ptr<OdometryPublisher> laser_odom_pub_ptr = std::make_shared<OdometryPublisher>(nh, "laser_odom", "world", "lidar", 100);
    std::shared_ptr<OdometryPublisher> gnss_pub_ptr = std::make_shared<OdometryPublisher>(nh, "gnss", "world", "lidar", 100);

    //========================= 前端初始化 =========================//
    std::shared_ptr<FrontEnd> front_end_ptr = std::make_shared<FrontEnd>();

    //========================= 轨迹保存 =========================//
    std::string result_dir ="/home/gec";
    CreateDirIfNotExists(result_dir);

    std::string traj_file_path = result_dir + "/trajectory_ndt.txt"; // 你可以改为 trajectory_icp.txt
    std::ofstream traj_file(traj_file_path);
    traj_file << std::fixed << std::setprecision(6);
    ROS_INFO_STREAM("Trajectory will be saved to: " << traj_file_path);

    //========================= 数据缓存 =========================//
    std::deque<CloudData> cloud_data_buff;
    std::deque<IMUData> imu_data_buff;
    std::deque<GNSSData> gnss_data_buff;

    Eigen::Matrix4f imu_to_lidar = Eigen::Matrix4f::Identity();
    bool transform_received = false;
    bool gnss_origin_position_inited = false;
    bool front_end_pose_inited = false;

    CloudData::CLOUD_PTR local_map_ptr(new CloudData::CLOUD());
    CloudData::CLOUD_PTR global_map_ptr(new CloudData::CLOUD());
    CloudData::CLOUD_PTR current_scan_ptr(new CloudData::CLOUD());

    double run_time = 0.0;
    double init_time = 0.0;
    bool time_inited = false;
    bool has_global_map_published = false;

    ros::Rate rate(100);

    //========================= 主循环 =========================//
    while (ros::ok())
    {
        ros::spinOnce();

        cloud_sub_ptr->ParseData(cloud_data_buff);
        imu_sub_ptr->ParseData(imu_data_buff);
        gnss_sub_ptr->ParseData(gnss_data_buff);

        if (!transform_received)
        {
            if (imu_to_lidar_ptr->LookupData(imu_to_lidar))
                transform_received = true;
        }
        else
        {
            while (cloud_data_buff.size() > 0 && imu_data_buff.size() > 0 && gnss_data_buff.size() > 0)
            {
                CloudData cloud_data = cloud_data_buff.front();
                IMUData imu_data = imu_data_buff.front();
                GNSSData gnss_data = gnss_data_buff.front();

                if (!time_inited)
                {
                    time_inited = true;
                    init_time = cloud_data.time;
                }
                else
                {
                    run_time = cloud_data.time - init_time;
                }

                double d_time = cloud_data.time - imu_data.time;
                if (d_time < -0.05)
                    cloud_data_buff.pop_front();
                else if (d_time > 0.05)
                {
                    imu_data_buff.pop_front();
                    gnss_data_buff.pop_front();
                }
                else
                {
                    cloud_data_buff.pop_front();
                    imu_data_buff.pop_front();
                    gnss_data_buff.pop_front();

                    Eigen::Matrix4f odometry_matrix = Eigen::Matrix4f::Identity();
                    if (!gnss_origin_position_inited)
                    {
                        gnss_data.InitOriginPosition();
                        gnss_origin_position_inited = true;
                    }
                    gnss_data.UpdataXYZ();
                    odometry_matrix(0, 3) = gnss_data.local_E;
                    odometry_matrix(1, 3) = gnss_data.local_N;
                    odometry_matrix(2, 3) = gnss_data.local_U;
                    odometry_matrix.block<3, 3>(0, 0) = imu_data.GetOrientationMatrix();
                    odometry_matrix *= imu_to_lidar.inverse();

                    gnss_pub_ptr->Publish(odometry_matrix);

                    if (!front_end_pose_inited)
                    {
                        front_end_pose_inited = true;
                        front_end_ptr->SetInitPose(odometry_matrix);
                    }

                    Eigen::Matrix4f laser_matrix = front_end_ptr->Update(cloud_data);
                    laser_odom_pub_ptr->Publish(laser_matrix);

                    //===== ✅ 保存轨迹到文件（兼容 evo） =====//
                    Eigen::Quaternionf q(laser_matrix.block<3, 3>(0, 0));
                    traj_file << cloud_data.time << " "
                              << laser_matrix(0, 3) << " " << laser_matrix(1, 3) << " " << laser_matrix(2, 3) << " "
                              << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";

                    front_end_ptr->GetCurrentScan(current_scan_ptr);
                    cloud_pub_ptr->Publish(current_scan_ptr);
                    if (front_end_ptr->GetNewLocalMap(local_map_ptr))
                        local_map_pub_ptr->Publish(local_map_ptr);

                    if (run_time > 30.0 && !has_global_map_published)
                    {
                        if (front_end_ptr->GetNewGlobalMap(global_map_ptr))
                        {
                            global_map_pub_ptr->Publish(global_map_ptr);
                            // has_global_map_published = true;
                        }
                    }
                }
            }
        }
        rate.sleep();
    }

    traj_file.close();
    ROS_INFO("Trajectory saved successfully!");
    return 0;
}
