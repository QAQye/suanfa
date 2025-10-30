#pragma once

#include <deque>
#include <memory>
#include <Eigen/Dense>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>

#include <lio_ndt/sensor_data/cloud_data.hpp>
#include "lio_ndt/method/optimized_ICP_GN.h"
#include "lio_ndt/method/common.h"

// 新增的匹配方法接口与实现头文件
#include "lio_ndt/method/registration_method.h"
#include "lio_ndt/method/pcl_icp_method.h"
#include "lio_ndt/method/pcl_ndt_method.h"

namespace lio_ndt
{
    class FrontEnd
    {
    public:
        class Frame
        {
        public:
            Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
            CloudData cloud_data;
        };

    public:
        FrontEnd();

        /**
         * @brief 主更新函数，输入当前帧点云，返回当前帧姿态（4x4矩阵）
         */
        Eigen::Matrix4f Update(const CloudData &cloud_data);

        /**
         * @brief 设置初始位姿（通常来自GNSS或手动定义）
         */
        bool SetInitPose(const Eigen::Matrix4f &init_pose);

        /**
         * @brief 设置预测位姿（例如由IMU或运动模型预测）
         */
        bool SetPredictPose(const Eigen::Matrix4f &predict_pose);

        /**
         * @brief 获取最新的局部地图（滑窗地图）
         */
        bool GetNewLocalMap(CloudData::CLOUD_PTR &local_map_ptr);

        /**
         * @brief 获取最新的全局地图
         */
        bool GetNewGlobalMap(CloudData::CLOUD_PTR &global_map_ptr);

        /**
         * @brief 获取当前帧的匹配点云（用于rviz显示）
         */
        bool GetCurrentScan(CloudData::CLOUD_PTR &current_scan_ptr);

    private:
        /**
         * @brief 更新关键帧、新建局部地图、更新目标点云
         */
        void UpdateNewFrame(const Frame &new_key_frame);

    private:
        // ============== 匹配器部分（新增） ==============
        /// 通用配准接口指针（PCL ICP / PCL NDT / OptimizedICP）
        std::shared_ptr<RegistrationMethod> matcher_;

        /// 保留原OptimizedICP对象，用于默认算法或兼容旧代码
        std::shared_ptr<OptimizedICPGN> opti_icp_;

        // ============== 滤波器部分 ==============
        pcl::VoxelGrid<CloudData::POINT> cloud_filter_;       // 当前帧滤波
        pcl::VoxelGrid<CloudData::POINT> local_map_filter_;   // 局部地图滤波
        pcl::VoxelGrid<CloudData::POINT> display_filter_;     // 显示滤波

        // ============== 地图与关键帧 ==============
        std::deque<Frame> local_map_frames_;   // 滑动窗口内关键帧
        std::deque<Frame> global_map_frames_;  // 全局关键帧

        bool has_new_local_map_ = false;
        bool has_new_global_map_ = false;

        CloudData::CLOUD_PTR local_map_ptr_;   // 当前局部地图
        CloudData::CLOUD_PTR global_map_ptr_;  // 全局地图
        CloudData::CLOUD_PTR result_cloud_ptr_; // 当前帧匹配结果点云

        Frame current_frame_; // 当前帧信息

        // ============== 位姿管理 ==============
        Eigen::Matrix4f init_pose_ = Eigen::Matrix4f::Identity();     // 初始位姿
        Eigen::Matrix4f predict_pose_ = Eigen::Matrix4f::Identity();  // 预测位姿
    };
}
