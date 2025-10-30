#pragma once

#include <deque>
#include <memory>
#include <Eigen/Dense>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>

#include <lio_ndt/sensor_data/cloud_data.hpp>
#include "lio_ndt/method/optimized_ICP_GN.h"
#include "lio_ndt/method/common.h"

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

        // 算法选择枚举
        enum MatchMethod { GN_ICP=0, PCL_ICP=1, PCL_NDT=2 };

    public:
        FrontEnd();

        Eigen::Matrix4f Update(const CloudData &cloud_data);

        bool SetInitPose(const Eigen::Matrix4f &init_pose);
        bool SetPredictPose(const Eigen::Matrix4f &predict_pose);

        bool GetNewLocalMap(CloudData::CLOUD_PTR &local_map_ptr);
        bool GetNewGlobalMap(CloudData::CLOUD_PTR &global_map_ptr);
        bool GetCurrentScan(CloudData::CLOUD_PTR &current_scan_ptr);

        // 设置匹配算法
        void SetMatchMethod(MatchMethod method);

    private:
        void UpdateNewFrame(const Frame &new_key_frame);

    private:
        // ============== 匹配器 ==============
        OptimizedICPGN icp_opti_;   // 原始 GN-ICP
        pcl::IterativeClosestPoint<CloudData::POINT, CloudData::POINT> pcl_icp_;
        pcl::NormalDistributionsTransform<CloudData::POINT, CloudData::POINT> pcl_ndt_;

        MatchMethod match_method_ = GN_ICP;

        // ============== 滤波器 ==============
        pcl::VoxelGrid<CloudData::POINT> cloud_filter_;
        pcl::VoxelGrid<CloudData::POINT> local_map_filter_;
        pcl::VoxelGrid<CloudData::POINT> display_filter_;

        // ============== 地图与关键帧 ==============
        std::deque<Frame> local_map_frames_;
        std::deque<Frame> global_map_frames_;

        bool has_new_local_map_ = false;
        bool has_new_global_map_ = false;

        CloudData::CLOUD_PTR local_map_ptr_;
        CloudData::CLOUD_PTR global_map_ptr_;
        CloudData::CLOUD_PTR result_cloud_ptr_;

        Frame current_frame_;

        // ============== 位姿管理 ==============
        Eigen::Matrix4f init_pose_ = Eigen::Matrix4f::Identity();
        Eigen::Matrix4f predict_pose_ = Eigen::Matrix4f::Identity();
    };
}
