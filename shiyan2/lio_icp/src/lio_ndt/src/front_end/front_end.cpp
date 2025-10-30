#include "lio_ndt/front_end/front_end.hpp"
#include <pcl/common/transforms.h>
#include <iostream>

namespace lio_ndt
{
    FrontEnd::FrontEnd() : icp_opti_(), local_map_ptr_(new CloudData::CLOUD()),
                           global_map_ptr_(new CloudData::CLOUD()),
                           result_cloud_ptr_(new CloudData::CLOUD()),
                           match_method_(PCL_NDT)
    {
        // GN-ICP 默认参数
        icp_opti_.SetMaxCorrespondDistance(1);
        icp_opti_.SetMaxIterations(2);
        icp_opti_.SetTransformationEpsilon(0.5);

        // 滤波器
        cloud_filter_.setLeafSize(1.5f,1.5f,1.5f);
        local_map_filter_.setLeafSize(1.0f,1.0f,1.0f);
        display_filter_.setLeafSize(1.0f,1.0f,1.0f);

        // PCL ICP
        pcl_icp_.setMaxCorrespondenceDistance(1.0);
        pcl_icp_.setMaximumIterations(2);
        pcl_icp_.setTransformationEpsilon(0.5);

        // PCL NDT
        pcl_ndt_.setResolution(1.0);
        pcl_ndt_.setMaximumIterations(2);
        pcl_ndt_.setTransformationEpsilon(0.5);
    }

    void FrontEnd::SetMatchMethod(MatchMethod method)
    {
        match_method_ = method;
    }

    Eigen::Matrix4f FrontEnd::Update(const CloudData &cloud_data)
    {
        current_frame_.cloud_data.time = cloud_data.time;
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloud_data.cloud_ptr, *current_frame_.cloud_data.cloud_ptr, indices);

        CloudData::CLOUD_PTR filtered_cloud_ptr(new CloudData::CLOUD());
        cloud_filter_.setInputCloud(current_frame_.cloud_data.cloud_ptr);
        cloud_filter_.filter(*filtered_cloud_ptr);

        static Eigen::Matrix4f step_pose = Eigen::Matrix4f::Identity();
        static Eigen::Matrix4f last_pose = init_pose_;
        static Eigen::Matrix4f predict_pose_local = init_pose_;
        static Eigen::Matrix4f last_key_frame_pose = init_pose_;

        if(local_map_frames_.empty())
        {
            current_frame_.pose = init_pose_;
            UpdateNewFrame(current_frame_);
            return current_frame_.pose;
        }

        // 匹配
        if(match_method_ == GN_ICP)
        {
            icp_opti_.Match(filtered_cloud_ptr, predict_pose_local, result_cloud_ptr_, current_frame_.pose);
            std::cout << "GN-ICP fitness score: " << icp_opti_.GetFitnessScore() << std::endl;
        }
        else if(match_method_ == PCL_ICP)
        {
            pcl_icp_.setInputSource(filtered_cloud_ptr);
            pcl_icp_.setInputTarget(local_map_ptr_);
            pcl::PointCloud<CloudData::POINT> aligned_cloud;
            pcl_icp_.align(aligned_cloud, predict_pose_local);
            current_frame_.pose = pcl_icp_.getFinalTransformation();
            *result_cloud_ptr_ = aligned_cloud;
            std::cout << "PCL-ICP converged: " << pcl_icp_.hasConverged()
                      << " score: " << pcl_icp_.getFitnessScore() << std::endl;
        }
        else if(match_method_ == PCL_NDT)
        {
            pcl_ndt_.setInputSource(filtered_cloud_ptr);
            pcl_ndt_.setInputTarget(local_map_ptr_);
            pcl::PointCloud<CloudData::POINT> aligned_cloud;
            pcl_ndt_.align(aligned_cloud, predict_pose_local);
            current_frame_.pose = pcl_ndt_.getFinalTransformation();
            *result_cloud_ptr_ = aligned_cloud;
            std::cout << "PCL-NDT converged: " << pcl_ndt_.hasConverged()
                      << " score: " << pcl_ndt_.getFitnessScore() << std::endl;
        }

        // 位姿预测更新
        step_pose = last_pose.inverse() * current_frame_.pose;
        predict_pose_local = current_frame_.pose * step_pose;
        last_pose = current_frame_.pose;

        if ((last_key_frame_pose.block<3,1>(0,3) - current_frame_.pose.block<3,1>(0,3)).norm() > 2.0)
        {
            UpdateNewFrame(current_frame_);
            last_key_frame_pose = current_frame_.pose;
        }

        return current_frame_.pose;
    }

    bool FrontEnd::SetInitPose(const Eigen::Matrix4f &init_pose)
    {
        init_pose_ = init_pose;
        return true;
    }

    bool FrontEnd::SetPredictPose(const Eigen::Matrix4f &predict_pose)
    {
        predict_pose_ = predict_pose;
        return true;
    }

    void FrontEnd::UpdateNewFrame(const Frame &new_key_frame)
    {
        Frame key_frame = new_key_frame;
        key_frame.cloud_data.cloud_ptr.reset(new CloudData::CLOUD(*new_key_frame.cloud_data.cloud_ptr));
        CloudData::CLOUD_PTR transformed_cloud_ptr(new CloudData::CLOUD());

        // 更新局部地图
        local_map_frames_.push_back(key_frame);
        while(local_map_frames_.size() > 20) local_map_frames_.pop_front();

        local_map_ptr_.reset(new CloudData::CLOUD());
        for(size_t i=0;i<local_map_frames_.size();i++)
        {
            pcl::transformPointCloud(*local_map_frames_[i].cloud_data.cloud_ptr, *transformed_cloud_ptr, local_map_frames_[i].pose);
            *local_map_ptr_ += *transformed_cloud_ptr;
        }
        has_new_local_map_ = true;

        // 更新GN-ICP目标点云
        if(match_method_ == GN_ICP)
        {
            if(local_map_frames_.size() < 10)
                icp_opti_.SetTargetCloud(local_map_ptr_);
            else
            {
                CloudData::CLOUD_PTR filtered_local_map_ptr(new CloudData::CLOUD());
                local_map_filter_.setInputCloud(local_map_ptr_);
                local_map_filter_.filter(*filtered_local_map_ptr);
                icp_opti_.SetTargetCloud(filtered_local_map_ptr);
            }
        }

        // 更新全局地图
        global_map_frames_.push_back(key_frame);
        if(global_map_frames_.size() % 100 != 0) return;

        global_map_ptr_.reset(new CloudData::CLOUD());
        for(size_t i=0;i<global_map_frames_.size();i++)
        {
            pcl::transformPointCloud(*global_map_frames_[i].cloud_data.cloud_ptr, *transformed_cloud_ptr, global_map_frames_[i].pose);
            *global_map_ptr_ += *transformed_cloud_ptr;
        }
        has_new_global_map_ = true;
    }

    bool FrontEnd::GetNewLocalMap(CloudData::CLOUD_PTR &local_map_ptr)
    {
        if(!has_new_local_map_) return false;
        display_filter_.setInputCloud(local_map_ptr_);
        display_filter_.filter(*local_map_ptr);
        return true;
    }

    bool FrontEnd::GetNewGlobalMap(CloudData::CLOUD_PTR &global_map_ptr)
    {
        if(!has_new_global_map_) return false;
        display_filter_.setInputCloud(global_map_ptr_);
        display_filter_.filter(*global_map_ptr);
        return true;
    }

    bool FrontEnd::GetCurrentScan(CloudData::CLOUD_PTR &current_scan_ptr)
    {
        display_filter_.setInputCloud(result_cloud_ptr_);
        display_filter_.filter(*current_scan_ptr);
        return true;
    }
}
