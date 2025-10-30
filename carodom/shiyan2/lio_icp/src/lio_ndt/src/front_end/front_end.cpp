#include <lio_ndt/front_end/front_end.hpp>

#include <cmath>
#include <pcl/common/transforms.h>
#include <glog/logging.h>

#include "lio_ndt/method/registration_method.h"
#include "lio_ndt/method/pcl_icp_method.h"
#include "lio_ndt/method/pcl_ndt_method.h"
// 如果 OptimizedICPGN 没有继承接口，可以写个 Adapter 或保持原样
#include "lio_ndt/method/optimized_ICP_GN.h"

namespace lio_ndt
{
    FrontEnd::FrontEnd() :
        local_map_ptr_(new CloudData::CLOUD()),
        global_map_ptr_(new CloudData::CLOUD()),
        result_cloud_ptr_(new CloudData::CLOUD())
    {
        // ================== 算法选择 ==================
        // 可以通过配置文件或ROS参数指定匹配方法
        // 防止 glog 尝试创建日志文件，直接输出到 stderr
        FLAGS_logtostderr = 1;
        std::string method_type = "gn_icp";   // 默认
        // 若在ROS节点中，可使用 ros::param::get("method_type", method_type);
//        ros::param::get("~method_type", method_type); // 可通过 launch 文件设置
        if (method_type == "pcl_icp") {
            LOG(INFO) << "[FrontEnd] Using PCL ICP Method";
            LOG(INFO) << "**************************************************************************";
            matcher_ = std::make_shared<PCLICPMethod>();
            auto icp = std::dynamic_pointer_cast<PCLICPMethod>(matcher_);
            icp->SetMaxCorrespondenceDistance(1.0);
            icp->SetMaximumIterations(50);
            icp->SetTransformationEpsilon(1e-6);
        }
        else if (method_type == "pcl_ndt") {
            LOG(INFO) << "[FrontEnd] Using PCL NDT Method";
            matcher_ = std::make_shared<PCLNDTMethod>();
            auto ndt = std::dynamic_pointer_cast<PCLNDTMethod>(matcher_);
            ndt->SetResolution(2.0);             // NDT 网格分辨率（单位 m）
            ndt->SetMaximumIterations(50);       // 迭代次数
            ndt->SetTransformationEpsilon(1e-3); // 收敛精度
        }
        else {
            LOG(INFO) << "[FrontEnd] Using Optimized ICP GN Method";
            opti_icp_.reset(new OptimizedICPGN());
            opti_icp_->SetMaxCorrespondDistance(1.0);
            opti_icp_->SetMaxIterations(30);
            opti_icp_->SetTransformationEpsilon(0.01);
        }

        // 体素滤波器参数
        cloud_filter_.setLeafSize(1.5f, 1.5f, 1.5f);
        local_map_filter_.setLeafSize(1.0f, 1.0f, 1.0f);
        display_filter_.setLeafSize(1.0f, 1.0f, 1.0f);
    }

    Eigen::Matrix4f FrontEnd::Update(const CloudData &cloud_data)
    {
        current_frame_.cloud_data.time = cloud_data.time;

        // 去NaN点
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cloud_data.cloud_ptr, *current_frame_.cloud_data.cloud_ptr, indices);

        // 下采样
        CloudData::CLOUD_PTR filtered_cloud_ptr(new CloudData::CLOUD());
        cloud_filter_.setInputCloud(current_frame_.cloud_data.cloud_ptr);
        cloud_filter_.filter(*filtered_cloud_ptr);

        static Eigen::Matrix4f step_pose = Eigen::Matrix4f::Identity();
        static Eigen::Matrix4f last_pose = init_pose_;
        static Eigen::Matrix4f predict_pose = init_pose_;
        static Eigen::Matrix4f last_key_frame_pose = init_pose_;

        if (local_map_frames_.empty()) {
            current_frame_.pose = init_pose_;
            UpdateNewFrame(current_frame_);
            return current_frame_.pose;
        }

        // ================ 匹配 ===================
        bool success = false;
        if (matcher_) {
            success = matcher_->ScanMatch(filtered_cloud_ptr, predict_pose, result_cloud_ptr_, current_frame_.pose);
        } else if (opti_icp_) {
            success = opti_icp_->Match(filtered_cloud_ptr, predict_pose, result_cloud_ptr_, current_frame_.pose);
        }

        if (!success) {
            LOG(WARNING) << "[FrontEnd] ScanMatch failed, using predict pose";
            current_frame_.pose = predict_pose;
        }

        std::cout << "fitness score: ";
        if (matcher_) {
            double score = matcher_->GetFitnessScore();   // 需要 RegistrationMethod 有此虚函数
            if (score < 0) {
                std::cout << "(no score available yet)" << std::endl;
            } else if (!std::isfinite(score)) {
                std::cout << "(not converged / inf)" << std::endl;
            } else {
                std::cout << score << std::endl;
            }
        } else if (opti_icp_) {
            std::cout << opti_icp_->GetFitnessScore() << std::endl;
        } else {
            std::cout << "(no matcher)" << std::endl;
        }

        // 更新预测位姿
        step_pose = last_pose.inverse() * current_frame_.pose;
        predict_pose = current_frame_.pose * step_pose;
        last_pose = current_frame_.pose;

        // 检查是否生成关键帧
        float move_distance = fabs(last_key_frame_pose(0, 3) - current_frame_.pose(0, 3)) +
                              fabs(last_key_frame_pose(1, 3) - current_frame_.pose(1, 3)) +
                              fabs(last_key_frame_pose(2, 3) - current_frame_.pose(2, 3));
        if (move_distance > 2.0) {
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

    void FrontEnd::UpdateNewFrame(const Frame &new_key_frame)
    {
        Frame key_frame = new_key_frame;
        key_frame.cloud_data.cloud_ptr.reset(new CloudData::CLOUD(*new_key_frame.cloud_data.cloud_ptr));

        CloudData::CLOUD_PTR transformed_cloud_ptr(new CloudData::CLOUD());

        local_map_frames_.push_back(key_frame);
        while (local_map_frames_.size() > 20) {
            local_map_frames_.pop_front();
        }

        local_map_ptr_.reset(new CloudData::CLOUD());
        for (auto &frame : local_map_frames_) {
            pcl::transformPointCloud(*frame.cloud_data.cloud_ptr, *transformed_cloud_ptr, frame.pose);
            *local_map_ptr_ += *transformed_cloud_ptr;
        }
        has_new_local_map_ = true;

        // 设置匹配目标点云
        CloudData::CLOUD_PTR target_map(new CloudData::CLOUD());
        if (local_map_frames_.size() < 10) {
            target_map = local_map_ptr_;
        } else {
            local_map_filter_.setInputCloud(local_map_ptr_);
            local_map_filter_.filter(*target_map);
        }

        if (matcher_) matcher_->SetTargetCloud(target_map);
        if (opti_icp_) opti_icp_->SetTargetCloud(target_map);

        // 全局地图更新
        global_map_frames_.push_back(key_frame);
        if (global_map_frames_.size() % 100 == 0) {
            global_map_ptr_.reset(new CloudData::CLOUD());
            for (auto &frame : global_map_frames_) {
                pcl::transformPointCloud(*frame.cloud_data.cloud_ptr, *transformed_cloud_ptr, frame.pose);
                *global_map_ptr_ += *transformed_cloud_ptr;
            }
            has_new_global_map_ = true;
        }
    }

    bool FrontEnd::GetNewLocalMap(CloudData::CLOUD_PTR &local_map_ptr)
    {
        if (has_new_local_map_) {
            display_filter_.setInputCloud(local_map_ptr_);
            display_filter_.filter(*local_map_ptr);
            return true;
        }
        return false;
    }

    bool FrontEnd::GetNewGlobalMap(CloudData::CLOUD_PTR &global_map_ptr)
    {
        if (has_new_global_map_) {
            display_filter_.setInputCloud(global_map_ptr_);
            display_filter_.filter(*global_map_ptr);
            return true;
        }
        return false;
    }

    bool FrontEnd::GetCurrentScan(CloudData::CLOUD_PTR &current_scan_ptr)
    {
        display_filter_.setInputCloud(result_cloud_ptr_);
        display_filter_.filter(*current_scan_ptr);
        return true;
    }
}
