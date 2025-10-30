#include "lio_ndt/method/pcl_ndt_method.h"
#include <iostream>

namespace lio_ndt {

PCLNDTMethod::PCLNDTMethod()
    : target_(new pcl::PointCloud<PointT>()),
      resolution_(1.0),
      max_iter_(30),
      trans_eps_(1e-6),
      last_fitness_(0.0)
{
    ndt_.setResolution(resolution_);
    ndt_.setMaximumIterations(max_iter_);
    ndt_.setTransformationEpsilon(trans_eps_);
}

void PCLNDTMethod::SetResolution(double res) {
    resolution_ = res;
    ndt_.setResolution(res);
}

void PCLNDTMethod::SetMaximumIterations(int it) {
    max_iter_ = it;
    ndt_.setMaximumIterations(it);
}

void PCLNDTMethod::SetTransformationEpsilon(double eps) {
    trans_eps_ = eps;
    ndt_.setTransformationEpsilon(eps);
}

bool PCLNDTMethod::SetTargetCloud(const CloudPtr &target) {
    if (!target) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    target_ = target;
    ndt_.setInputTarget(target_);
    return true;
}

bool PCLNDTMethod::ScanMatch(const CloudPtr &source,
                             const Eigen::Matrix4f &predict_pose,
                             CloudPtr &transformed_source,
                             Eigen::Matrix4f &result_pose) {
    if (!source || !target_) return false;

    ndt_.setResolution(resolution_);
    ndt_.setMaximumIterations(max_iter_);
    ndt_.setTransformationEpsilon(trans_eps_);
    ndt_.setInputSource(source);

    pcl::PointCloud<PointT> aligned;
    ndt_.align(aligned, predict_pose);  // predict_pose 作为初值

    if (!ndt_.hasConverged()) {
        last_fitness_ = std::numeric_limits<double>::infinity();
        std::cerr << "[PCLNDT] Not converged!" << std::endl;
        return false;
    }
    last_fitness_ = ndt_.getFitnessScore();
    // ✅ NDT 已经返回全局变换矩阵（在世界坐标系下）
    result_pose = ndt_.getFinalTransformation();
    // 保存变换后的点云
    transformed_source = boost::make_shared<pcl::PointCloud<PointT>>(aligned);
    return true;
}

double PCLNDTMethod::GetFitnessScore() const {
    return last_fitness_;
}

} // namespace lio_ndt
