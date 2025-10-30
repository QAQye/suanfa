#include "lio_ndt/method/pcl_icp_method.h"
#include <iostream>

namespace lio_ndt {

PCLICPMethod::PCLICPMethod()
    : target_(new pcl::PointCloud<PointT>()),
      max_corr_dist_(1.0),
      max_iter_(50),
      trans_eps_(1e-6),
      fitness_eps_(1e-6) {
    icp_.setMaxCorrespondenceDistance(max_corr_dist_);
    icp_.setMaximumIterations(max_iter_);
    icp_.setTransformationEpsilon(trans_eps_);
    icp_.setEuclideanFitnessEpsilon(fitness_eps_);
}

void PCLICPMethod::SetMaxCorrespondenceDistance(double d) {
    max_corr_dist_ = d;
    icp_.setMaxCorrespondenceDistance(d);
}

void PCLICPMethod::SetMaximumIterations(int it) {
    max_iter_ = it;
    icp_.setMaximumIterations(it);
}

void PCLICPMethod::SetTransformationEpsilon(double eps) {
    trans_eps_ = eps;
    icp_.setTransformationEpsilon(eps);
}

void PCLICPMethod::SetEuclideanFitnessEpsilon(double e) {
    fitness_eps_ = e;
    icp_.setEuclideanFitnessEpsilon(e);
}

bool PCLICPMethod::SetTargetCloud(const CloudPtr &target) {
    if (!target) return false;
    std::lock_guard<std::mutex> lock(mutex_);
    target_ = target;
    icp_.setInputTarget(target_);
    return true;
}

bool PCLICPMethod::ScanMatch(const CloudPtr &source,
                             const Eigen::Matrix4f &predict_pose,
                             CloudPtr &transformed_source,
                             Eigen::Matrix4f &result_pose) {
    if (!source || !target_) return false;

    // 参数同步
    icp_.setMaxCorrespondenceDistance(max_corr_dist_);
    icp_.setMaximumIterations(max_iter_);
    icp_.setTransformationEpsilon(trans_eps_);
    icp_.setEuclideanFitnessEpsilon(fitness_eps_);

    pcl::PointCloud<PointT> src_trans;
    pcl::transformPointCloud(*source, src_trans, predict_pose);
    auto src_trans_ptr = boost::make_shared<pcl::PointCloud<PointT>>(src_trans);

    icp_.setInputSource(src_trans_ptr);
    pcl::PointCloud<PointT> aligned;
    icp_.align(aligned);

    if (!icp_.hasConverged()) {
        last_fitness_ = std::numeric_limits<double>::infinity();
        return false;
    }

    last_fitness_ = icp_.getFitnessScore();
    Eigen::Matrix4f T = icp_.getFinalTransformation();
    result_pose = T * predict_pose;

    transformed_source = boost::make_shared<pcl::PointCloud<PointT>>(aligned);
    return true;
}

double PCLICPMethod::GetFitnessScore() const {
    return last_fitness_;
}

} // namespace lio_ndt
