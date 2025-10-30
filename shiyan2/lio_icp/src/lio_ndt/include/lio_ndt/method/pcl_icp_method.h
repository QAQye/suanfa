#pragma once
#include "registration_method.h"
#include <pcl/registration/icp.h>
#include <pcl/common/transforms.h>
#include <mutex>
#include <limits>

namespace lio_ndt {

class PCLICPMethod : public RegistrationMethod {
public:
    using PointT = RegistrationMethod::PointT;
    using CloudPtr = RegistrationMethod::CloudPtr;

    PCLICPMethod();
    ~PCLICPMethod() override = default;

    // RegistrationMethod 接口
    bool SetTargetCloud(const CloudPtr &target) override;
    bool ScanMatch(const CloudPtr &source,
                   const Eigen::Matrix4f &predict_pose,
                   CloudPtr &transformed_source,
                   Eigen::Matrix4f &result_pose) override;

    double GetFitnessScore() const override;

    // 参数设置
    void SetMaxCorrespondenceDistance(double d);
    void SetMaximumIterations(int it);
    void SetTransformationEpsilon(double eps);
    void SetEuclideanFitnessEpsilon(double e);

private:
    pcl::IterativeClosestPoint<PointT, PointT> icp_;
    CloudPtr target_;
    std::mutex mutex_;

    double max_corr_dist_;
    int max_iter_;
    double trans_eps_;
    double fitness_eps_;
    double last_fitness_ = std::numeric_limits<double>::infinity();
};

} // namespace lio_ndt

