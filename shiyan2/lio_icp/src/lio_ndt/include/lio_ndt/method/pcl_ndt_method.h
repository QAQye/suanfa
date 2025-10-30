#pragma once
#include "registration_method.h"
#include <pcl/registration/ndt.h>
#include <pcl/common/transforms.h>
#include <mutex>
#include <limits>

namespace lio_ndt {

class PCLNDTMethod : public RegistrationMethod {
public:
    using PointT = RegistrationMethod::PointT;
    using CloudPtr = RegistrationMethod::CloudPtr;

    PCLNDTMethod();
    ~PCLNDTMethod() override = default;

    bool SetTargetCloud(const CloudPtr &target) override;
    bool ScanMatch(const CloudPtr &source,
                   const Eigen::Matrix4f &predict_pose,
                   CloudPtr &transformed_source,
                   Eigen::Matrix4f &result_pose) override;

    double GetFitnessScore() const override;

    // 参数设置
    void SetResolution(double res);
    void SetMaximumIterations(int it);
    void SetTransformationEpsilon(double eps);

private:
    pcl::NormalDistributionsTransform<PointT, PointT> ndt_;
    CloudPtr target_;
    std::mutex mutex_;

    double resolution_;
    int max_iter_;
    double trans_eps_;
    double last_fitness_ = std::numeric_limits<double>::infinity();
};

} // namespace lio_ndt

