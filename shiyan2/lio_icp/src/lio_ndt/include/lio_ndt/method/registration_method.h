// registration_method.h
#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include <memory>

namespace lio_ndt {

/**
 * 抽象基类：统一匹配器接口（front_end 调用该接口）
 *
 * 使用 pcl::PointXYZ 作为点类型示例；如果你的项目使用 PointXYZI 或自定义类型，
 * 请将下面的 typedef 替换为对应类型，或使用模板化实现。
 */
class RegistrationMethod {
public:
    using PointT = pcl::PointXYZ;
    using CloudPtr = pcl::PointCloud<PointT>::Ptr;

    virtual ~RegistrationMethod() = default;
    // registration_method.h 增加这一 virtual
    virtual double GetFitnessScore() const { return -1.0; } // 默认未实现返回 -1
    /// 设置目标（map）点云
    virtual bool SetTargetCloud(const CloudPtr &target) = 0;
    
    // registration_method.h 增加这一 virtual
    /**
     * 扫描匹配主函数
     * @param source 输入源点云（当前帧）
     * @param predict_pose 先验/预测位姿（4x4），用于做初值
     * @param transformed_source 输出：配准后的源点云（在 target 坐标系下）
     * @param result_pose 输出：最终的位姿（4x4），将 source -> global 的变换写入
     * @return 是否收敛/成功
     */
    virtual bool ScanMatch(const CloudPtr &source,
                           const Eigen::Matrix4f &predict_pose,
                           CloudPtr &transformed_source,
                           Eigen::Matrix4f &result_pose) = 0;
};
} // namespace lio_ndt
