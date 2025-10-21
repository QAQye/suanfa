//
// Created by Zhang Zhimeng on 2021/3/24.
//

#ifndef OPTIMIZED_ICP_GN_H
#define OPTIMIZED_ICP_GN_H

#include <eigen3/Eigen/Core>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
// 输入：

// 源点云 (source_cloud_ptr)

// 目标点云 (target_cloud_ptr)

// 初始预测位姿 (predict_pose)

// 输出：

// 优化后的旋转+平移矩阵 (result_pose)

// 转换后的源点云 (transformed_source_cloud_ptr)

// 提供：

// 最大迭代次数 (max_iterations_)

// 最大对应点距离 (max_correspond_distance_)

// 收敛条件 (transformation_epsilon_)
class OptimizedICPGN
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW; // eigen自动内存对齐
    // 初始化 KdTree，用于快速最近邻搜索。

    // KdTree 是 ICP 中寻找最近点的核心数据结构。

    OptimizedICPGN();

//     保存目标点云指针。
// 构建 KdTree，方便在 ICP 中快速查找源点最近的目标点
    bool SetTargetCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr &target_cloud_ptr);
// source_cloud_ptr：源点云
// predict_pose：初始预测位姿
// 输出：
// transformed_source_cloud_ptr：配准后的点云
// result_pose：最终优化位姿矩阵
    bool Match(const pcl::PointCloud<pcl::PointXYZ>::Ptr &source_cloud_ptr,
               const Eigen::Matrix4f &predict_pose,
               pcl::PointCloud<pcl::PointXYZ>::Ptr &transformed_source_cloud_ptr,
               Eigen::Matrix4f &result_pose);

    float GetFitnessScore(float max_range = std::numeric_limits<float>::max()) const;

    void SetMaxIterations(unsigned int iter);

    /*!
     * 设置最大对应点距离
     * note：该参数对于ICP的精度影响较大
     * @param max_correspond_distance
     */
    void SetMaxCorrespondDistance(float max_correspond_distance);

    /*!
     * 设置增量变换矩阵收敛的条件
     * @brief 当求解的增量小于该条件时说明ICP迭代收敛
     * @param transformation_epsilon
     */
    void SetTransformationEpsilon(float transformation_epsilon);

    bool HasConverged() const;

private:
    unsigned int max_iterations_{};
    float max_correspond_distance_{}, transformation_epsilon_{};

    bool has_converge_ = false;

    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud_ptr_ = nullptr;
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud_ptr_ = nullptr;
    Eigen::Matrix4f final_transformation_;

    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtree_flann_ptr_ = nullptr; // In order to search
};

#endif //OPTIMIZED_ICP_GN_H
