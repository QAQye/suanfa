#include <iostream>
#include <thread>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/ndt.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>

// 把目标点云切成很多小方块（网格）。
// 每个方块里点的分布用 均值 μ 和协方差 Σ 描述。
// 对于源点云的每个点，计算它在目标网格的高斯概率。
// 优化源点云的位姿（旋转和平移），让所有源点云点落入目标点云的高斯模型的概率最大。
// 将目标点云划分成 3D 网格（Voxel）。对每个网格，计算：
// 均值向量 μ：网格中点的平均位置。协方差矩阵 Σ：点的分布方向和形状。
// 这样，目标点云被转化成一组 高斯分布。可以理解为：每个小方块都是一个模糊的“云”，源点云只要落在云里就“匹配得好”。
using namespace std::chrono_literals;

int main(int argc, char **argv)
{
    // 定义目标点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // 用于初始化目标点云
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("../data/room_scan1.pcd", *target_cloud) == -1)
    {
        PCL_ERROR("Could not find pcd \n");
        return (-1);
    }
    std::cout << "load " << target_cloud->size() << " data points from target_cloud" << endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // 初始化源点云
    if (pcl::io::loadPCDFile<pcl::PointXYZ>("../data/room_scan2.pcd", *source_cloud) == -1)
    {
        PCL_ERROR("Could not find pcd \n");
        return (-1);
    }
    std::cout << "load " << source_cloud->size() << " data points from source_cloud" << endl;
    // 降采样点云使用 体素网格滤波（Voxel Grid Filter）对源点云降采样。
    // setLeafSize(0.3, 0.3, 0.3) 定义了每个立方体体素的大小。
    // 降采样后的点云保存在 filter_cloud 中。
    pcl::PointCloud<pcl::PointXYZ>::Ptr filter_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::ApproximateVoxelGrid<pcl::PointXYZ> approximate_voxel_filter;
    approximate_voxel_filter.setLeafSize(0.3, 0.3, 0.3);
    approximate_voxel_filter.setInputCloud(source_cloud);
    approximate_voxel_filter.filter(*filter_cloud);
    std::cout << "Filter cloud contain " << filter_cloud->size() << "data points from source_cloud" << endl;
    //  初始位姿
    // 定义了源点云的初始变换：
    // 绕 Z轴旋转 0.69 弧度。
    // 平移向量为 (1.0, 0, 0)。
    // 将旋转和平移组合成 4x4 矩阵 init_guss。
    Eigen::AngleAxisf init_rotation(0.69, Eigen::Vector3f::UnitZ());
    Eigen::Translation3f init_translasition(1.0, 0, 0);
    Eigen::Matrix4f init_guss = (init_translasition * init_rotation).matrix();
    // 配准结果存储
    pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    // =======================   ndt   =======================
    // 补全相关代码
    // 创建ndt对象
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
    // 设置相关参数
    ndt.setInputSource(filter_cloud);
    ndt.setInputTarget(target_cloud);
    // 设置收敛条件
    ndt.setTransformationEpsilon(0.01);
    ndt.setStepSize(0.1);
    ndt.setResolution(1.0);
    ndt.setMaximumIterations(30);
//     核心配准方法。
// output：配准后的源点云。
// initial_guess：初始位姿矩阵（前面定义的 init_guss）。
    ndt.align(*output_cloud, init_guss);
    if(ndt.hasConverged()){
    cout << "NDT 配准成功" << endl;
    } else {
        cout << "NDT 配准失败" << endl;
    }
    cout << "配准分数: " << ndt.getFitnessScore() << endl;


    // =======================   ndt   =======================
    

    pcl::visualization::PCLVisualizer::Ptr viewer_final(new pcl::visualization::PCLVisualizer("3D viewer"));
    viewer_final->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(target_cloud, 255, 0, 0);
    viewer_final->addPointCloud<pcl::PointXYZ>(target_cloud, target_color, "target_cloud");
    viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target_cloud");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> output_color(output_cloud, 0, 255, 0);
    viewer_final->addPointCloud<pcl::PointXYZ>(output_cloud, output_color, "output_cloud");
    viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "output_cloud");
    viewer_final->addCoordinateSystem(1.0, "global");
    viewer_final->initCameraParameters();

    while (!viewer_final->wasStopped())
    {
        viewer_final->spinOnce(100);
        std::this_thread::sleep_for(100ms);
    }
    return 0;
}