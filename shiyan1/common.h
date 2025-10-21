
#ifndef OPTIMIZED_ICP_COMMON_H
#define OPTIMIZED_ICP_COMMON_H

#include <Eigen/Dense>
using namespace std;
// 将向量映射为反对称矩阵
// 模板函数，模板函数表示Derived可以表示为任意类型，根据传入的类型来进行自动匹配
// Eigen::MatrixBase<Derived>：是 Eigen 所有矩阵类型的“通用父类”。
// Derived 将由编译器根据传入参数类型推导（例如 Eigen::Vector3f 实参会让 Derived = Eigen::Matrix<float,3,1>）。

template<typename Derived>

// 返回类型 Eigen::Matrix<typename Derived::Scalar, 3, 3>:

// Derived::Scalar 是输入向量的元素类型（float/double），因此返回的 3×3 矩阵与输入标量类型一致。
// Eigen::MatrixBase<Derived>允许传入任意 Eigen 向量/矩阵的表达式（不仅限于具体类型）。
// 定义vhat矩阵，将叉乘转化为矩阵乘法运算
// 反对称矩阵主对角线上的元素一定是为0的也就是hat矩阵
Eigen::Matrix<typename Derived::Scalar, 3, 3> Hat(const Eigen::MatrixBase<Derived> &v)
{
    Eigen::Matrix<typename Derived::Scalar, 3, 3> skew_mat; // 反对称矩阵
    // 将所有的值都设置为0
    skew_mat.setZero();
    skew_mat(0, 1) = -v(2);
    skew_mat(0, 2) = v(1);
    skew_mat(1, 2) = -v(0);
    skew_mat(1, 0) = v(2);
    skew_mat(2, 0) = -v(1);
    skew_mat(2, 1) = v(0);
    return skew_mat;
}
// 这里Derived：表示你传进来的类型，比如 Eigen::Vector3f。
// typename Derived::Scalar：表示矩阵里元素的数据类型（float 或 double）。
// Eigen::MatrixBase<Derived>：是 Eigen 所有矩阵类型的“通用父类”。
// 如果你传入 Eigen::Vector3f，就会生成一个专门处理 float 的版本。
// 如果你传入 Eigen::Vector3d，就会生成一个处理 double 的版本。
// 这段代码实现了 旋转向量（轴角表示）到旋转矩阵的转换，也就是 SO(3) 的指数映射，核心使用 Rodrigues 公式。
template<typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3> SO3Exp(const Eigen::MatrixBase<Derived> &v) 
{
    // 定义一个 3×3 矩阵 R，用于存储旋转矩阵结果。
    Eigen::Matrix<typename Derived::Scalar, 3, 3> R;
    // v.norm() 返回向量的 模长，即旋转向量的大小：
    // 这里的 θ 表示旋转角度（弧度）。旋转向量的方向表示旋转轴，长度表示旋转角度。
    typename Derived::Scalar theta = v.norm(); // 计算v的模
//     将旋转向量归一化，得到单位旋转轴：
// 后续 Rodrigues 公式需要 单位向量 表示旋转轴。
    Eigen::Matrix<typename Derived::Scalar, 3, 1> v_normalized = v.normalized(); // 归一化
    // 罗德里格斯公式
    R = cos(theta) * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() + 
        (typename Derived::Scalar(1.0) - cos(theta)) * v_normalized * v_normalized.transpose() + 
        sin(theta) * Hat(v_normalized);

    return R;
}

#endif //OPTIMIZED_ICP_COMMON_H
