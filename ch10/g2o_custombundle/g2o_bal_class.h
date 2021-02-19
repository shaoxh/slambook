#include <Eigen/Core>
#include "g2o/core/base_vertex.h"
#include "g2o/core/base_binary_edge.h"

#include "ceres/autodiff.h"

#include "tools/rotation.h"
#include "common/projection.h"

// 自定义顶点类 camera
// 9 代表存储的顶点向量的维数; VectorXd 表示存储这个 9 维向量的类型
// camera : 9 dims array with
// [0-2] : angle-axis rotation 前三个变量是旋转角度变量
// [3-5] : translateion 中间三个是平移变量
// [6-8] : camera parameter, 最后三个是相机焦距和畸变参数
// [6] focal length, [7-8] second and forth order radial distortion
//    const T& l1 = camera[7];
//    const T& l2 = camera[8];
//
//    T r2 = xp*xp + yp*yp;
//    T distortion = T(1.0) + r2 * (l1 + l2 * r2);
//
//    const T& focal = camera[6];
//    predictions[0] = focal * distortion * xp;
//    predictions[1] = focal * distortion * yp;
class VertexCameraBAL : public g2o::BaseVertex<9,Eigen::VectorXd>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexCameraBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void setToOriginImpl() {}

    // 顶点类的更新方法
    // 把带更新的 double 指针量首先构建一个 9 维度的 VectorXd
    // 然后累加到 estimate 中
    virtual void oplusImpl ( const double* update )
    {
        Eigen::VectorXd::ConstMapType v ( update, VertexCameraBAL::Dimension );
        _estimate += v;
    }

};

// 自定义顶点类 for 地表点
// 顶点是一个 3 维度的向量 Vector3d 类型
class VertexPointBAL : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexPointBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void setToOriginImpl() {}

    // 顶点类的更新方法
    // 把带更新的 double 指针量首先构建一个 9 维度的 VectorXd
    // 然后累加到 estimate 中
    virtual void oplusImpl ( const double* update )
    {
        Eigen::Vector3d::ConstMapType v ( update );
        _estimate += v;
    }
};

// 自定义边 edge
// 是一个继承自 BaseBinaryEdge 的二元边
// 边的顶点-0 放的是自定义相机顶点 VertexCameraBAL
// 边的顶点-1 放的是自定义地标顶点 VertexPointBAL
class EdgeObservationBAL : public g2o::BaseBinaryEdge<2, Eigen::Vector2d,VertexCameraBAL, VertexPointBAL>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeObservationBAL() {}

    virtual bool read ( std::istream& /*is*/ )
    {
        return false;
    }

    virtual bool write ( std::ostream& /*os*/ ) const
    {
        return false;
    }

    virtual void computeError() override   // The virtual function comes from the Edge base class. Must define if you use edge.
    {
        const VertexCameraBAL* cam = static_cast<const VertexCameraBAL*> ( vertex ( 0 ) );
        const VertexPointBAL* point = static_cast<const VertexPointBAL*> ( vertex ( 1 ) );

        // 这里实际上就是调用 operator 函数
        // operator 函数中会  _error 中的量
        // 仍然是重投影误差：error 是 3D 地标点投影成像素后与原本的像素的距离差
        // residuals[0] = predictions[0] - T ( measurement() ( 0 ) );
        // residuals[1] = predictions[1] - T ( measurement() ( 1 ) );
        ( this->operator()(cam->estimate().data(), point->estimate().data(), _error.data()) );
//        ( *this ) ( cam->estimate().data(), point->estimate().data(), _error.data() );

    }

    // 为了使用 Ceres 求导功能而定义的函数，让本类成为拟函数类
    template<typename T>
    bool operator() ( const T* camera, const T* point, T* residuals ) const
    {
        T predictions[2];
        // 方法返回 3D 点按照相机参数模型（包含畸变矫正的 9 参数模型）投影到像素坐标系上的坐标值
        // 返回值在 predictions 里面，是个 2 维的向量，即 （u，v）坐标
        // predictions : 2D predictions with center of the image plane.
        CamProjectionWithDistortion ( camera, point, predictions );
        residuals[0] = predictions[0] - T ( measurement() ( 0 ) );
        residuals[1] = predictions[1] - T ( measurement() ( 1 ) );

        return true;
    }


    virtual void linearizeOplus() override
    {
        // use numeric Jacobians
        // BaseBinaryEdge<2, Vector2d, VertexCameraBAL, VertexPointBAL>::linearizeOplus();
        // return;
        
        // using autodiff from ceres. Otherwise, the system will use g2o numerical diff for Jacobians

        const VertexCameraBAL* cam = static_cast<const VertexCameraBAL*> ( vertex ( 0 ) );
        const VertexPointBAL* point = static_cast<const VertexPointBAL*> ( vertex ( 1 ) );
        // 声明一个 Ceres 里面的 AutoDiff 去自动求导
        typedef ceres::internal::AutoDiff<EdgeObservationBAL, double, VertexCameraBAL::Dimension, VertexPointBAL::Dimension> BalAutoDiff;

        // 声明两个变量用于求导过程
        // 这里的 Dimensioin 就是边的 Dimension，这里是二元边，所以应该是 2
        Eigen::Matrix<double, Dimension, VertexCameraBAL::Dimension, Eigen::RowMajor> dError_dCamera;
        Eigen::Matrix<double, Dimension, VertexPointBAL::Dimension, Eigen::RowMajor> dError_dPoint;
        // 初始化 AutoDiff 里面的成员变量
        // 有四个需要初始化
        // T const *const *parameters,
        // int num_outputs,
        // T *function_value,
        // T **jacobians
        double *parameters[] = { const_cast<double*> ( cam->estimate().data() ), const_cast<double*> ( point->estimate().data() ) };
        double *jacobians[] = { dError_dCamera.data(), dError_dPoint.data() };
        double value[Dimension];
        // AutoDiff 的用法。需要在第一个 functor 的变脸中提供 operator（）函数成员，以用来计算
        // 这里的 functor 是 this 函数，就是 当前函数所在的那个类：EdgeObservationBAL
        bool diffState = BalAutoDiff::Differentiate ( *this, parameters, Dimension, value, jacobians );

        // copy over the Jacobians (convert row-major -> column-major)
        if ( diffState )
        {
            _jacobianOplusXi = dError_dCamera;
            _jacobianOplusXj = dError_dPoint;
        }
        else
        {
            assert ( 0 && "Error while differentiating" );
            _jacobianOplusXi.setZero();
            _jacobianOplusXj.setZero();
        }
    }
};
