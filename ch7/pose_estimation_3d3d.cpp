#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>

using namespace std;
using namespace cv;

void find_feature_matches(
        const Mat &img_1, const Mat &img_2,
        std::vector<KeyPoint> &keypoints_1,
        std::vector<KeyPoint> &keypoints_2,
        std::vector<DMatch> &matches);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

void pose_estimation_3d3d(
        const vector<Point3f> &pts1,
        const vector<Point3f> &pts2,
        Mat &R, Mat &t
);

void bundleAdjustment(
        const vector<Point3f> &points_3d,
        const vector<Point3f> &points_2d,
        Mat &R, Mat &t
);

// g2o edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d &point) : _point(point) {}

    virtual void computeError() {
        const g2o::VertexSE3Expmap *pose = static_cast<const g2o::VertexSE3Expmap *> ( _vertices[0] );
        // measurement is p, point is p'
        cout<<"EDGE: _point "<< _point<< endl;
        cout<<"EDGE: map_point "<< pose->estimate().map(_point)<< endl;
        cout<<"EDGE: _measurement "<< _measurement<< endl;
        // 值得注意的是，这里的 map 方法其实就是把参数中的 _point 点按照 post 的 R 和 t 做一下映射
        // 也就是把 _point 映射到上一个（或下一个）位姿中 camera 空间下的坐标（也就是 measurement 空间下的坐标）
        // 这样就可以使用 measurement 的减法来计算 _error;减法应该经过重载的
        // map 的方法如下所示：
        //       Vector3D map(const Vector3D & xyz) const
        //      {
        //        return _r*xyz + _t;
        //      }
        _error = _measurement - pose->estimate().map(_point);
    }

    virtual void linearizeOplus() {
        // 若想自己构建图优化的约束，就要首先找到误差方程以及求偏导的方法
        // 求偏导的方法就是设置 jacobian 矩阵

        g2o::VertexSE3Expmap *pose = static_cast<g2o::VertexSE3Expmap *>(_vertices[0]);
        g2o::SE3Quat T(pose->estimate());
        Eigen::Vector3d xyz_trans = T.map(_point);
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];

        _jacobianOplusXi(0, 0) = 0;
        _jacobianOplusXi(0, 1) = -z;
        _jacobianOplusXi(0, 2) = y;
        _jacobianOplusXi(0, 3) = -1;
        _jacobianOplusXi(0, 4) = 0;
        _jacobianOplusXi(0, 5) = 0;

        _jacobianOplusXi(1, 0) = z;
        _jacobianOplusXi(1, 1) = 0;
        _jacobianOplusXi(1, 2) = -x;
        _jacobianOplusXi(1, 3) = 0;
        _jacobianOplusXi(1, 4) = -1;
        _jacobianOplusXi(1, 5) = 0;

        _jacobianOplusXi(2, 0) = -y;
        _jacobianOplusXi(2, 1) = x;
        _jacobianOplusXi(2, 2) = 0;
        _jacobianOplusXi(2, 3) = 0;
        _jacobianOplusXi(2, 4) = 0;
        _jacobianOplusXi(2, 5) = -1;
    }

    bool read(istream &in) {}

    bool write(ostream &out) const {}

protected:
    // 边的属性里面有一个 _point 对象
    // 它负责 collect 被 R & t 作用的 3D 坐标
    // 即：如果 R & t 对应的是从第一帧到第二帧的变换的话
    // 那么 _point 记录的就是第一帧相机 camera 坐标系下 3D 点 de  coordinates
    Eigen::Vector3d _point;
    // 这个时候 _measurements 就是第二帧相机 camera 坐标系下 3D 点的坐标
    // 因此 setMeasurements 函数 set 的就是第二帧相机中对应的 3D 点坐标
};

int main(int argc, char **argv) {
    if (argc != 5) {
        cout << "usage: pose_estimation_3d3d img1 img2 depth1 depth2" << endl;
        return 1;
    }
    //-- 读取图像
    Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    // 建立3D点
    Mat depth1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);       // 深度图为16位无符号数，单通道图像
    Mat depth2 = imread(argv[4], CV_LOAD_IMAGE_UNCHANGED);       // 深度图为16位无符号数，单通道图像
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts1, pts2;

    for (DMatch m:matches) {
        // 建立第一帧中观察到的 3D 点 d1
        // 建立第二帧中观察到的 3D 点 d2
        ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
        ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
        if (d1 == 0 || d2 == 0)   // bad depth
            continue;
        // 第一帧的 3D 点 d1 对应的 camera 坐标系的 3D 点 p1
        // 第二帧的 3D 点 d2 对应的 camera 坐标系的 3D 点 p2
        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        // depth 转换成距离
        float dd1 = float(d1) / 5000.0;
        float dd2 = float(d2) / 5000.0;
        // 构造齐次坐标
        pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
        pts2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));
    }

    cout << "3d-3d pairs: " << pts1.size() << endl;
    Mat R, t;
    // 进行 ICP 的 R t 求解通过 SVD 方法
    pose_estimation_3d3d(pts1, pts2, R, t);
    cout << "ICP via SVD results: " << endl;
    cout << "R = " << R << endl;
    cout << "t = " << t << endl;
    cout << "R_inv = " << R.t() << endl;
    cout << "t_inv = " << -R.t() * t << endl;

    cout << "calling bundle adjustment" << endl;

    // 进行 ICP 中的 R t 求解通过 BA 方法
    bundleAdjustment(pts1, pts2, R, t);

    // verify p1 = R*p2 + t
    for (int i = 0; i < 5; i++) {
        cout << "p1 = " << pts1[i] << endl;
        cout << "p2 = " << pts2[i] << endl;
        cout << "(R*p2+t) = " <<
             R * (Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, pts2[i].z) + t
             << endl;
        cout << endl;
    }
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
    return Point2d
            (
                    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
            );
}

// 通过 3D-3D 的方法计算相机在两帧之间的 R t
// known 两帧之间的匹配点
// known the 3D coordinates of matched pair-points
// their 3D coordinates are different in two separate coordinate system
// differences between the two systems are that of pose and location (that is R & t)
void pose_estimation_3d3d(
        const vector<Point3f> &pts1,
        const vector<Point3f> &pts2,
        Mat &R, Mat &t
) {
    // 首先求出权重点，然后给没一个点去权（去质心）
    Point3f p1, p2;     // center of mass
    int N = pts1.size();
    for (int i = 0; i < N; i++) {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = Point3f(Vec3f(p1) / N);
    p2 = Point3f(Vec3f(p2) / N);
    vector<Point3f> q1(N), q2(N); // remove the center
    for (int i = 0; i < N; i++) {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // compute q1*q2^T
    // q1*q2^T 就是 （3 × 1） 矩阵 点乘 （1 × 3） 矩阵得到 3 × 3 矩阵
    // W 就是 sigma（q1*q2^T），即对上面每一个 qi 的结果求和
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i < N; i++) {
        W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    cout << "W=" << W << endl;

    // SVD on W
    // 实例化一个 svd 分解的对象，实例化后就分解完成了
    // 分解完的结果就在 svdU 和 svdV 当中
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    cout << "lamda= " << svd.singularValues() << endl;
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    if (U.determinant() * V.determinant() < 0) {
        for (int x = 0; x < 3; ++x) {
            U(x, 2) *= -1;
        }
    }

    cout << "U=" << U << endl;
    cout << "V=" << V << endl;

    // SVD 的结果中反演出 R 和 t
    // R = U 乘以 V 的转置
    // t = 第二帧的质心点 减去 经过 R 变换后的第一帧质心点
    Eigen::Matrix3d R_ = U * (V.transpose());
    Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

    // convert to cv::Mat
    // 转换成 opencv 中的 Mat 类型
    R = (Mat_<double>(3, 3) <<
                            R_(0, 0), R_(0, 1), R_(0, 2),
            R_(1, 0), R_(1, 1), R_(1, 2),
            R_(2, 0), R_(2, 1), R_(2, 2)
    );
    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

void bundleAdjustment(
        const vector<Point3f> &pts1,
        const vector<Point3f> &pts2,
        Mat &R, Mat &t) {
    // 初始化g2o
    // 实例化一个 typedef
    // 实例化一个线性求解器 linearSolver
    // 实例化一个 linearSolver 的矩阵求解块 block
    // 实例化一个包含 block 指针的迭代求解器 solver
    // 声明一个最优化过程 optimizer，并设置其中的算子为 solver
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3> > Block;  // pose维度为 6, landmark 维度为 3
    Block::LinearSolverType *linearSolver = new g2o::LinearSolverEigen<Block::PoseMatrixType>(); // 线性方程求解器
    Block *solver_ptr = new Block(linearSolver);      // 矩阵块求解器
    g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // vertex
    // 图优化的顶点
    // 顶点是相机的位姿 R t 组成的 6 惟向量
    g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap(); // camera pose
    pose->setId(0);
    pose->setEstimate(g2o::SE3Quat(
            Eigen::Matrix3d::Identity(),
            Eigen::Vector3d(0, 0, 0)
    ));
    optimizer.addVertex(pose);

    // edges
    // 图优化的边
    // 自定义的边，继承自
    // 边只有一个顶点，就是 3D 点的位置
    int index = 1;
    vector<EdgeProjectXYZRGBDPoseOnly *> edges;
    for (size_t i = 0; i < pts1.size(); i++) {
        // 边的属性里面有一个 _point 对象
        // 它负责 collect 被 R & t 作用的 3D 坐标
        // 即：如果 R & t 对应的是从第一帧到第二帧的变换的话
        // 那么 _point 记录的就是第一帧相机 camera 坐标系下 3D 点 de  coordinates
        EdgeProjectXYZRGBDPoseOnly *edge = new EdgeProjectXYZRGBDPoseOnly(
                Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
        edge->setId(index);
        edge->setVertex(0, dynamic_cast<g2o::VertexSE3Expmap *> (pose));
        // 这个时候 _measurements 就是第二帧相机 camera 坐标系下 3D 点的坐标
        // 因此 setMeasurements 函数 set 的就是第二帧相机中对应的 3D 点坐标
        edge->setMeasurement(Eigen::Vector3d(
                pts1[i].x, pts1[i].y, pts1[i].z));
        edge->setInformation(Eigen::Matrix3d::Identity() * 1e4);
        optimizer.addEdge(edge);
        index++;
        edges.push_back(edge);
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    // 开始优化的步骤
    // verbose 为 true
    // 初始化和设置优化次数 10 次
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization costs time: " << time_used.count() << " seconds." << endl;

    cout << endl << "after optimization:" << endl;
    // 被优化后的顶点（这里是 pose）结果放在 pose 的 estimate 变量当中
    // 联想被优化后的 3D 坐标结果放在哪呢？
    cout << "T=" << endl << Eigen::Isometry3d(pose->estimate()).matrix() << endl;

}
