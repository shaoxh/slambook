#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>

using namespace std;
using namespace cv;

void find_feature_matches (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches );

// 像素坐标转相机归一化坐标
Point2d pixel2cam ( const Point2d& p, const Mat& K );

void bundleAdjustment (
    const vector<Point3f> points_3d,
    const vector<Point2f> points_2d,
    const Mat& K,
    Mat& R, Mat& t
);

int main ( int argc, char** argv )
{
    if ( argc != 5 )
    {
        cout<<"usage: pose_estimation_3d2d img1 img2 depth1 depth2"<<endl;
        return 1;
    }
    //-- 读取图像
    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches ( img_1, img_2, keypoints_1, keypoints_2, matches );
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

    // 建立3D点
    Mat d1 = imread ( argv[3], CV_LOAD_IMAGE_UNCHANGED );       // 深度图为16位无符号数，单通道图像
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    for ( DMatch m:matches )
    {
        // ptr is a pointer of Mat.ptr(0)[1] means element at first row & second column
        ushort d = d1.ptr<unsigned short> (int ( keypoints_1[m.queryIdx].pt.y )) [ int ( keypoints_1[m.queryIdx].pt.x ) ];
        if ( d == 0 )   // bad depth
            continue;
        // trans from a depth to distance
        float dd = d/5000.0;
        Point2d p1 = pixel2cam ( keypoints_1[m.queryIdx].pt, K );
        // 建立一个齐次坐标
        pts_3d.push_back ( Point3f ( p1.x*dd, p1.y*dd, dd ) );
        pts_2d.push_back ( keypoints_2[m.trainIdx].pt );
    }

    cout<<"3d-2d pairs: "<<pts_3d.size() <<endl;

    Mat r, t;
    // EPnP 采用第一帧相机中的 3D 坐标和第二帧相机中的 3D 点对应的像素坐标当作输入，输出第一帧和第二帧之间的 R t
    // 原理上，首先计算出这些像素坐标在第二帧的 camera 坐标系底下的 3D 坐标，然后计算 camera 3D 坐标到真 3D 坐标的 R t
    // 上述的真 3D 坐标，实际上就是这些点在第一帧相片中的 camera 坐标系底下的坐标
    // 因此就求解出了两个 camera 坐标系之间同一堆地面点对应的不同坐标，因此就可以计算这两个坐标系之间的转换
    // 坐标系之间的转换就是两帧之间的位姿 R t 转换
    solvePnP ( pts_3d, pts_2d, K, Mat(), r, t, false ); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    Mat R;
    cv::Rodrigues ( r, R ); // r为旋转向量形式，用Rodrigues公式转换为矩阵

    cout<<"R="<<endl<<R<<endl;
    cout<<"t="<<endl<<t<<endl;

    cout<<"calling bundle adjustment"<<endl;

    // 根据重投影方程建立 BA 进行短期优化
    // 输入：空间 3D 点坐标，与它们对应的第二个相机中的像素点坐标，第二个相机的姿态 R t，相机的内参数 K
    // 输出：BA 后的 R t
    // 节点：第二个相机的位置和姿态值、所有特征点的空间 3D 位置
    // 边：每个空间 3D 点在第二个相机中的投影像素与原本特征点对应的第二个相机中像素的距离差
    bundleAdjustment ( pts_3d, pts_2d, K, R, t );
}

void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector< DMatch >& matches )
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}

Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}

// 建立 BA 图优化的基本额步骤
void bundleAdjustment (
    const vector< Point3f > points_3d,
    const vector< Point2f > points_2d,
    const Mat& K,
    Mat& R, Mat& t )
{
    // 初始化g2o
    // 声明 typeof g2o
    typedef g2o::BlockSolver< g2o::BlockSolverTraits<6,3> > Block;  // pose 维度为 6, landmark 维度为 3
    // 实例化一个线性求解器指针
    Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>(); // 线性方程求解器
    // 用线性求解器实例化一个（矩阵）块求解器
    Block* solver_ptr = new Block ( linearSolver );     // 矩阵块求解器
    // 用块求解器实例化一个求解器
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
    // 声明一个最优化过程，并且 set 这个过程的求解算子为“线性、块、求解器”算子
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );

    // vertex
    // 实例化一个李代数类型的位姿顶点，并且 add 到最优化过程中
    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
    Eigen::Matrix3d R_mat;
    R_mat <<
          R.at<double> ( 0,0 ), R.at<double> ( 0,1 ), R.at<double> ( 0,2 ),
               R.at<double> ( 1,0 ), R.at<double> ( 1,1 ), R.at<double> ( 1,2 ),
               R.at<double> ( 2,0 ), R.at<double> ( 2,1 ), R.at<double> ( 2,2 );
    pose->setId ( 0 );
    // 位姿节点的状态量是用 SE3Quat 描述的。
    // SE3 就是欧式变换群对应的李代数，是一个 6 维的变量，描述了欧式空间的 3 维旋转和 3 维平移
    pose->setEstimate ( g2o::SE3Quat (
                            R_mat,
                            Eigen::Vector3d ( t.at<double> ( 0,0 ), t.at<double> ( 1,0 ), t.at<double> ( 2,0 ) )
                        ) );
    // 把位姿节点 add 到优化过程中
    optimizer.addVertex ( pose );

    // 实例化一组 3D 地表点当作定点，添加到优化过程中
    // 优化过程中添加 3D landmark 的顶点
    int index = 1;
    for ( const Point3f p:points_3d )   // landmarks
    {
        g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
        point->setId ( index++ );
        point->setEstimate ( Eigen::Vector3d ( p.x, p.y, p.z ) );
        point->setMarginalized ( true ); // g2o 中必须设置 marg 参见第十讲内容
        optimizer.addVertex ( point );
    }

    // parameter: camera intrinsics
    // 实例化一个相机内参对象，添加到优化过程的 param 当中
    // 包括一个焦距（focal length）和主点（fx，fy）
    // 此处没有的相机基线（可能是双目相机 g2o 优化的时候使用的参数）
    g2o::CameraParameters* camera = new g2o::CameraParameters (
        K.at<double> ( 0,0 ), Eigen::Vector2d ( K.at<double> ( 0,2 ), K.at<double> ( 1,2 ) ), 0
    );
    camera->setId ( 0 );
    optimizer.addParameter ( camera );

    // edges
    // 实例化一组边添加到优化过程中
    // 所谓的重投影误差：边的含义是 3D 地表经过相机内参数矩阵投影到相机上的像素，与原本 3D 地表对应的那一组像素，之间的距离差
    // 边从 index 1 开始记录
    index = 1;
    for ( const Point2f p:points_2d )
    {
        // 这个边 EdgeProjectXYZ2UV 是 g2o 里恰好有的
        // 它按照模板函数继承自一个二元边：BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap>
        // 前两个模板值说的是 measurement 类型：是一个 2 维向量，用 vector2D 表示
        // 后两个模板值就是这个二元边的两个定点类型
        // 顶点-1 类型是 VertexSBAPointXYZ，即地标向量
        // 顶点-2 类型是 VertexSE3Expmap，即用 se3 表示的位姿向量
        g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
        edge->setId ( index );
        // 边的顶点-1：是 3D 地表点的位置
        // 也就是顶点集合里的每一个 3D 顶点
        edge->setVertex ( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> ( optimizer.vertex ( index ) ) );
        // 边的顶点-2：是相机光心的位置和朝向
        // 优化过程会根据边的两个定点以及前面 set 进去的相机内参 param 自动计算出边中的 顶点-1 投影后的坐标是多少
        edge->setVertex ( 1, pose );
        // 上述计算出的投影后的坐标会与 measurement（测量值）进行求差
        // 优化过程会自动调整这个差别，达到总体最小二乘的解
        edge->setMeasurement ( Eigen::Vector2d ( p.x, p.y ) );
        edge->setParameterId ( 0,0 );
        // 先验信息矩阵，是一个 2 × 2 的矩阵
        // 推断描述了两个顶点的可信度？
        edge->setInformation ( Eigen::Matrix2d::Identity() );
        // 优化过程添加边
        optimizer.addEdge ( edge );
        index++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    // 3 个步骤启动优化过程
    // verbose、initialize、optimize 100 次迭代
    optimizer.setVerbose ( true );
    optimizer.initializeOptimization();
    optimizer.optimize ( 100 );
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
    cout<<"optimization costs time: "<<time_used.count() <<" seconds."<<endl;

    cout<<endl<<"after optimization:"<<endl;
    // 优化结果中对 pose 的优化的结果在 pose 的 estimate 中
    // 优化结果中对 3D 地标的优化
    cout<<"T="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<endl;
}
