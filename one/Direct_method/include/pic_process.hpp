#ifndef PIC_PROCESS_H_
#define PIC_PROCESS_H_
#include <iostream>
#include <deque>
#include <vector>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <sophus/so3.h>
#include <sophus/se3.h>

#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
using namespace std;
using namespace cv;
struct Descrip
{
    Mat descriptors_1;
    Mat descriptors_2;
};
struct RTmatrix{
    Mat R;
    Mat t;
};
struct Keypoints{
    vector<KeyPoint> keypoints1;
    vector<KeyPoint> keypoints2;
};
extern vector< DMatch > good_matches;
Keypoints  feature_extract(deque<Mat> picseries);
RTmatrix pose_estimation_2d2d ( Keypoints get_points,
                            std::vector< DMatch > matches,
                            Mat& R, Mat& t );
void triangulation ( 
    const vector< KeyPoint >& keypoint_1, 
    const vector< KeyPoint >& keypoint_2, 
    const std::vector< DMatch >& matches,
    const Mat& R, const Mat& t, 
    vector< Point3d >& points );
    
#endif