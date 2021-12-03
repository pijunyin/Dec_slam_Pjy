#ifndef DIRECT_H_
#define DIRECT_H_
#include <iostream>
#include <unistd.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
using namespace std;
using namespace cv;
using namespace g2o;
struct Measurement
{
    Measurement (Eigen::Vector3d p,float g):pos_world(p),grayscale(g){}
    Eigen::Vector3d pos_world;
    float grayscale;
};
void pose_estimate_direct(const vector< Measurement >& measurements, cv::Mat* gray, Eigen::Matrix3f& K, Eigen::Isometry3d& Tcw); 
Eigen::Vector3d project2Dto3D ( int x, int y, int d, float fx, float fy, float cx, float cy, float scale );
Eigen::Vector2d project3Dto2D ( float x, float y, float z, float fx, float fy, float cx, float cy );

#endif