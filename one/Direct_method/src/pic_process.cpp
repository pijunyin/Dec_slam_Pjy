#include "../include/pic_process.hpp"
Eigen::Matrix3f K;
// 相机内参
float cx = 325.5;
float cy = 253.5;
float fx = 518.0;
float fy = 519.0;

Keypoints get_points;
void triangulation ( 
    const vector< KeyPoint >& keypoint_1, 
    const vector< KeyPoint >& keypoint_2, 
    const std::vector< DMatch >& matches,
    const Mat& R, const Mat& t, 
    vector< Point3d >& points );

Point2f pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2f
    (
        ( p.x - K.at<double>(0,2) ) / K.at<double>(0,0), //像素坐标系转为相机坐标，先减去中心点的平移，
        ( p.y - K.at<double>(1,2) ) / K.at<double>(1,1) 
    );
}

vector< DMatch > good_matches;//good match 构成一个向量，包含了两张相邻图片间所有点的匹配信息，应该建立一个vector来保存所有匹配信息
                            //dmatch中包含queryIdx、trainIdx、imgIdx、distance四个参数，queryIdx代表的是第几个特征点匹配到了下一张图
                            //的第trainIdx个特征点
Keypoints  feature_extract(deque<Mat> picseries){
    Mat pic1=picseries[0];
    Mat pic2=picseries[1];

    vector<KeyPoint> keypoints1,keypoints2;
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();

    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );

    detector->detect(pic1,keypoints1);
    detector->detect(pic2,keypoints2);

    get_points.keypoints1=keypoints1;
    get_points.keypoints2=keypoints2;
    descriptor->compute(pic1,keypoints1,descriptors_1);
    descriptor->compute(pic2,keypoints2,descriptors_2);
    Descrip get_discrp;
    get_discrp.descriptors_1=descriptors_1;
    get_discrp.descriptors_2=descriptors_1;
    vector<DMatch> matches;

    matcher->match ( descriptors_1, descriptors_2, matches );

    double min_dist=10000, max_dist=0;

    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }    
    min_dist = min_element( matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance;} )->distance;
    max_dist = max_element( matches.begin(), matches.end(), [](const DMatch& m1, const DMatch& m2) {return m1.distance<m2.distance;} )->distance;


    for(int i=0;i<descriptors_1.rows;i++){
        if(matches[i].distance<=max(2*min_dist,30.0)){
            good_matches.push_back(matches[i]);
        }
    }

    Mat img_match;
    Mat img_goodmatch;
    drawMatches ( pic1, keypoints1, pic2, keypoints2, matches, img_match );
    drawMatches ( pic1, keypoints1, pic2, keypoints2, good_matches, img_goodmatch );
    imshow ( "所有匹配点对", img_match );
    imshow ( "优化后匹配点对", img_goodmatch );
    return get_points;
}

RTmatrix pose_estimation_2d2d ( Keypoints get_points,
                            std::vector< DMatch > matches,
                            Mat& R, Mat& t )
{
    Mat K = ( Mat_<double> ( 3,3 ) << 1201.27, 0, 959.5, 0, 1201.27, 539.5, 0, 0, 1 ); 
    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;
    vector<KeyPoint> keypoints_1=get_points.keypoints1;
    vector<KeyPoint> keypoints_2=get_points.keypoints2;
    for ( int i = 0; i < ( int ) matches.size(); i++ )
    {
        points1.push_back ( keypoints_1[matches[i].queryIdx].pt );
        points2.push_back ( keypoints_2[matches[i].trainIdx].pt );
    }

    //-- 计算基础矩阵
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat ( points1, points2, CV_FM_8POINT );
    //cout<<"fundamental_matrix is "<<endl<< fundamental_matrix<<endl;

    //-- 计算本质矩阵
    Point2d principal_point ( 959.5, 539.5 );	//相机光心
    double focal_length = 3.6;			//相机焦距, TUM dataset标定值 像元尺寸 3um 1201.27
    Mat essential_matrix;
    essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point );
    recoverPose ( essential_matrix, points1, points2, R, t, focal_length, principal_point );
    RTmatrix rt;
    rt.R=R;
    rt.t=t;
    vector<Point3d> points;
    triangulation( keypoints_1, keypoints_2, good_matches, R, t, points );
    return rt;
}

void triangulation ( 
    const vector< KeyPoint >& keypoint_1, 
    const vector< KeyPoint >& keypoint_2, 
    const std::vector< DMatch >& matches,
    const Mat& R, const Mat& t, 
    vector< Point3d >& points )
{
    Mat T1 = (Mat_<float> (3,4) <<
        1,0,0,0,
        0,1,0,0,
        0,0,1,0);
    Mat T2 = (Mat_<float> (3,4) <<
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0)
    );

    Mat K = ( Mat_<double> ( 3,3 ) << 1201.27, 0, 959.5, 0, 1201.27, 539.5, 0, 0, 1 );
    vector<Point2f> pts_1, pts_2;
    for ( DMatch m:matches )
    {
        // 将像素坐标转换至相机坐标
        pts_1.push_back ( pixel2cam( keypoint_1[m.queryIdx].pt, K) );
        pts_2.push_back ( pixel2cam( keypoint_2[m.trainIdx].pt, K) );
    }
    
    Mat pts_4d;
    cv::triangulatePoints( T1, T2, pts_1, pts_2, pts_4d );
    
    // 转换成非齐次坐标
    for ( int i=0; i<pts_4d.cols; i++ )
    {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3,0); // 归一化
        Point3d p (
            x.at<float>(0,0), 
            x.at<float>(1,0), 
            x.at<float>(2,0) 
        );
        points.push_back( p );
    }
}

