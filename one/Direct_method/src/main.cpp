#include <iostream>
#include <vector>
#include <deque>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "../include/pic_process.hpp"
#include "../include/direct.hpp"
using namespace std;
using namespace cv;
using namespace g2o;
deque<Mat> getcampic(VideoCapture cap,deque<Mat> picseries){
    Mat frame;
    cap>>frame;
    if(picseries.size()==2){
        picseries.pop_front();
        picseries.push_back(frame);
    }else{
        picseries.push_back(frame);
    }
    return picseries;
}

int main(){
    VideoCapture cap;
    deque<Mat> picseries;
    vector<Measurement> measurements;
    Keypoints keypoints;
    RTmatrix rt;
    cap.open(0);
    // Mat K = ( Mat_<double> ( 3,3 ) << 1201.27, 0, 959.5, 0, 1201.27, 539.5, 0, 0, 1 );
    float cx = 959.5;
    float cy = 539.5;
    float fx = 1201.27;
    float fy = 1201.27;
    Eigen::Matrix3f K;
    int index=0;
    K<<fx,0.f,cx,0.f,fy,cy,0.f,0.f,1.0f;
    Eigen::Isometry3d Tcw = Eigen::Isometry3d::Identity();
    while(true){
        picseries=getcampic(cap,picseries);
        Mat gray,gray_move;
        cv::Mat prev_color;
        if(picseries.size()==2)
        {
            
            Mat pic1=picseries[0];
            Mat static_pic;
            Mat pic2=picseries[1];
            cv::cvtColor ( pic1, gray, cv::COLOR_BGR2GRAY );
            cv::cvtColor ( pic2, gray_move, cv::COLOR_BGR2GRAY );
            if(index==0){ 
                static_pic=pic1;
                keypoints=feature_extract(picseries);//include two pics' keypoints
                //rt=pose_estimation_2d2d ( keypoints,good_matches,rt.R, rt.t );//related move between two pics
                
                //for ( auto kp:keypoints.keypoints1 )//all points in a pic
                for(int i=0;i<keypoints.keypoints1.size();i++)
                {
                    // vector<Point3d> points;   
                    // triangulation ( keypoints.keypoints1, keypoints.keypoints2,good_matches,rt.R, rt.t, points );
                    // 去掉邻近边缘处的点
                    if ( keypoints.keypoints1[i].pt.x < 20 || keypoints.keypoints1[i].pt.y < 20 || ( keypoints.keypoints1[i].pt.x+20 ) >pic1.cols || ( keypoints.keypoints1[i].pt.y+20 ) >pic1.rows )
                        continue;
                    float grayscale = float ( gray.ptr<uchar> ( cvRound ( keypoints.keypoints1[i].pt.y ) ) [ cvRound ( keypoints.keypoints1[i].pt.x ) ] );
                    Eigen::Vector3d pts;//points
                    float xx=keypoints.keypoints1[i].pt.x;
                    float yy=keypoints.keypoints1[i].pt.y;
                    float zz=1;
                    //cout<<points[i].x<<endl;
                    pts=Eigen::Vector3d ( xx, yy, zz );
                    measurements.push_back ( Measurement ( pts, grayscale ) );//point3d Eigen::Vector3d 将现在这张图上的点送入观测
                }
                prev_color = pic1.clone();
                index=1;
            }

            pose_estimate_direct ( measurements, &gray_move, K, Tcw );//送入了现在这张图上的点的观测和来的下一张图
            cout<<"Tcw="<<Tcw.matrix() <<endl;

            cv::Mat img_show ( static_pic.rows*2, static_pic.cols, CV_8UC3 );
            prev_color.copyTo ( img_show ( cv::Rect ( 0,0,static_pic.cols, static_pic.rows ) ) );
            pic1.copyTo ( img_show ( cv::Rect ( 0,static_pic.rows,static_pic.cols, static_pic.rows ) ) );
            // for ( Measurement m:measurements )
            // {
            //     if ( rand() > RAND_MAX/5 )
            //         continue;
            //     Eigen::Vector3d p = m.pos_world;
            //     Eigen::Vector2d pixel_prev = project3Dto2D ( p ( 0,0 ), p ( 1,0 ), p ( 2,0 ), fx, fy, cx, cy );
            //     Eigen::Vector3d p2 = Tcw*m.pos_world;
            //     Eigen::Vector2d pixel_now = project3Dto2D ( p2 ( 0,0 ), p2 ( 1,0 ), p2 ( 2,0 ), fx, fy, cx, cy );
            //     if ( pixel_now(0,0)<0 || pixel_now(0,0)>=pic1.cols || pixel_now(1,0)<0 || pixel_now(1,0)>=pic1.rows )
            //         continue;

            //     float b = 255*float ( rand() ) /RAND_MAX;
            //     float g = 255*float ( rand() ) /RAND_MAX;
            //     float r = 255*float ( rand() ) /RAND_MAX;
            //     cv::circle ( img_show, cv::Point2d ( pixel_prev ( 0,0 ), pixel_prev ( 1,0 ) ), 8, cv::Scalar ( b,g,r ), 2 );
            //     cv::circle ( img_show, cv::Point2d ( pixel_now ( 0,0 ), pixel_now ( 1,0 ) +pic1.rows ), 8, cv::Scalar ( b,g,r ), 2 );
            //     cv::line ( img_show, cv::Point2d ( pixel_prev ( 0,0 ), pixel_prev ( 1,0 ) ), cv::Point2d ( pixel_now ( 0,0 ), pixel_now ( 1,0 ) +pic1.rows ), cv::Scalar ( b,g,r ), 1 );
            // }
            // cv::imshow ( "result", img_show );
        }

        

        int p = waitKey(50);
        if(p=='q')
            break;
        }
    return 0;
}