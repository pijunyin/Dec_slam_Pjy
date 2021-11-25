/*
This program is about using direct method to pose estimate.(mono)
Open your computer camera and output R t related to initial pose.
                                                                                by pjy
*/
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace std;
using namespace cv;
int main(){
    VideoCapture cap;
    cap.open(0);
    while(true){
        Mat frame;
       cap>>frame;
       imshow("pic",frame);
        int p=waitKey(1);
        if(p=='c')
            break;
    }
    return 0;
}