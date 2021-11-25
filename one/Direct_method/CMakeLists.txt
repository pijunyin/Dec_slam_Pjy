cmake_minimum_required(VERSION 2.8)
project(directmethod)
set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
set(OpenCV_DIR /opt/homebrew/Cellar/opencv@3/3.4.15/share/OpenCV)
find_package( OpenCV 3.1 REQUIRED )

include_directories( 
    ${OpenCV_INCLUDE_DIRS} 
    "/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3"
)
add_executable( main main.cpp  )

