cmake_minimum_required(VERSION 2.9)
project(direct_method)

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )
#set( CMAKE_CXX_FLAGS "-Wno-error=deprecated-declarations -Wno-deprecated-declarations ")


list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
set(OpenCV_DIR /opt/homebrew/Cellar/opencv@3/3.4.15/share/OpenCV)
find_package(OpenCV 3.1 REQUIRED)
find_package( G2O REQUIRED )
find_package( CSPARSE REQUIRED )
find_package( Sophus REQUIRED )
include_directories(
    ${OpenCV_INCLUDE_DIRS} 
    ${G2O_INCLUDE_DIRS}
    ${Sophus_INCLUDE_DIRS} 
    ${CSPARSE_INCLUDE_DIR}
    "/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3"
    "/opt/homebrew/Cellar/g2o/2017-07-30/include"
    "/usr/local/include"
    "/usr/local/include/g2o"
)
link_directories(
    #"/usr/local/lib"
    "/opt/homebrew/Cellar/g2o/2017-07-30/lib"
)
set( G2O_LIBS 
    g2o_core g2o_types_sba g2o_solver_csparse g2o_stuff g2o_csparse_extension 
)

add_library(pic_process ./src/pic_process.cpp)
add_library(direct ./src/direct.cpp)
add_executable(main ./src/main.cpp)
target_link_libraries(main 
    ${OpenCV_LIBS}
    ${G2O_LIBS}
    direct
    pic_process  
    )