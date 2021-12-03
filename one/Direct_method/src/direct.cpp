#include "../include/direct.hpp"
#include "../include/pic_process.hpp"
using namespace std;
using namespace cv;
using namespace g2o;


//start a edge connect vertex and set error be photometric error(direct method)
// an unary edge with one vertex SE3Expmap (the pose of camera)
class EdgeSE3ProjectDirect: public BaseUnaryEdge< 1, double, VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectDirect() {}

    EdgeSE3ProjectDirect ( Eigen::Vector3d point, float fx, float fy, float cx, float cy, cv::Mat* image )
        : x_world_ ( point ), fx_ ( fx ), fy_ ( fy ), cx_ ( cx ), cy_ ( cy ), image_ ( image )
    {}
    
    virtual void computeError()
    {
        const VertexSE3Expmap* v  =static_cast<const VertexSE3Expmap*> ( _vertices[0] );
        Eigen::Vector3d x_local = v->estimate().map ( x_world_ );
        // float x = x_local[0]*fx_/x_local[2] + cx_;//将这个空间点投影到下一张图的相同位置，其实都不需要三维点
        // float y = x_local[1]*fy_/x_local[2] + cy_;
        float x = x_local[0];//将这个空间点投影到下一张图的相同位置，其实都不需要三维点
        float y = x_local[1];
        //cout<<"x: "<<x_local[0]<<endl;
        // check x,y is in the image
        if ( x-4<0 || ( x+4 ) >image_->cols || ( y-4 ) <0 || ( y+4 ) >image_->rows )//去除太靠边的二维点
        {
            _error ( 0,0 ) = 0.0;
            this->setLevel ( 1 );
            
        }
        else
        {
            _error ( 0,0 ) = getPixelValue ( x,y ) - _measurement;//将得到的目标坐标灰度值与原坐标相减
        }
    }

    // plus in manifold
    virtual void linearizeOplus( )
    {
        if ( level() == 1 )
        {
            _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
            return;
        }
        VertexSE3Expmap* vtx = static_cast<VertexSE3Expmap*> ( _vertices[0] );
        Eigen::Vector3d xyz_trans = vtx->estimate().map ( x_world_ );   // q in book
        //返回当前顶点的状态
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double invz = 1.0/xyz_trans[2];
        double invz_2 = invz*invz;

        // float u = x*fx_*invz + cx_;
        // float v = y*fy_*invz + cy_;
        float u = x;
        float v = y;
        // jacobian from se3 to u,v
        // NOTE that in g2o the Lie algebra is (\omega, \epsilon), where \omega is so(3) and \epsilon the translation
        Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;

        jacobian_uv_ksai ( 0,0 ) = - x*y*invz_2 *fx_;
        jacobian_uv_ksai ( 0,1 ) = ( 1+ ( x*x*invz_2 ) ) *fx_;
        jacobian_uv_ksai ( 0,2 ) = - y*invz *fx_;
        jacobian_uv_ksai ( 0,3 ) = invz *fx_;
        jacobian_uv_ksai ( 0,4 ) = 0;
        jacobian_uv_ksai ( 0,5 ) = -x*invz_2 *fx_;

        jacobian_uv_ksai ( 1,0 ) = - ( 1+y*y*invz_2 ) *fy_;
        jacobian_uv_ksai ( 1,1 ) = x*y*invz_2 *fy_;
        jacobian_uv_ksai ( 1,2 ) = x*invz *fy_;
        jacobian_uv_ksai ( 1,3 ) = 0;
        jacobian_uv_ksai ( 1,4 ) = invz *fy_;
        jacobian_uv_ksai ( 1,5 ) = -y*invz_2 *fy_;

        Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;

        jacobian_pixel_uv ( 0,0 ) = ( getPixelValue ( u+1,v )-getPixelValue ( u-1,v ) ) /2;
        jacobian_pixel_uv ( 0,1 ) = ( getPixelValue ( u,v+1 )-getPixelValue ( u,v-1 ) ) /2;

        _jacobianOplusXi = jacobian_pixel_uv*jacobian_uv_ksai;
    }

    // dummy read and write functions because we don't care...
    virtual bool read ( std::istream& in ) {return 0;}
    virtual bool write ( std::ostream& out ) const {return 0;}

protected:
    // get a gray scale value from reference image (bilinear interpolated)
    inline float getPixelValue ( float x, float y )//从新给的图上寻找目标坐标的灰度值
    {
        uchar* data = & image_->data[ int ( y ) * image_->step + int ( x ) ];
        float xx = x - floor ( x );
        float yy = y - floor ( y );
        return float (
                   ( 1-xx ) * ( 1-yy ) * data[0] +
                   xx* ( 1-yy ) * data[1] +
                   ( 1-xx ) *yy*data[ image_->step ] +
                   xx*yy*data[image_->step+1]
               );
    }
public:
    Eigen::Vector3d x_world_;   // 3D point in world frame
    float cx_=0, cy_=0, fx_=0, fy_=0; // Camera intrinsics
    cv::Mat* image_=nullptr;    // reference image
};

int vertexid=0,terms=0;
void pose_estimate_direct(const vector< Measurement >& measurements, cv::Mat* gray, Eigen::Matrix3f& K, Eigen::Isometry3d& Tcw){
    //initial a optimizer
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> DirectBlock;
    DirectBlock::LinearSolverType* linearSolver=new g2o::LinearSolverDense<DirectBlock::PoseMatrixType>();//1:using linear optimizer
    DirectBlock* solver_ptr=new DirectBlock(linearSolver);//2:using optimizer above to start a blocksolver
    //3:after initialize is settings:chose LM method 4:sparse optimizer
    g2o::OptimizationAlgorithmLevenberg* solver=new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm ( solver );
    optimizer.setVerbose( false);
    //5:add vertex and edge
    g2o::VertexSE3Expmap* pose=new g2o::VertexSE3Expmap();//initialize a pose to add into vertex

    pose->setEstimate(g2o::SE3Quat(Tcw.rotation(), Tcw.translation()));
    pose->setId(0);
    optimizer.addVertex ( pose );//将第一张图的位姿设置为初始位姿

    //如果是使用RGBD相机的自然得到了3D点，放入g2o中优化，同时需要输入灰度图像
    int id=1;
    for(Measurement m: measurements){
        //define an edge
        EdgeSE3ProjectDirect* edge=new EdgeSE3ProjectDirect(
            m.pos_world,//means 3d point
            K ( 0,0 ), K ( 1,1 ), K ( 0,2 ), K ( 1,2 ), gray
        );
        edge->setVertex ( 0, pose );//旋转和平移
        edge->setMeasurement ( m.grayscale );
        edge->setInformation ( Eigen::Matrix<double,1,1>::Identity() );//
        edge->setId ( id++ );
        optimizer.addEdge ( edge ); 
    }
   // cout<<"edges in graph: "<<optimizer.edges().size() <<endl;
    optimizer.initializeOptimization();
    optimizer.optimize ( 500 );
    Tcw = pose->estimate();
}

Eigen::Vector3d project2Dto3D ( int x, int y, int d, float fx, float fy, float cx, float cy, float scale )
{
    float zz = float ( d ) /scale;
    float xx = zz* ( x-cx ) /fx;
    float yy = zz* ( y-cy ) /fy;
    return Eigen::Vector3d ( xx, yy, zz );
}

Eigen::Vector2d project3Dto2D ( float x, float y, float z, float fx, float fy, float cx, float cy )
{
    float u = x;
    float v = y;
    return Eigen::Vector2d ( u,v );
}