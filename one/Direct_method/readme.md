

# <font size=6 font color=black><center>g2o与图优化</center></font>

## <font size=5>图优化</font>

​		图优化主要是后端优化求解空间点和位姿的时候使用，将常规的优化问题转换为图的形式来表述从而优化，在求解时采用GN方法或者LM方法。 

​		图是由顶点（vertex）和边（edge）组成，**顶点是需要优化的变量，边表示顶点间的关系**，边可以是有向也可以是无向的，当图中存在连接了两个以上顶点的边时，就称为超图。

​		将Slam问题转化为图：首先知道Slam的核心是根据已有的观测数据，计算机器人的运动轨迹和地图。机器人在某个位置观测了路标，则会产生一个观测值，这个观测值跟观测方程计算出来的结果存在差异。

​		在图优化的过程中有多种参数表达和观测方程的形式，如果我们以机器人的状态 $x_{k}$  作为优化变量，以最小化误差函数为目标函数，就可以求得一个$x$的估计值。我们可以选择待求量为机器人的位姿（4*4的变换矩阵）也可以选择空间点（三维坐标或者二维坐标）。在选择观测方程也就是图优化中的边时，也有多种选择：

- 当图优化的顶点（优化量）为pose时，边选择为$T_{1}=\triangle T*T_{2}$.

- 当能得到目标点的像素坐标时，那么顶点为3d姿态，包括T和一个空间点 x=(x,y,z)，观测数据为像素坐标[u,v],观测方程为$z=K(RX+t)$ .   

此时已经得到了图优化的边和顶点，对于一个带有n条边的图，其目标函数可以表示为：
$$
\mathop{\min}\limits_x \sum\limits_{k = 1}^n e_{k}(x_{k},z_{k})^T\omega_{k}  e_{k}(x_{k},z_{k}) \tag{1}
$$
e表示的是误差，包含了之前所说的带优化变量x和观测方程z，信息矩阵是协方差的逆，表示对相关误差项对一个估计。此时如果一个相机以pose $T_{k}$对空间点$X_{k}$  进行了一次观测，得到了$Z_{k}$，那么这条二元边表示为：
$$
e_{k}(x_{k},T_{k},z_{k})=(z_{k}-K(Rx_{k}+t))^T\omega_{k}(z_{k}-K(Rx_{k})-t) \tag{2}
$$
之后采用GN方法用增量式求解。

对于单目相机，可以通过双目相机对极几何求解出Rt，得到了我们图优化的顶点，我们可以求出深度，那么就得到了空间点，用上述的观测方程建立优化关系。

使用g2o图优化需要维护*sparseOptimizer*，需要为其设定优化算法求解器，同时也需要为其指定顶点、边。

---

## g2o使用与设计		

​	下图下半部分表示了为优化器选择算法和求解器。首先要新建一个线性求解器，然后要为其指定一个*optimizerAlgorithm*，这个继承自GN、LM和Pd，我们三选一

```c++
typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> DirectBlock;
    DirectBlock::LinearSolverType* linearSolver=new 		           					  g2o::LinearSolverDense<DirectBlock::PoseMatrixType>();
    DirectBlock* solver_ptr=new DirectBlock(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver=new                       		g2o::OptimizationAlgorithmLevenberg(solver_ptr);
```

​		从上图中还可以看到这个线性求解器还需要包含两个步骤：

- 需要SparseBlockMatrix求解$H\triangle x=-b$ 中的$H$和$b$
- 需要BlockSolver求解上式，求解器可以从上述三选一

```c++
		g2o::VertexSE3Expmap* pose=new g2o::VertexSE3Expmap();
    pose->setEstimate(g2o::SE3Quat(Tcw.rotation(), Tcw.translation()));
    pose->setId(0);
    optimizer.addVertex ( pose );
```

设置好了求解器、选择好了Blocksolver和迭代策略就需要根据具体问题求解双视图问题了。

---

<img src="https://images2015.cnblogs.com/blog/606958/201603/606958-20160321233900042-681579456.png" alt="g" style="zoom:75%;" />

# <center>单目半稠密直接法</center>

## 单目图像采集

​	需要获得多个视图才能进行稀疏特征点直接法。

```c++
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
```

​	用长度为2的队列存储收到的两张图像。在得到两张图像后需要提取图像中的特征点，这里选择fast特征点：

```cc
    vector<cv::KeyPoint> keypoints1;
    cv::Ptr<FastFeatureDetector>       					             																						     				detector=cv::FastFeatureDetector::create();
    detector->detect(pic1,keypoints1);
```

​	对于取特征点我们只需要在第一张图中求取即可，因为半稠密直接法从第一帧图像提取部分特征点，之后用这些点进行直接法位姿估计。

​	接下来需要建立图优化关系，我们先按照之前的操作指定优化算法和求解器，选择优化算法为LM，求解器为线性求解器。配置好了后需要往g2o中指定顶点和边。

```c++
		g2o::VertexSE3Expmap* pose=new g2o::VertexSE3Expmap();//initialize a pose to add into vertex
    pose->setEstimate(g2o::SE3Quat(Tcw.rotation(), Tcw.translation()));
    pose->setId(0);
    optimizer.addVertex ( pose );
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
        edge->setInformation ( Eigen::Matrix<double,1,1>::Identity() );
        edge->setId ( id++ );
        optimizer.addEdge ( edge );  
    }
```

​	添加了一个顶点和这个顶点对应的多条边，顶点设定为位姿，数个边代表数个光度误差度量。直接法定义边误差定义的代码如下：

```c++
    virtual void computeError()
    {
        const VertexSE3Expmap* v  =static_cast<const VertexSE3Expmap*> ( _vertices[0] );
        Eigen::Vector3d x_local = v->estimate().map ( x_world_ );
        // float x = x_local[0]*fx_/x_local[2] + cx_;//将这个空间点投影到下一张图的相同位置，其实都不需要三维点
        // float y = x_local[1]*fy_/x_local[2] + cy_;
        float x = x_local[0];//将这个空间点投影到下一张图的相同位置，其实都不需要三维点
        float y = x_local[1];
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
```

​	分两种情况计算了边带来的误差，如果超出图像边界的话设置误差为0，图优化的目的就是在多种情况下最小化这个误差。完成了误差计算的定义后还需要定义雅各比矩阵计算方法，用以更新位姿。我们误差定义如下：
$$
e(\xi \oplus \delta \xi)=e(\xi)-\frac{\partial I_{2}}{\partial u}\frac{\partial u}{\partial q}\frac{\partial q}{\partial \delta \xi} \delta \xi
$$
​	上述右边部分$\frac{\partial u}{\partial q}$是图像梯度，$u$代表二维坐标，$\frac{\partial q}{\partial \delta \xi}$是第二个图上坐标对李代数的求导。这个关系式表达的是第二张图上的投影坐标对位姿的关系，进而我们可以调整位姿。但在设置雅各比矩阵的时候我们需要设置深度为1，因为单纯单目相机不能获取深度信息，简化后会影响收敛速度和精度。定义好雅各比矩阵之后我们对g2o的设置就结束了。

---

​	现在开始主程序设计。我们首先需要读取图片，代码参考`getcampic`，得到了图片以后，由于我们采用的是稀疏光流法，需要先在第一帧提取部分特征点，之后的计算都基于这一帧的位姿计算。将得到的特征点全部放入观测中，用于之后的计算。

```c++
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
  float zz=5000;//由于单目相机没有深度，假设了一个深度，这样会影响直接法收敛的速度和精度。目前LSDslam通过建立关键帧然后初始化的方法也能获取深度，我简化了。
  //cout<<points[i].x<<endl;
  pts=Eigen::Vector3d ( xx, yy, zz );
  measurements.push_back ( Measurement ( pts, grayscale ) );//point3d Eigen::Vector3d 将现在这张图上的点送入观测
}
```

​		设置完成之后 `pose_estimate_direct ( measurements, &gray_move, K, Tcw ); `就可以开始优化了.

​		目前出现的问题是编译不报错了，但是执行的时候会报`segmentation fault`我目前正在使用GDB工具调试。

---

# <center>总结</center>

​		单目相机实现直接法首先我取了第一帧图像作为基准，在这帧图像上提取处特征点用于之后的直接法位姿估计。在图优化时因为我们只有一个待优化变量所以只需要建立一个vertex，每个点匹配需要添加一条边，直接法时这个边需要设置为像素点点光度误差。

​		由于单目相机没有深度信息，但我们在计算像素坐标与李代数导数时需要用到深度信息，所以我用一个常数作为替代和简化，单纯的单目相机不能实现直接法，但是已经有LSM-slam使用建立关键帧然后初始化得深度的方法实现了，相当于获得了深度信息，那么也就不需要简化了。

​		在跑这个程序时记得改include和cmakelist里面的地址，不然肯定报错。
