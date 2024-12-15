#define _CRT_SECURE_NO_WARNINGS

#ifndef CAMERA_NODE_H_
#define CAMERA_NODE_H_

#include <iostream>
#include <fstream>
#include <mutex>
#include <thread>
#include <sstream>
#include <chrono>
#include <memory>
#include <math.h>
#include <time.h>
#include <numeric>
#include <cmath>
#include <vector>
#include <atomic>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include<opencv2/core/eigen.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include <librealsense2/hpp/rs_sensor.hpp>
#include <librealsense2/rs_advanced_mode.hpp>
#include <librealsense2/rs_advanced_mode.h>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/header.h>
#include "common_msgs/msg/depth_color.hpp"

using namespace std;
using namespace cv;
using namespace rs2;
using namespace Eigen;
using namespace rs400;
using namespace std::chrono_literals;
using std::vector;

class Camera_Node : public rclcpp::Node {
public:
	Camera_Node(std::string d435_path, std::string inside_path, std::string outside_path);
	Camera_Node(const Camera_Node&) = delete;
	Camera_Node operator=(const Camera_Node&) = delete;
	~Camera_Node() {outsideCamera.release();};

	void Update(rs2::frameset fs);									
	inline Mat GetSrcImage(void){return srcImage;}							//获得彩色源图像
	inline Mat GetDepthImage(void){return depthImage;}						//获得深度图像

	void getCameraExtrinsics(rs2::pipeline_profile &profiles);
	void GetCameraParam(rs2_intrinsics& _color_intrin, rs2_intrinsics& _depth_intrin, 
						rs2_extrinsics& _depth2color_extrin, rs2_extrinsics& _color2depth_extrin);

	void ProcessInside();

public:
	uint16_t* data;
	rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr d435_pub;
	rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr inside_pub;
	rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr outside_pub;
	rclcpp::Publisher<common_msgs::msg::DepthColor>::SharedPtr d435_pub_dep;
	VideoCapture insideCamera;
	VideoCapture outsideCamera;

private:
	rs2::context 			ctx;
	size_t 					device_count;	
	rs2_intrinsics 			color_intrin;			//颜色相机内参
	rs2_intrinsics 			depth_intrin;			//深度相机内参
	rs2_extrinsics 			depth2color_extrin;		//深度向颜色外参
	rs2_extrinsics 			color2depth_extrin;
	rs2::pointcloud  		rs2Cloud;				//深度图计算得到的点云 RS2指realsense2系列相机
	rs2::points      		rs2Points;				//点云格式的点
	rs2::align       		alignToColor;			//对齐图像
	rs2::pipeline    		pipe;					//数据传输管道
	rs2::config      		cfg;					//初始化
	rs2::frameset    		frameSet;				//帧设置
	rs2::frameset    		alignProcess;
	rs2::colorizer  		color_map;				
	rs2::temporal_filter 	tem_filter;				//时间过滤器
	rs2::spatial_filter 	spat_filter;			//

	size_t firstFrameSet = 5;
	std::string video_d435;
	std::string video_inside;
	std::string video_outside;

	cv::Mat 				srcImage;				//源图片，彩色图片source image
	cv::Mat         		depthImage;				//深度图
};
#endif
