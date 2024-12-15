#include "camera_node.h"

std::mutex colorDepthMutex;
std::map<int, std::string> stream_names;

#define IFCAMERA
// #define IFCAMERA_INSIDE

Camera_Node::Camera_Node(std::string d435_path = "/home/gray/Videos/20231122_151116.bag", 
						std::string inside_path = "/home/gray/Videos/2.avi", 
						std::string outside_path = "/home/gray/Videos/2.avi"):  
						Node("Camera_Node"), 
						alignToColor(RS2_STREAM_COLOR), 
						video_d435(d435_path),
						video_inside(inside_path),
						video_outside(outside_path) 
{	
	d435_pub = this->create_publisher<sensor_msgs::msg::Image>("/d435_sensor", 10);
	inside_pub = this->create_publisher<sensor_msgs::msg::Image>("/inside_sensor", 10);
	outside_pub = this->create_publisher<sensor_msgs::msg::Image>("/outside_sensor", 10);
	d435_pub_dep = this->create_publisher<common_msgs::msg::DepthColor>("/d435_dep_sensor", 10);

	#ifdef IFCAMERA_INSIDE 
		insideCamera.open(0);
		capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
		capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
		capture.set(cv::CAP_PROP_FPS, 30);
		outsideCamera.open(outside_path);
	#else
		insideCamera.open(video_inside);
	#endif 
	#ifndef IFCAMERA
		cfg.enable_device_from_file(video_d435);
	#else
		auto devices = ctx.query_devices();//获取设备列表
		device_count = devices.size();					//获取传感器连接数量
		cout << "device_count:" << device_count << endl;
		while (!device_count && rclcpp::ok()) { // 持续访问，直到有设备连接为止
			devices = ctx.query_devices();
			device_count = devices.size();
			cout << "device_count:" << device_count << endl;
			cout << "No device detected. Is it plugged in?\n";
		}
		auto dev = devices[0];
		std::ifstream file("src/camera_node/modeJson/short_range.json");  // 加载相机参数设置文件，保存相对路径
		if (file.good()) {
			std::string str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
			auto prof = cfg.resolve(pipe);
			if (auto advanced_mode_dev = prof.get_device().as<rs2::serializable_device>()) {
				advanced_mode_dev.load_json(str);
			} else {
				cout << "Current device doesn't support advanced-mode!\n";
				return;
			}
		}
		cfg.disable_all_streams();												// 失能所有数据流
		cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);      // 使能深度相机输入流
		cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);		// 使能彩色相机输入流
	#endif

	rs2::pipeline_profile profiles = pipe.start(cfg);
	#ifdef IFCAMERA
		auto colorSensors = profiles.get_device().query_sensors()[1];
	#endif
	getCameraExtrinsics(profiles);
	for (int i = 0; i < 10; i++) {		
		frameSet = pipe.wait_for_frames();	
	}
	pipe.stop();
	profiles = pipe.start(cfg, [&](const rs2::frame& frame) 
		{
			rs2::frameset fs = frame.as<rs2::frameset>();
			Update(fs);
		});
    for (auto p : profiles.get_streams()) {
		stream_names[p.unique_id()] = p.stream_name();
	}
    std::cout << "RealSense callback sample" << std::endl;
}

inline void Camera_Node::Update(rs2::frameset fs) 
{
	alignProcess = alignToColor.process(fs);
	rs2::video_frame colorFrame = fs.get_color_frame();				// 获取彩色帧
	rs2::depth_frame alignedDepthFrame = alignProcess.get_depth_frame();

	srcImage = cv::Mat(cv::Size(640, 480), CV_8UC3, (void*)colorFrame.get_data(), cv::Mat::AUTO_STEP);				// 将彩色图像数据存储在Mat矩阵中
	depthImage = cv::Mat(cv::Size(640, 480), CV_16UC1, (void*)alignedDepthFrame.get_data(), cv::Mat::AUTO_STEP);	// 将深度图像数据存储在Mat矩阵中

	auto image_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", srcImage).toImageMsg();
	d435_pub->publish(*image_msg);

	auto dep_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "16UC1", depthImage).toImageMsg();
	auto depColor_msg = common_msgs::msg::DepthColor();
	depColor_msg.color = *image_msg;
	depColor_msg.depth = *dep_msg;
	d435_pub_dep->publish(depColor_msg);

	rclcpp::Time now = this->get_clock()->now();
	// RCLCPP_INFO(this->get_logger(), "camera talk: sec:%lf nano:%ld", now.seconds());
}

void Camera_Node::getCameraExtrinsics(rs2::pipeline_profile &profiles) 
{
	const rs2::stream_profile color_profile = profiles.get_stream(RS2_STREAM_COLOR);
    const rs2::stream_profile depth_profile = profiles.get_stream(RS2_STREAM_DEPTH);
    depth_intrin = depth_profile.as<rs2::video_stream_profile>().get_intrinsics();
    color_intrin = color_profile.as<rs2::video_stream_profile>().get_intrinsics();
    depth2color_extrin = depth_profile.as<rs2::video_stream_profile>().get_extrinsics_to(color_profile);
    color2depth_extrin = color_profile.as<rs2::video_stream_profile>().get_extrinsics_to(depth_profile);
}

void Camera_Node::GetCameraParam(rs2_intrinsics& _color_intrin, rs2_intrinsics& _depth_intrin, 
					rs2_extrinsics& _depth2color_extrin, rs2_extrinsics& _color2depth_extrin) 
{
	// std::lock_guard<std::mutex> lock(colorDepthMutex);
	_color_intrin = color_intrin;
    _depth_intrin = depth_intrin;
    _depth2color_extrin = depth2color_extrin;
    _color2depth_extrin = color2depth_extrin;
}

void Camera_Node::ProcessInside() 
{
	while(rclcpp::ok()) {
		Mat frame;
		insideCamera.read(frame);
		if (frame.empty()) {
			break;
		}
		auto image_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", frame).toImageMsg();
        this->inside_pub->publish(*image_msg);
		rclcpp::Time now = this->get_clock()->now();
		// RCLCPP_INFO(this->get_logger(), "camera talk: sec:%lf nano:%ld", now.seconds());
		waitKey(100);
	}
	insideCamera.release();
}

int main(int argc, char* argv[]) 
{
	rclcpp::init(argc, argv);
	auto node = std::make_shared<Camera_Node>("/home/gray/Videos/20231122_151116.bag", "/home/gray/Videos/2.avi", "/home/gray/Videos/2.avi");		
	std::thread process(&Camera_Node::ProcessInside, node);	 
	rclcpp::spin(node);
	rclcpp::shutdown();
	return 0;
}
