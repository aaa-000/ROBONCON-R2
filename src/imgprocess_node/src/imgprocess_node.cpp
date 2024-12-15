#include <sstream>
#include <memory>
#include <fstream>
#include <dirent.h>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "yolov8.hpp"
#include "BYTETracker.h"
#include "decision.hpp"
#include "classification.h"
#include "common_msgs/msg/depth_color.hpp"

class Imgprocess_Node : public rclcpp::Node {
public:
    Imgprocess_Node() : Node("Detect") {
        d435_sub = this->create_subscription<common_msgs::msg::DepthColor>(
            "/d435_dep_sensor", 10, std::bind(&Imgprocess_Node::imageD435Callback, this, std::placeholders::_1));
		
		inside_sub = this->create_subscription<sensor_msgs::msg::Image>(
            "/inside_sensor", 10, std::bind(&Imgprocess_Node::imageInsideCallback, this, std::placeholders::_1));
		
		outside_sub = this->create_subscription<sensor_msgs::msg::Image>(
            "/outside_sensor", 10, std::bind(&Imgprocess_Node::imageOutsideCallback, this, std::placeholders::_1));		
		
		bucket_d435_result = this->create_publisher<sensor_msgs::msg::Image>("/bucket_result", 10);

		ball_inside_result = this->create_publisher<sensor_msgs::msg::Image>("/ball_result", 10);
		
		bucket_outside_result = this->create_publisher<sensor_msgs::msg::Image>("/outside_result", 10);

		cudaSetDevice(0);
		yolov8 = new YOLOv8("./models/1-10.engine");
		yolov8->make_pipe(true);

		judge = new Decision();

		d435_bucket_ball = 0;

		int fps=60;
		bytetracker = new BYTETracker(fps, 30);

		classifier = new Classification("./models/resnetBall.onnx");

		if (NULL == opendir("data")) {
			::mkdir("data", 0777);
		}
		time_t currentTime;
        time(&currentTime);
        currentTime     = currentTime + 8 * 3600;
        tm *t           = gmtime(&currentTime);
        std::string filename = "data/des" + cv::format("%.2d%.2d\n%.2d:%.2d", t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min) + ".txt";
        outfile_ctrl.open(filename.c_str());
		RCLCPP_INFO(this->get_logger(), "make pipe ok!!!");
    }

	~Imgprocess_Node() {
		delete yolov8;
		delete judge;
		delete bytetracker;
		delete classifier;
	}

private:
    void imageD435Callback(const common_msgs::msg::DepthColor::SharedPtr msg) {
        try {
			if (!d435_bucket_ball) {
				auto t1 = rclcpp::Clock().now();
				auto start = rclcpp::Clock().now();
				cv::Mat srcImage = cv_bridge::toCvCopy(msg->color, "bgr8")->image;
				yolov8->detect(srcImage);
				auto output_stracks = bytetracker->update(yolov8->objs_bucket);
				judge->GetStateBucket(srcImage, output_stracks, yolov8->objs_ball);
				cv::Mat showImage = judge->DrawBucketState(srcImage, judge->tmpBucketRoi);
				
				auto image_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", showImage).toImageMsg();
				bucket_d435_result->publish(*image_msg);

				auto end = rclcpp::Clock().now();
				RCLCPP_INFO(this->get_logger(), "[single circle time] sec:%lf", (end.seconds() - start.seconds())*1000);
				outfile_ctrl << t1.nanoseconds() << std::endl;
			} else {
				auto t1 = rclcpp::Clock().now();
				auto start = rclcpp::Clock().now();
				cv::Mat srcImage = cv_bridge::toCvCopy(msg->color, "bgr8")->image;
				cv::Mat depthImage = cv_bridge::toCvCopy(msg->depth, "16UC1")->image;
				yolov8->detect(srcImage);
				auto output_stracks = bytetracker->update(yolov8->objs_ball);
				judge->GetBallDirection(srcImage, depthImage, output_stracks);
				cv::Mat showImage = judge->Draw(srcImage, output_stracks);

				auto image_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", showImage).toImageMsg();
				bucket_d435_result->publish(*image_msg);

				auto end = rclcpp::Clock().now();
				RCLCPP_INFO(this->get_logger(), "[single circle time] sec:%lf", (end.seconds() - start.seconds())*1000);
				outfile_ctrl << t1.nanoseconds() << std::endl;
			}
        } catch (const cv_bridge::Exception& e) {
            // RCLCPP_ERROR(this->get_logger(), "CV Bridge Exception: %s", e.what());
        }
    }

	void imageInsideCallback(const sensor_msgs::msg::Image::SharedPtr msg) {      
		try {
			auto start = rclcpp::Clock().now();
			cv::Mat srcImage = cv_bridge::toCvCopy(msg, "bgr8")->image;
			Mat tmpM;
			srcImage(cv::Rect(320, 140, 200, 200)).copyTo(tmpM);
			classifier->ClassInfer(tmpM);

			std::string finnalLabel = classifier->classNames[classifier->finalResult.id] + ":" + to_string(classifier->finalResult.confidence);
			int finnalBaseLine;
			Size finnalLabelSize = getTextSize(finnalLabel, FONT_HERSHEY_SIMPLEX, 0.5, 1, &finnalBaseLine);
			int left = 50, top = 50;
			putText(srcImage, finnalLabel, Point(left, top), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);

			auto image_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", srcImage).toImageMsg();
			ball_inside_result->publish(*image_msg);

			auto end = rclcpp::Clock().now();
			// RCLCPP_INFO(this->get_logger(), "[single circle time] sec:%lf", (end.seconds() - start.seconds())*1000);

		} catch (const cv_bridge::Exception& e) {
            // RCLCPP_ERROR(this->get_logger(), "CV Bridge Exception: %s", e.what());
        }
    }

	void imageOutsideCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
			auto start = rclcpp::Clock().now();

			auto end = rclcpp::Clock().now();
			// RCLCPP_INFO(this->get_logger(), "[single circle time] sec:%lf", (end.seconds() - start.seconds())*1000);

        } catch (const cv_bridge::Exception& e) {
            // RCLCPP_ERROR(this->get_logger(), "CV Bridge Exception: %s", e.what());
        }
    }

    rclcpp::Subscription<common_msgs::msg::DepthColor>::SharedPtr d435_sub;
	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr inside_sub;
	rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr outside_sub;
	
	rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr bucket_d435_result;
	rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr ball_inside_result;
	rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr bucket_outside_result;

	bool d435_bucket_ball; // d435看桶还是看球的标志位，0表示看桶，1表示看球，可以由运控给

	YOLOv8* yolov8;
	Decision* judge;
	BYTETracker* bytetracker;
	Classification* classifier;   
	std::ofstream outfile_ctrl;
};

int main(int argc, char* argv[]) {
	rclcpp::init(argc, argv);
	auto node = std::make_shared<Imgprocess_Node>();
    rclcpp::spin(node);
	rclcpp::shutdown();
	return 0;
}
