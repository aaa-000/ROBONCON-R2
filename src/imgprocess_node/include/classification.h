#define _CRT_SECURE_NO_WARNINGS

#ifndef CLASSIFICATION_H_
#define CLASSIFICATION_H_

#include <iostream>
#include <fstream>
#include <mutex>
#include <thread>
#include <sstream>
#include <math.h>
#include <time.h>
#include <numeric>
#include <cmath>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <Eigen/Dense>

using namespace cv;
using namespace std;
using namespace dnn;

struct Output
{
    int id;           //结果类别id
    float confidence; //结果置信度
    Rect box;     //矩形框
};

class Classification {
public:
	Classification(const string &path, bool isCuda = true);
	Classification(const Classification&) = delete;
	Classification& operator=(const Classification&) = delete;
	~Classification();

	void ClassInfer(cv::Mat srcImage);															
	void ClassInferBatch(const std::vector<cv::Mat>& srcImages);

	Mat dstImage;
	Mat saveImg;
	void viewer(void);																			// 视觉可视化

	std::vector<std::string> classNames = {"blue", "purple", "red"}; 				      //存放类别名称
	std::vector<std::string> classNamesBatch = {"b", "r", "n"}; 				      //存放类别名称
	Output finalResult;
	vector<Output> finalResultBatch;

//----------------------------------------------------------
private:
	bool ClassInferTarget(cv::Mat srcImage);    
	bool ClassInferTargetBatch(const std::vector<cv::Mat>& images);
    bool readModel(cv::dnn::Net &net, const std::string &netPath, bool isCuda); 				// 读取模型
	void release(void);																			// 数据归零

//-----------------------------------------------------------
private:
	// ClassInferor
    
	bool BallFlag;

	// opencv
	Mat srcImage;
	
	Scalar Color = Scalar(0, 0, 255);

	// Dnn
	Net net;
    float classThreshold = 0.30;                                  		  //置信度阈值
};


#endif

