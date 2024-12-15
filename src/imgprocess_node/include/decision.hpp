#ifndef DECISION_HPP
#define DECISION_HPP

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <vector>
#include <eigen3/Eigen/Dense>
#include "STrack.h"
#include "common.hpp"
#include "classification.h"

enum BallStateColor {BLUE = 0, RED, NONE, PURPLE};
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor> DETECTBOXSS;  

typedef struct BucketColor {
	BucketColor() : order{{2, 0}, {2, 0}, {2, 0}}, load_num(0) {}
    std::string color;			// 桶内颜色的字符显示
    cv::Rect rect;				// 框所在矩形框，用于确定是第几个框
	vector<vector<int>> order;	// vector<int>前一个是框内球的分类，后一个球的y方向高度，用于确定球的顺序，外层vector 0是低，2是高
	int load_num;				// 已经往里填放的球的个数，用于有序放入
	int track_id;
} BucketState;

typedef struct BallColor {
    std::string colorName;			// 桶内颜色的字符显示
    cv::Rect rect;				// 框所在矩形框，用于确定是第几个框
	float depth;
	int color;
	int track_id;
} BallState;

class Decision {
public: 
	Decision() : trackId2Order{1, 2, 3, 4, 5}, correspond2Order(5), orderId(5),
				 bucketState{{2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}, {2, 2, 2}} {
		classifier = new Classification("/home/gray/code/amending/A_RC/r2/ROBOT_R2/models/resNet34Bucket.onnx");
		correspond2Order.clear();
		TargetBall.track_id = -1;
	}
	~Decision() {};

	void GetStateBucket(const cv::Mat& srcImage, const std::vector<STrack>& bucket_inline, const std::vector<det::Object>& objects);
	void GetBallDirection(const cv::Mat& srcImage, const cv::Mat& depthImage, const std::vector<STrack>& ball_inline);

	cv::Mat DrawBucketState(const cv::Mat& frame, std::vector<BucketState>& bucketState);
	cv::Mat Draw(const cv::Mat& frame, const std::vector<STrack>& ball_inline);

	float GetDepthOfBall(const cv::Rect bbox, const cv::Mat depthImg) {return 0;}
public:
	std::vector<std::vector<int>> bucketState;		// 桶的状态保存
	std::vector<int> trackId2Order;					// 跟踪的id号和桶顺序的对应，认为相机正方面向桶的序号从左到右是0~4，每次查询有没有新id号出现，如果有需要重置跟踪器
	std::string trackIdString;
	std::vector<int> orderId;
	std::vector<int> correspond2Order;
	bool reTrackFlag;								// 跟踪id号改变，（已经跟丢或者新的轨迹开始）
	bool correspondFlag;							// 跟踪id号满五个
	std::vector<BucketState> tmpBucketRoi;			// 临时存放桶的Roi位置
	Classification* classifier; 

	std::vector<BallState> tmpBallState;
	BallState TargetBall;
	bool TargetBallFlag;

	int count_;
};

#endif