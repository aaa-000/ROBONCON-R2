#include "decision.hpp"

void Decision::GetStateBucket(const cv::Mat& srcImage, const std::vector<STrack>& bucket_inline, const std::vector<det::Object>& objects) 
{
	if (!bucket_inline.size()) {
		return;
	}
	tmpBucketRoi.clear();
	correspond2Order.clear();

	for (auto i : bucket_inline) {
		BucketState tmp;
		tmp.rect = cv::Rect(i.tlwh.at(0), i.tlwh.at(1), i.tlwh.at(2), i.tlwh.at(3));
		tmp.track_id = i.track_id;
		tmpBucketRoi.push_back(tmp);
		correspond2Order.push_back(i.track_id);
	}

	sort(tmpBucketRoi.begin(), tmpBucketRoi.end(), [](const BucketState &p0, const BucketState &p1)
		{
			return abs(p0.rect.tl().x) < abs(p1.rect.tl().x);
		});

	if (correspondFlag) { // 判断有没有新的id产生
		orderId.assign(trackId2Order.begin(), trackId2Order.end());
		sort(orderId.begin(), orderId.end());
		sort(correspond2Order.begin(), correspond2Order.end());
		if (correspond2Order.back() > orderId.back()) {
			correspondFlag = false;
		}
	}

	if (!correspondFlag) { // 判断新的id是否是五个
		if (correspond2Order.size() < 5) {
			return;
		} else if (correspond2Order.size() == 5) {
			trackIdString.clear();
			for (int i = 0; i < trackId2Order.size(); i++) {
				trackId2Order.at(i) = tmpBucketRoi.at(i).track_id;
				trackIdString = trackIdString + cv::format("%d ", trackId2Order.at(i));
			}
			correspondFlag = true;
		}
	}

	if (!objects.size()) {
		return;
	}
	
	std::vector<cv::Mat> tmpV;
	std::vector<Output> classResult;
	cv::Mat dstImage;
	srcImage.copyTo(dstImage);
	for (auto i : objects) {
		cv::Mat tmp;
		dstImage(i.rect).copyTo(tmp);
		tmpV.push_back(tmp);
	}

	classifier->ClassInferBatch(tmpV);
	classResult.assign(classifier->finalResultBatch.begin(), classifier->finalResultBatch.end());

	for (int i = 0; i < tmpBucketRoi.size(); i++) {
		for (int j = 0; j < objects.size(); j++) {
			if (!(cv::Rect(tmpBucketRoi.at(i).rect) & cv::Rect(objects.at(j).rect)).empty()) {
				tmpBucketRoi.at(i).order[tmpBucketRoi.at(i).load_num] = {classResult.at(j).id, (int)(objects.at(j).rect.tl().y)};
				tmpBucketRoi.at(i).load_num++;
			}
		}
		sort(tmpBucketRoi.at(i).order.begin(), tmpBucketRoi.at(i).order.end(), [](const vector<int> &p0, const vector<int> &p1)
			{
				return abs(p0.at(1)) > abs(p1.at(1));
			});

		tmpBucketRoi.at(i).color = classifier->classNamesBatch[tmpBucketRoi.at(i).order.at(0).at(0)] + 
								   classifier->classNamesBatch[tmpBucketRoi.at(i).order.at(1).at(0)] + 
								   classifier->classNamesBatch[tmpBucketRoi.at(i).order.at(2).at(0)];
		for (int k = 0; k < 5; k++) {
			if (tmpBucketRoi.at(i).track_id == trackId2Order.at(k)) {
				bucketState.at(k) = {tmpBucketRoi.at(i).order.at(0).at(0), tmpBucketRoi.at(i).order.at(1).at(0), tmpBucketRoi.at(i).order.at(2).at(0)};
			}
		}
	}
}

cv::Mat Decision::DrawBucketState(const cv::Mat& frame, std::vector<BucketState>& bucketState) 
{
	cv::Mat result = frame.clone();
	if (!bucketState.size()) {
		return result;
	}
	cv::putText(result, trackIdString, cv::Point(0, 30),
					0, 0.6, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
	for (unsigned long i = 0; i < bucketState.size(); i++) {
		bool vertical = bucketState.at(i).rect.width / bucketState.at(i).rect.height > 1.6;
		if (bucketState.at(i).rect.width * bucketState.at(i).rect.height > 20 && !vertical) {
			cv::putText(result, cv::format("%d", bucketState[i].track_id) + bucketState.at(i).color, cv::Point(bucketState.at(i).rect.tl().x, bucketState.at(i).rect.tl().y - 5),
					0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
			cv::rectangle(result, bucketState.at(i).rect, cv::Scalar(255, 0, 0), 2);
		}
	}
	return result;
}


void Decision::GetBallDirection(const cv::Mat& srcImage, const cv::Mat& depthImage, const std::vector<STrack>& ball_inline)
{
	if (!ball_inline.size()) {
		return;
	}
	cv::Mat dep = depthImage.clone();
	tmpBallState.clear();
	TargetBallFlag = false;

	for (auto i : ball_inline) {
		BallState tmp;
		tmp.rect = cv::Rect(i.tlwh.at(0), i.tlwh.at(1), i.tlwh.at(2), i.tlwh.at(3));
		tmp.track_id = i.track_id;
		tmpBallState.push_back(tmp);
		tmpBallState.back().depth = GetDepthOfBall(tmpBallState.back().rect, dep) < 1e-4 ? 999 : GetDepthOfBall(tmpBallState.back().rect, dep);
		if (tmp.track_id = TargetBall.track_id) {
			TargetBall = tmpBallState.back();
		}
	}

	if (TargetBall.track_id != -1 && TargetBall.depth < 10) {
		TargetBallFlag = true;
		return;
	}

	std::vector<cv::Mat> tmpV;
	std::vector<Output> classResult;
	cv::Mat dstImage;
	srcImage.copyTo(dstImage);
	for (auto i : tmpBallState) {
		cv::Mat tmp;
		dstImage(i.rect).copyTo(tmp);
		tmpV.push_back(tmp);
	}

	classifier->ClassInferBatch(tmpV);
	classResult.assign(classifier->finalResultBatch.begin(), classifier->finalResultBatch.end());

	int num_site = 0;
	for (int i = 0; i < tmpBallState.size(); i++) {
		tmpBallState.at(i).color = classResult.at(i).id;
		if (tmpBallState.at(i).color != BallStateColor::PURPLE) {
			num_site++;
		}
	}

	if (num_site == 0) {
		return;
	}

	sort(tmpBallState.begin(), tmpBallState.end(), [](const BallState &p0, const BallState &p1)
		{
			return p0.color < p1.color;
		});
	for (int i = tmpBallState.size()-1; i > num_site-1; i--) {
		tmpBallState.pop_back();
	}
	sort(tmpBallState.begin(), tmpBallState.end(), [](const BallState &p0, const BallState &p1)
		{
			return p0.depth < p1.depth;
		});
	TargetBall = tmpBallState.front();
	TargetBallFlag = TargetBall.depth < 10;
}

cv::Mat Decision::Draw(const cv::Mat& frame, const std::vector<STrack>& ball_inline)
{
	cv::Mat result = frame.clone();
	if (!ball_inline.size()) {
		return result;
	}
	cv::putText(result, trackIdString, cv::Point(0, 30),
					0, 0.6, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
	for (unsigned long i = 0; i < ball_inline.size(); i++) {
		bool vertical = ball_inline.at(i).tlwh[2] / ball_inline.at(i).tlwh[3] > 1.6;
		if (ball_inline.at(i).tlwh[2] * ball_inline.at(i).tlwh[3] > 20 && !vertical) {
			cv::putText(result, cv::format("%d", ball_inline[i].track_id), cv::Point(ball_inline.at(i).tlwh[0], ball_inline.at(i).tlwh[1] - 5),
					0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
			cv::rectangle(result, cv::Rect(ball_inline.at(i).tlwh[0], ball_inline.at(i).tlwh[1], ball_inline.at(i).tlwh[2], ball_inline.at(i).tlwh[3]), cv::Scalar(255, 0, 0), 2);
		}
	}
	return result;
}