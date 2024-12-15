#include "classification.h"

Classification::Classification(const string &path, bool isCuda) {
    if (readModel(net, path, isCuda)) {
		cout << "read net ok!" << endl;
	} else {
		cout << "load net failed !!" << endl;
	}
}

bool Classification::readModel(cv::dnn::Net &net, const std::string &netPath, bool isCuda) {
	try {
		net = readNet(netPath);
	} catch (const std::exception &) {
		return false;
	}
	
	if (isCuda) {
		cout << "cuda is using !!!!!!!" << endl;
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
	} else {
		cout << "cpu is using !!" << endl;
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}

	return true;
}

void Classification::ClassInfer(cv::Mat srcImage) {
	release();
	BallFlag = ClassInferTarget(srcImage);
}

bool Classification::ClassInferTarget(cv::Mat srcImage) {
	cv::TickMeter tk;
	Mat netInputImg, tmp, blob;
	srcImage.copyTo(tmp);
	
	cv::cvtColor(tmp, tmp, cv::COLOR_BGR2RGB);
	resize(tmp, tmp, Size(256, 256), 0, 0, INTER_AREA);

	int cropSize = 224;
    int startY = (tmp.rows - cropSize) / 2;
    int startX = (tmp.cols - cropSize) / 2;
    cv::Rect cropRegion(startX, startY, cropSize, cropSize);
    tmp = tmp(cropRegion);
    tmp.convertTo(netInputImg, CV_32FC3, 1 / 255.f);
    
    cv::Scalar mean(0.485, 0.456, 0.406);
    cv::Scalar stdDev(0.229, 0.224, 0.225);
    netInputImg = (netInputImg - mean) / stdDev;
	

	blobFromImage(netInputImg, blob, 1, cv::Size(224, 224), cv::Scalar(0, 0, 0), false, false);
	// blob = blobFromImages(netInputImg, 1.0, Size(224, 224), Scalar(0, 0, 0), false, false);

	net.setInput(blob);
	cv::Mat netOutputImg;
	

	try {
		tk.start();	
		netOutputImg = net.forward();
		tk.stop();
		// cout << "------common time:" << tk.getTimeMilli() << endl;
		// std::cout << "forward ok" << std::endl;
	} catch (const cv::Exception& e) {
		std::cout << "forward failed" << e.what() << std::endl;
		return false;
	}

	cv::Point classIdPoint;
	double confidence;
	cv::minMaxLoc(netOutputImg.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
	int predictedClass = classIdPoint.x;
	std::string className = classNames[predictedClass];
	// std::cout << "Predicted Class: " << className << " with confidence: " << confidence << std::endl;
	finalResult.id = predictedClass;
	finalResult.confidence = confidence;

	return true;
}

void Classification::ClassInferBatch(const std::vector<cv::Mat>& srcImages) {
	release();
	BallFlag = ClassInferTargetBatch(srcImages);
}

bool Classification::ClassInferTargetBatch(const std::vector<cv::Mat>& images) {
	cv::TickMeter tk;
	finalResultBatch.clear();
	vector<Mat> netInputImgs, tmpImages;
	cv::Mat blob;
	for (auto i : images) {
		cv::Mat tmp;
		i.copyTo(tmp);
		tmpImages.push_back(tmp);
	}

	for (auto tmp : tmpImages) {
		cv::Mat netInputImg;
		cv::cvtColor(tmp, tmp, cv::COLOR_BGR2RGB);
		resize(tmp, tmp, Size(256, 256), 0, 0, INTER_AREA);

		int cropSize = 224;
		int startY = (tmp.rows - cropSize) / 2;
		int startX = (tmp.cols - cropSize) / 2;
		cv::Rect cropRegion(startX, startY, cropSize, cropSize);
		tmp = tmp(cropRegion);
		tmp.convertTo(netInputImg, CV_32FC3, 1 / 255.f);
		
		cv::Scalar mean(0.485, 0.456, 0.406);
		cv::Scalar stdDev(0.229, 0.224, 0.225);
		netInputImg = (netInputImg - mean) / stdDev;
		netInputImgs.push_back(netInputImg);
	}
	
	// blobFromImage(netInputImg, blob, 1, cv::Size(224, 224), cv::Scalar(0, 0, 0), false, false);
	blob = blobFromImages(netInputImgs, 1.0, Size(224, 224), Scalar(0, 0, 0), false, false);
	net.setInput(blob);
	cv::Mat netOutputImg;

	try {
		tk.start();	
		netOutputImg = net.forward();
		tk.stop();
	} catch (const cv::Exception& e) {
		std::cout << "forward failed" << e.what() << std::endl;
		return false;
	}

	for (int n = 0; n < netOutputImg.rows; n++) {
		cv::Point classIdPoint;
		double confidence;
		Mat probMat = netOutputImg(Rect(0, n, 2, 1)).clone();
		// std::cout << netOutputImg.at<float>(n, 0) << "\t" << netOutputImg.at<float>(n, 1) << "\t" << netOutputImg.at<float>(n, 2) << "\n";
		Mat result = probMat.reshape(1, 1);
		minMaxLoc(result, NULL, &confidence, NULL, &classIdPoint);
		Output tmp;
		tmp.id = classIdPoint.x;
		tmp.confidence = confidence;
		finalResultBatch.push_back(tmp);
	}

	// for (int i = 0; i < images.size(); i++) {
	// 	resize(tmpImages.at(i), tmpImages.at(i), Size(256, 256), 0, 0, INTER_AREA);
	// 	cv::putText(tmpImages.at(i), cv::format("%d", finalResultBatch.at(i).id), cv::Point(0, 20),
	// 				0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
	// 	imshow(cv::format("%d", i), tmpImages.at(i));
	// }
	// waitKey(200);
	
	return true;
}

void Classification::viewer(void) {
	dstImage.copyTo(saveImg);
	imshow("dstImage", dstImage);
	waitKey(1);
}

void Classification::release() {
	
    BallFlag = 0;
}

Classification::~Classification() {    

}
