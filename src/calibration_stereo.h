#pragma once
#include <opencv2/opencv.hpp>

//using namespace cv;
using namespace std;

class Calibration {
public:
	Calibration(int board_width, int board_height, float square_width, float square_height);

	void InitCalibration();
	int FindChessboard(cv::Mat &view1, cv::Mat &view2, bool reg = false);
	bool RunCalibration();
	bool LoadCalibrationData();
	void Undistort(const cv::Mat &view1, cv::Mat &rview1, const cv::Mat &view2, cv::Mat &rview2);
	cv::Mat getQ();

private:
	const cv::Size boardSize;
	const cv::Size2f squareSize;
	

	vector<vector<cv::Point2f> > imagePoints1, imagePoints2;
	cv::Size imageSize;
	cv::Mat map1, map2, map3, map4, Q;
};
