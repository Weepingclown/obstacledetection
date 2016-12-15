
#include "calibration_stereo.h"

Calibration::Calibration(int board_width, int board_height, float square_width, float square_height)
	: boardSize(board_width, board_height), squareSize(square_width, square_height)
{
}

void Calibration::InitCalibration()
{
	imagePoints1.clear();
	imagePoints2.clear();
}

int Calibration::FindChessboard(cv::Mat &view1, cv::Mat &view2, bool reg)
{
	assert(view1.size() == view2.size());
	imageSize = view1.size();

	cv::Mat viewGray1, viewGray2;
	cvtColor(view1, viewGray1, CV_BGR2GRAY);
	cvtColor(view2, viewGray2, CV_BGR2GRAY);

	vector<cv::Point2f> pointbuf1, pointbuf2;

	bool found1 = findChessboardCorners(view1, boardSize, pointbuf1,
		CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
	bool found2 = findChessboardCorners(view2, boardSize, pointbuf2,
		CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);

	if (found1 && found2) {
		// improve the found corners' coordinate accuracy
		cornerSubPix(viewGray1, pointbuf1, cv::Size(1, 1), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		cornerSubPix(viewGray2, pointbuf2, cv::Size(1, 1), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

		if (reg) {
			imagePoints1.push_back(pointbuf1);
			imagePoints2.push_back(pointbuf2);
		}
		drawChessboardCorners(view1, boardSize, cv::Mat(pointbuf1), found1);
		drawChessboardCorners(view2, boardSize, cv::Mat(pointbuf2), found2);
	}

	return imagePoints1.size();
}

bool Calibration::RunCalibration()
{
	vector<vector<cv::Point3f>> objectPoints(1);

	for (int i = 0; i < boardSize.height; i++) {
		for (int j = 0; j < boardSize.width; j++) {
			objectPoints[0].push_back(cv::Point3f(j*squareSize.width, i*squareSize.height, 0.f));
		}
	}
	objectPoints.resize(imagePoints1.size(), objectPoints[0]);

	//
	// 좌우 camera의 체스판 영상의 점들로부터 camera matrix, distortion coefficients와 R, P 행렬을 계산한다
	//

	cv::Mat cameraMatrix1 = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat cameraMatrix2 = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat distCoeffs1 = cv::Mat::zeros(8, 1, CV_64F);
	cv::Mat distCoeffs2 = cv::Mat::zeros(8, 1, CV_64F);
	cv::Mat R, T, E, F;

	double rms = stereoCalibrate(objectPoints, imagePoints1, imagePoints2,
		cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
		imageSize, R, T, E, F,
		CV_CALIB_FIX_ASPECT_RATIO | CV_CALIB_ZERO_TANGENT_DIST | CV_CALIB_SAME_FOCAL_LENGTH |
		CV_CALIB_RATIONAL_MODEL | CV_CALIB_FIX_K3 | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5,
		cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));

	bool ok = checkRange(cameraMatrix1) && checkRange(distCoeffs1) && checkRange(cameraMatrix2) && checkRange(distCoeffs2);
	if (ok) {
		cv::Mat R1, R2, P1, P2;
		cv::Rect validRoi1, validRoi2;

		stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
			imageSize, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi1, &validRoi2);

		// save intrinsic parameters
		cv::FileStorage fs("intrinsics.yml", CV_STORAGE_WRITE);
		if (!fs.isOpened()) return false;

		fs << "M1" << cameraMatrix1;
		fs << "D1" << distCoeffs1;
		fs << "M2" << cameraMatrix2;
		fs << "D2" << distCoeffs2;
		fs.release();

		fs.open("extrinsics.yml", CV_STORAGE_WRITE);
		if (!fs.isOpened()) return false;

		fs << "R" << R;
		fs << "T" << T;
		fs << "R1" << R1;
		fs << "R2" << R2;
		fs << "P1" << P1;
		fs << "P2" << P2;
		fs << "Q" << Q;
		fs << "imageSize" << imageSize;
		fs.release();

		initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, CV_16SC2, map1, map2);
		initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, imageSize, CV_16SC2, map3, map4);
	}

	return ok;
}

bool Calibration::LoadCalibrationData()
{
	// reading intrinsic parameters
	cv::Mat cameraMatrix1 = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat cameraMatrix2 = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat distCoeffs1 = cv::Mat::zeros(8, 1, CV_64F);
	cv::Mat distCoeffs2 = cv::Mat::zeros(8, 1, CV_64F);

	cv::FileStorage fs("intrinsics.yml", CV_STORAGE_READ);
	if (!fs.isOpened()) return false;

	fs["M1"] >> cameraMatrix1;
	fs["D1"] >> distCoeffs1;
	fs["M2"] >> cameraMatrix2;
	fs["D2"] >> distCoeffs2;

	//read extrinsic parameters
	cv::Mat R, T;
	cv::Mat R1, R2, P1, P2;
	cv::Rect validRoi1, validRoi2;

	fs.open("extrinsics.yml", CV_STORAGE_READ);
	if (!fs.isOpened()) return false;

	fs["R"] >> R;
	fs["T"] >> T;
	fs["R1"] >> R1;
	fs["P1"] >> P1;
	fs["R2"] >> R2;
	fs["P2"] >> P2;
	fs["Q"] >> Q;
	
	cv::FileNode is = fs["imageSize"];
	imageSize.width = is[0];
	imageSize.height = is[1];

	initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, imageSize, CV_16SC2, map1, map2);
	initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, imageSize, CV_16SC2, map3, map4);

	return true;
}

cv::Mat Calibration::getQ()
{
	return this->Q;
}

void Calibration::Undistort(const cv::Mat &view1, cv::Mat &rview1, const cv::Mat &view2, cv::Mat &rview2)
{
	if (map1.data && map2.data && map3.data && map4.data) {
		remap(view1, rview1, map1, map2, cv::INTER_LINEAR);
		remap(view2, rview2, map3, map4, cv::INTER_LINEAR);
	}
}
