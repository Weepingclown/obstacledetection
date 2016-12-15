#include <opencv2/calib3d.hpp>

#include "Header.h"
#include <iostream>
#include <fstream>
#include <ros/ros.h>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <opencv2/highgui/highgui.hpp>


//#include "calibration_stereo.h"
#include <time.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;


// struct Real_xyz
// {
	
// 	float x;
// 	float y;
// 	float z;
// };

enum { MODE_NOTHING = 0, MODE_CALIBRATION = 1, MODE_UNDISTORTION = 2 };
static int mode = MODE_NOTHING;
static const int nframes = 30;		// ÀÎ½ÄÇÒ Ã¼½ºÆÇ ¼ö
static bool reg_chessboard = false;
static bool lineOn = false;
// static Calibration calib(4, 7, 0.314f / 9, 0.209f / 6);

cv::Rect Ground_box;
int drawing_box_flag = 0;
cv::Vec3d printval;
cv::Point mousePos;
cv::PCA pca;
double p_to_plane_thresh;

void rotateImage(const cv::Mat &input, cv::Mat &output, double alpha, double beta, double gamma, double dx, double dy, double dz, double f)
{
	alpha = (alpha - 90.)*CV_PI / 180.;
	beta = (beta - 90.)*CV_PI / 180.;
	gamma = (gamma - 90.)*CV_PI / 180.;
	// get width and height for ease of use in matrices
	double w = (double)input.cols;
	double h = (double)input.rows;
	
	cv::Mat RX = (cv::Mat_<double>(3, 3) <<
		1, 0, 0, 
		0, cos(alpha), -sin(alpha),
		0, sin(alpha), cos(alpha));
	cv::Mat RY = (cv::Mat_<double>(3, 3) <<
		cos(beta), 0, -sin(beta), 
		0, 1, 0, 
		sin(beta), 0, cos(beta));
	cv::Mat RZ = (cv::Mat_<double>(3, 3) <<
		cos(gamma), -sin(gamma), 0, 
		sin(gamma), cos(gamma), 0, 
		0, 0, 1);
	// Composed rotation matrix with (RX, RY, RZ)
	cv::Mat R = RX * RY * RZ;
	
	cv::Mat trans = R;
	trans.convertTo(trans, CV_32FC1);
	cv::Mat xyz = input.reshape(3, input.size().area());
	vector<cv::Mat> channels(3);
	split(xyz, channels);
	cv::Mat mix(0,0,CV_32FC1);
	hconcat(channels[0], channels[1], mix);
	hconcat(mix, channels[2], mix);
	//hconcat(mix, Mat::ones(mix.rows, 1, CV_32FC1), mix);
	mix.convertTo(mix, CV_32FC1);
	//mix = mix * trans;
	output = cv::Mat(input.size(), CV_32FC3);
	
	for (int y = 0; y < input.rows; y++)
	{
		for (int x = 0; x < input.cols; x++)
		{
			int index = y * input.rows + x;
			output.at<cv::Vec3f>(cv::Point(x, y)) = cv::Vec3f(mix.at<double>(index, 0), mix.at<double>(index, 1), mix.at<double>(index, 2));
		}
	}
}

cv::Rect computeROI(cv::Size2i src_sz, cv::Ptr<cv::StereoMatcher> matcher_instance)
{
	int min_disparity = matcher_instance->getMinDisparity();
	int num_disparities = matcher_instance->getNumDisparities();
	int block_size = matcher_instance->getBlockSize();

	int bs2 = block_size / 2;
	int minD = min_disparity, maxD = min_disparity + num_disparities - 1;

	int xmin = maxD + bs2;
	int xmax = src_sz.width + minD - bs2;
	int ymin = bs2;
	int ymax = src_sz.height - bs2;

	cv::Rect r(xmin, ymin, xmax - xmin, ymax - ymin);
	return r;
}

void onMouseEvent(int event, int x, int y, int flags, void* param)
{
	cv::Mat *src = (cv::Mat *)param;
	
	switch (event) {
	case CV_EVENT_MOUSEMOVE:
		mousePos = cv::Point(x, y);
		if (drawing_box_flag == 1)
		{
			Ground_box.width = x - Ground_box.x;
			Ground_box.height = y - Ground_box.y;
		}
		break;

	case CV_EVENT_LBUTTONDOWN:
		drawing_box_flag = 1;
		Ground_box = cv::Rect(x, y, 0, 0);
		break;

	case CV_EVENT_LBUTTONUP:
		drawing_box_flag = -1;
		if (Ground_box.width < 0) {
			Ground_box.x += Ground_box.width;
			Ground_box.width *= -1;
		}
		if (Ground_box.height < 0) {
			Ground_box.y += Ground_box.height;
			Ground_box.height *= -1;
		}
		break;
	}
}

void eular2rot(double yaw, double pitch, double roll, cv::Mat& dest)
{
	double theta = yaw / 180.0*CV_PI;
	double pusai = pitch / 180.0*CV_PI;
	double phi = roll / 180.0*CV_PI;

	double datax[3][3] = { { 1.0,0.0,0.0 },
	{ 0.0,cos(theta),-sin(theta) },
	{ 0.0,sin(theta),cos(theta) } };
	double datay[3][3] = { { cos(pusai),0.0,sin(pusai) },
	{ 0.0,1.0,0.0 },
	{ -sin(pusai),0.0,cos(pusai) } };
	double dataz[3][3] = { { cos(phi),-sin(phi),0.0 },
	{ sin(phi),cos(phi),0.0 },
	{ 0.0,0.0,1.0 } };
	cv::Mat Rx(3, 3, CV_64F, datax);
	cv::Mat Ry(3, 3, CV_64F, datay);
	cv::Mat Rz(3, 3, CV_64F, dataz);
	cv::Mat rr = Rz*Rx*Ry;
	rr.copyTo(dest);
}

void lookat(cv::Point3d from, cv::Point3d to, cv::Mat& destR)
{
	double x = (to.x - from.x);
	double y = (to.y - from.y);
	double z = (to.z - from.z);

	double pitch = asin(x / sqrt(x*x + z*z)) / CV_PI*180.0;
	double yaw = asin(-y / sqrt(y*y + z*z)) / CV_PI*180.0;
	
	eular2rot(yaw, pitch, 0, destR);
}

template <class T>
static void fillOcclusion_(cv::Mat& src, T invalidvalue)
{
	int bb = 1;
	const int MAX_LENGTH = src.cols*0.5;
#pragma omp parallel for
	for (int j = bb; j<src.rows - bb; j++)
	{
		T* s = src.ptr<T>(j);
		const T st = s[0];
		const T ed = s[src.cols - 1];
		s[0] = 255;
		s[src.cols - 1] = 255;
		for (int i = 0; i<src.cols; i++)
		{
			if (s[i] <= invalidvalue)
			{
				int t = i;
				do
				{
					t++;
					if (t>src.cols - 1)break;
				} while (s[t] <= invalidvalue);

				const T dd = min(s[i - 1], s[t]);
				if (t - i>MAX_LENGTH)
				{
					for (int n = 0; n<src.cols; n++)
					{
						s[n] = invalidvalue;
					}
				}
				else
				{
					for (; i<t; i++)
					{
						s[i] = dd;
					}
				}
			}
		}
	}
}

cv::Mat makeQMatrix(cv::Point2d image_center, double focal_length, double baseline)
{
	cv::Mat Q = cv::Mat::eye(4, 4, CV_64F);
	Q.at<double>(0, 3) = -image_center.x;
	Q.at<double>(1, 3) = -image_center.y;
	Q.at<double>(2, 3) = focal_length;
	Q.at<double>(3, 3) = 0.0;
	Q.at<double>(2, 2) = 0.0;
	Q.at<double>(3, 2) = 1.0 / baseline;

	return Q;
}

template <class T>
static void projectImagefromXYZ_(cv::Mat& image, cv::Mat& destimage, cv::Mat& disp, cv::Mat& destdisp, cv::Mat& xyz, cv::Mat& R, cv::Mat& t, cv::Mat& K, cv::Mat& dist, cv::Mat& mask, bool isSub)
{
	if (destimage.empty())destimage = cv::Mat::zeros(cv::Size(image.size()), image.type());
	if (destdisp.empty())destdisp = cv::Mat::zeros(cv::Size(image.size()), disp.type());

	vector<cv::Point2f> pt;
	if (dist.empty()) dist = cv::Mat::zeros(cv::Size(5, 1), CV_32F);
	cv::projectPoints(xyz, R, t, K, dist, pt);

	destimage.setTo(0);
	destdisp.setTo(0);

#pragma omp parallel for
	for (int j = 1; j<image.rows - 1; j++)
	{
		int count = j*image.cols;
		uchar* img = image.ptr<uchar>(j);
		uchar* m = mask.ptr<uchar>(j);
		for (int i = 0; i<image.cols; i++, count++)
		{
			int x = (int)(pt[count].x + 0.5);
			int y = (int)(pt[count].y + 0.5);
			
			if (m[i] == 255)continue;
			if (pt[count].x >= 1 && pt[count].x<image.cols - 1 && pt[count].y >= 1 && pt[count].y<image.rows - 1)
			{
				short v = destdisp.at<T>(y, x);
				if (v<disp.at<T>(j, i))
				{
					destimage.at<uchar>(y, 3 * x + 0) = img[3 * i + 0];
					destimage.at<uchar>(y, 3 * x + 1) = img[3 * i + 1];
					destimage.at<uchar>(y, 3 * x + 2) = img[3 * i + 2];
					destdisp.at<T>(y, x) = disp.at<T>(j, i);

				}
			}
		}
	}


}

Real_xyz DepthToWorld(int x, int y, float depthValue)
{
	
	const double fx_d = 1 / 325.0;
	const double fy_d = 1 / 594.0;
	const double cx_d = 169;
	const double cy_d = 419.3;



	Real_xyz result;
	//const double depth = RawDepthToMeters(depthValue);

	result.x = float((x - cx_d) * depthValue * fx_d);
	result.y = float((y - cy_d) * depthValue * fy_d);
	result.z = float(depthValue);
	return result;
}

int main(int argc, char **argv)
{
	int thres = 200;

	map<string, cv::Mat> aa;
	cv::Mat tt = aa.find("view1")->second;

	pair<string, cv::Mat> bb;
	
	

	


	cv::VideoCapture capture[2];
	
	capture[0].open(0);
	
	cv::Vec3d x0;
	cv::Vec3d nrm;

	
	if (!capture[0].isOpened())
	{
		cout << "Capture could not be opened succesfully" << endl;
		cv::waitKey(0);
		return 0;
	}

	double fps = 15;
	int fourcc = CV_FOURCC('X', 'V', 'I', 'D'); // codec
	bool isColor = true;

	cv::Mat frame[2];
	cv::Mat canvas;
	

	capture[0] >> canvas;
	
	//Mat frame[2];
	frame[0] = canvas(cv::Rect(0, 0, canvas.cols / 2, canvas.rows));
	frame[1] = canvas(cv::Rect(canvas.cols / 2, 0, canvas.cols / 2, canvas.rows));
	
	//frame[0] = depth_img.clone();

	bool recordOn = false;

	cv::VideoWriter *vDisparity = new cv::VideoWriter;
	vDisparity->open("disparity.avi", fourcc, fps, frame[0].size(), false);

	cv::VideoWriter *vFilled = new cv::VideoWriter;
	vFilled->open("filled.avi", fourcc, fps, frame[0].size(), true);

	cv::VideoWriter *vDepth = new cv::VideoWriter;
	vDepth->open("depth.avi", fourcc, fps, frame[0].size(), false);

	cv::VideoWriter *vView = new cv::VideoWriter;
	vView->open("view.avi", fourcc, fps, frame[0].size(), true);

	cv::VideoWriter *vObs = new cv::VideoWriter;
	vObs->open("obos.avi", fourcc, fps, frame[0].size(), true);

	cv::VideoWriter *vXyz = new cv::VideoWriter;
	vXyz->open("xyz.avi", fourcc, fps, frame[0].size(), true);

	int max_disp = 16 * 5;
	int sgbmWinSize = 3;
	//int numberOfDisparities = ((frame[0].rows / 8) + 15) & -16;;
	int numberOfDisparities = 16 * 6;
	int cn = frame[0].channels();

	
	//cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
	cv::Mat conf_map = cv::Mat(frame[0].size(), CV_8U);
	cv::Mat filtered;
	cv::Rect ROI;
	double lambda = 8000.0;
	double sigma = 1.5;
	double vis_multi = 1.0;

	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16 * 5, 9);
	// wls_filter = cv::ximgproc::createDisparityWLSFilter(sgbm);
	//cv::Ptr<cv::StereoMatcher> sm = cv::ximgproc::createRightMatcher(sgbm);


	sgbm->setPreFilterCap(63);
	sgbm->setBlockSize(sgbmWinSize);
	sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
	
	sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);

	cv::Mat img16S = cv::Mat(frame[0].size(), CV_16S);
	cv::Mat img16Sr = cv::Mat(frame[0].size(), CV_16S);
	cv::Mat img8U = cv::Mat(frame[0].size(), CV_8UC1);

	cv::Mat Q, depth(frame[0].size(), CV_32FC3);

	char c;
	cv::namedWindow("filled", 0);
	setMouseCallback("filled", onMouseEvent, &img8U);

	const double focal_length = 598.57;
	const double baseline = 14.0;

	cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
	K.at<double>(0, 0) = focal_length;
	K.at<double>(1, 1) = focal_length;
	K.at<double>(0, 2) = (frame[0].cols - 1.0) / 2.0;
	K.at<double>(1, 2) = (frame[0].rows - 1.0) / 2.0;

	cv::Mat dist = cv::Mat::zeros(5, 1, CV_64F);
	cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
	cv::Mat t = cv::Mat::zeros(3, 1, CV_64F);

	cv::Mat destimage, destdisp, dispshow;
	cv::Mat mask = cv::Mat::zeros(frame[0].size(), CV_8U);

	cv::Point3d viewpoint(0.0, 0.0, baseline * 10);
	cv::Point3d lookatpoint(0.0, 0.0, -baseline*10.0);
	const double step = baseline;
	int key = 0;
	bool isSub = false;


	cv::Mat G_point = (cv::Mat_<float>(1, 3) << 0, 0, 0);
	cv::Mat G_point_Rot = (cv::Mat_<float>(1, 3) << 0, 0, 0);
	cv::Mat Rot_fit;

	Q = makeQMatrix(cv::Point2d((frame[0].cols - 1.0) / 2.0, (frame[0].rows - 1.0) / 2.0), focal_length, baseline * 16);


	double alpha = 0, beta, gamma;
	cv::Mat sample;


	while (1)
	{
		

		capture[0] >> canvas;

		frame[0] = canvas(cv::Rect(0, 0, canvas.cols / 2, canvas.rows));
		frame[1] = canvas(cv::Rect(canvas.cols / 2, 0, canvas.cols / 2, canvas.rows));
		mask = cv::Mat::zeros(frame[0].size(), CV_8U);
		if (mode == MODE_CALIBRATION)
		{
			

			// int iframes = calib.FindChessboard(frame[0], frame[1], reg_chessboard);
			// if (iframes >= nframes) {
			// 	if (calib.RunCalibration()) mode = MODE_UNDISTORTION;
			// 	else mode = MODE_NOTHING;
			// }
			// reg_chessboard = false;

			// putText(canvas, cv::format("Recognized chessboard = %d/%d", iframes, nframes), cv::Point(10, 25), 1, 1, cv::Scalar(0, 0, 255));

		}
		else if (mode == MODE_UNDISTORTION)
		{
			cv::Mat temp1 = frame[0].clone();
			cv::Mat temp2 = frame[1].clone();
			// calib.Undistort(temp1, frame[0], temp2, frame[1]);

			int64  st = cv::getTickCount();
			//
			sgbm->compute(frame[0], frame[1], img16S);
			//sm->compute(frame[1], frame[0], img16Sr);

			
			
			//ximgproc::getDisparityVis(img16S, img8U, vis_multi);
			img16S.convertTo(img8U, CV_8UC1, 255 / (numberOfDisparities*16.));
			imshow("disparity", img8U);

			
			// wls_filter->setLambda(lambda);
			// wls_filter->setSigmaColor(sigma);
			
			// wls_filter->filter(img16S, frame[0], filtered, img16Sr);
			
			// ROI = wls_filter->getROI();
			// cv::Mat ROIframe = frame[0](ROI);
			// imshow("ROI", frame[0](ROI));

			// filtered = filtered(ROI);
			filtered = frame[0].clone();

			cv::Mat filtered_vis;
			//ximgproc::getDisparityVis(filtered, filtered_vis, vis_multi);
			filtered.convertTo(filtered_vis, CV_8U, 255 / (numberOfDisparities*16.));
			
			imshow("filtered", filtered_vis);
			
			putText(filtered_vis, cv::format("Calculation Time : %d ms", (int)((cv::getTickCount() - st) * 1000 / cv::getTickFrequency())), cv::Point(10, 25), 1, 1, cv::Scalar::all(255));

			cv::Mat temp3;
			////cvtColor(img8U, temp3, CV_GRAY2BGR);
			temp3 = filtered_vis.clone();
			cvtColor(temp3, temp3, CV_GRAY2BGR);

			

			

			if (drawing_box_flag == 1 && Ground_box.width)
			{
				rectangle(temp3, Ground_box.tl(), Ground_box.br(), cv::Scalar(0, 0, 255), 1);
			}

			if (drawing_box_flag == -1 && Ground_box.width > 0 && Ground_box.height > 0)
			{
				sample = filtered(cv::Rect(Ground_box.tl(), Ground_box.br()));

				Real_xyz World_XYZ;

				cv::Mat_<double> f(0, 3);

				for (int y = 0; y < sample.rows; y++)
				{
					for (int x = 0; x < sample.cols; x++)
					{
						float inputdepth = (float)sample.at<short>(cv::Point(x, y));
						if (inputdepth > 100)
						{
							//World_XYZ = DepthToWorld(x, y, inputdepth);
							World_XYZ.x = x;
							World_XYZ.y = y;
							World_XYZ.z = inputdepth;
							cv::Mat temp = (cv::Mat_<double>(1, 3) << World_XYZ.x, World_XYZ.y, World_XYZ.z);
							f.push_back(temp);
						}
						
					}
				}

				if (f.rows > 100)
				{
					cv::Mat_<double> mean;
					pca(f, mean, CV_PCA_DATA_AS_ROW);
					p_to_plane_thresh = pca.eigenvalues.at<double>(2);
					thres = p_to_plane_thresh - 200;
					cv::Vec3d basis_x = pca.eigenvectors.row(0);
					cv::Vec3d basis_y = pca.eigenvectors.row(1);
					nrm = pca.eigenvectors.row(2);
					nrm = nrm / norm(nrm);
					x0 = pca.mean;

					cout << x0 << endl;
					cout << nrm << endl;

					drawing_box_flag = 0;
					alpha = 0;

					int state;
					//FILE * file;

					//fopen(&file, "Ground_pram.txt", "wt");

					FILE *file = fopen("Ground_pram.txt","wt");


					if (file == NULL)
					{
						cout << "file open error!\n" << endl;
					}
					fprintf(file, "%3.5f %3.5f %3.5f %3.5f %3.5f %3.5f %3.5f", x0[0], x0[1], x0[2], nrm[0], nrm[1], nrm[2], p_to_plane_thresh);
					state = fclose(file);
					if (state != 0)
					{
						cout << "file close error!\n" << endl;
					}

					Rot_fit = cv::Mat();
				}
				else
				{
					cout << "too few ground data" << endl;
				}
			}

			if (x0[0] != 0 && nrm[0] != 0 && Rot_fit.rows == 0)
			{
				cv::Mat Grond_plane = (cv::Mat_<float>(1, 3) << 0, 1, 0);

				cv::Mat Grond_nomal_sensing = (cv::Mat_<float>(1, 3) << nrm[0], nrm[1], nrm[2]);
				cv::Mat RotationAxis = (cv::Mat_<float>(1, 3) << 0, 0, 0);
				RotationAxis = Grond_nomal_sensing.cross(Grond_plane);
				RotationAxis = RotationAxis / norm(RotationAxis);
				float RotationAngle_Radian = acos(Grond_plane.dot(Grond_nomal_sensing) / (norm(Grond_nomal_sensing)*norm(Grond_plane)));
				cv::Mat Identity_mat = Identity_mat.eye(3, 3, CV_32F);
				cv::Mat p_p = RotationAxis.t() * RotationAxis;
				float a1 = RotationAxis.at<float>(0, 0);
				float a2 = RotationAxis.at<float>(0, 1);
				float a3 = RotationAxis.at<float>(0, 2);
				cv::Mat P_P = (cv::Mat_<float>(3, 3) << 0, a3, -a2, a3, 0, a1, a2, -a1, 0);
				//Rot_G_fit = cos(RotationAngle_Radian)*Identity_mat + (1 - cos(RotationAngle_Radian))*p_p - sin(RotationAngle_Radian)*P_P;
				Rot_fit = cos(RotationAngle_Radian)*Identity_mat + (1 - cos(RotationAngle_Radian))*p_p - sin(RotationAngle_Radian)*P_P;


			}
			
			cv::Mat Mat_Free = cv::Mat::zeros(500, 350, CV_8UC1);
			cv::Mat Mat_Obstacle = cv::Mat::zeros(500, 350, CV_8UC1);

			double maxDepth;
			double minDepth;
			cv::minMaxLoc(filtered, &minDepth, &maxDepth);

			if (x0[0] != 0 && nrm[0] != 0)
			{
				

				int sampling_gap = 3;
				int occupy_circle_size = 5;
				int free_circle_size = 3;

				int index;
				Real_xyz Ground_xyz;
				for (int y = 0; y<filtered.rows; y = y + sampling_gap)
				{
					int temp_gap = sampling_gap;
					
					for (int x = 0; x < filtered.cols; x = x + temp_gap)
					{

						/*unsigned short realDepth = filtered.at<ushort>(Point(x, y));
						float val0 = (float)realDepth;*/
						float val0 = (float)filtered.at<short>(cv::Point(x, y));

						if (val0 > 600 && val0 < 5000)
						{
							

							Ground_xyz.x = x;
							Ground_xyz.y = y;
							Ground_xyz.z = val0;

							cv::Vec3d w = cv::Vec3d(Ground_xyz.x, Ground_xyz.y, Ground_xyz.z) - x0;
							double error = fabs(nrm.dot(w));

							cv::Point robot = cv::Point(250 / 2, 500 / 2);

							Ground_xyz = DepthToWorld(x, y, val0);

							int grid_Gx1 = round((float)x * 350.0 / filtered.cols);
							int grid_Gx3 = (val0 / 500) * 350 - 500;
							
							
							
							if (mousePos == cv::Point(x, y))
								cout << val0 << endl;
							//if (x > 80) 
							{
								//p_to_plane_thresh
								if (error > thres)
								{
									//cv::circle(ROIframe, cv::Point(x, y), 1, cv::Scalar(255), -1);
									cv::circle(frame[1], cv::Point(x, y), 1, cv::Scalar(255), -1);
								}

								if (grid_Gx1 < Mat_Free.cols && grid_Gx1 >= 0 && grid_Gx3 >= 0 && grid_Gx3 < Mat_Free.rows)
								{
									if (error <= thres)
									{
										
										//cv::line(Mat_Free, cv::Point(grid_Gx1, grid_Gx3), robot, cv::Scalar(255), 4);
										cv::circle(Mat_Free, cv::Point(grid_Gx1, grid_Gx3), occupy_circle_size, cv::Scalar(255), -occupy_circle_size);
									}
									else
									{
										cv::circle(Mat_Obstacle, cv::Point(grid_Gx1, grid_Gx3), occupy_circle_size, cv::Scalar(255), -occupy_circle_size);
									}
								}
							}

						}
					}
				}
				imshow("ROIframe", frame[0]);
				imshow("free", Mat_Free);
				imshow("obstacle", Mat_Obstacle);
				if (recordOn) *vObs << frame[0];
			}

			imshow("filled", temp3);
			if(recordOn)*vFilled << temp3;

			//cout << mousePos << "-->" << xyzt.at<Vec3f>(mousePos) << endl;

		}


		if (lineOn)
			for (int j = 0; j < canvas.rows; j += 16)
				line(canvas, cv::Point(0, j), cv::Point(canvas.cols, j), cv::Scalar(0, 255, 0), 1, 8);


		imshow("canvas", canvas);

		c = cvWaitKey(30);

		if (c == 'f')
		{
			isSub = isSub ? false : true;
		}
		if (c == 'w')
		{
			viewpoint.y += step;
		}
		if (c == 's')
		{
			viewpoint.y -= step;
		}
		if (c == 'a')
		{
			viewpoint.x += step;
		}
		if (c == 'd')
		{
			viewpoint.x -= step;
		}
		if (c == 'q')
		{
			viewpoint.z += step;
		}
		if (c == 'e')
		{
			viewpoint.z -= step;
		}

		if (c == 'h')
		{
			lookatpoint.y += step;
		}
		if (c == 'y')
		{
			lookatpoint.y -= step;
		}
		if (c == 'j')
		{
			lookatpoint.x += step;
		}
		if (c == 'g')
		{
			lookatpoint.x -= step;
		}
		if (c == 't')
		{
			lookatpoint.z += step;
		}
		if (c == 'u')
		{
			lookatpoint.z -= step;
		}
		if (c == 'o')
		{
			thres--;
			cout << "thres : " << thres << endl;
		}
		if (c == 'p')
		{
			thres++;
			cout << "thres : " << thres << endl;
		}

		if (c == 27) // ESC
		{
			delete vDepth;
			delete vDisparity;
			delete vFilled;
			delete vObs;
			delete vView;
			break;
		}
		else if (c == 99) // c
		{
			reg_chessboard = true;
			mode = MODE_CALIBRATION;
			cv::destroyWindow("depth_map");
		}

		else if (c == 108) // l
		{
			//calib.LoadCalibrationData();
			//Q = calib.getQ();
			mode = MODE_UNDISTORTION;
			cv::resizeWindow("filled", canvas.cols, canvas.rows * 2);


			ifstream MyFile;

			char file_name[64];
			sprintf(file_name, "Ground_pram.txt");
			MyFile.open(file_name, ios::in);

			if (MyFile.is_open())
			{	
				MyFile >> x0[0] >> x0[1] >> x0[2] >> nrm[0] >> nrm[1] >> nrm[2] >> p_to_plane_thresh;
				thres = p_to_plane_thresh - 200;
				MyFile.close();
			}

			
		}
		else if (c == 109) // o
		{
			lineOn = !lineOn;
		}
		else if (c == 48) // 0
		{
			mode = MODE_NOTHING;
			cv::destroyWindow("depth_map");
		}

	}


	return 0;
}
