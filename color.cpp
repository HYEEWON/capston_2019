#include "opencv2/opencv.hpp"
#include <iostream>
#include <math.h>
#include <time.h>
#include "color.h"

using namespace std;
using namespace cv;

int range_count = 0;
int hue = 0;
int low_hue1 = 0, low_hue2 = 0;
int high_hue1 = 0, high_hue2 = 0;
Point2d * xy;
bool state = false;

void callback(int range)
{
	int low_hue = hue - range;
	int high_hue = hue + range;

	if (low_hue < 10) {
		range_count = 2;

		high_hue1 = 180;
		low_hue1 = low_hue + 180;
		high_hue2 = high_hue;
		low_hue2 = 0;
	}
	else if (high_hue > 170) {
		range_count = 2;

		high_hue1 = low_hue;
		low_hue1 = 180;
		high_hue2 = high_hue - 180;
		low_hue2 = 0;
	}
	else {
		range_count = 1;

		low_hue1 = low_hue;
		high_hue1 = high_hue;
	}
}

void getObjectHistogram(Mat &frame, Rect object_region, Mat &globalHistogram, Mat &objectHistogram)
{
	const int channels[] = { 0, 1 };
	const int histSize[] = { 64, 64 };
	float range[] = { 0, 256 };
	const float *ranges[] = { range, range };

	// Histogram in object region
	Mat objectROI = frame(object_region);
	calcHist(&objectROI, 1, channels, noArray(), objectHistogram, 2, histSize, ranges, true, false);


	// A priori color distribution with cumulative histogram
	calcHist(&frame, 1, channels, noArray(), globalHistogram, 2, histSize, ranges, true, true);


	// Boosting: Divide conditional probabilities in object area by a priori probabilities of colors
	for (int y = 0; y < objectHistogram.rows; y++) {
		for (int x = 0; x < objectHistogram.cols; x++) {
			objectHistogram.at<float>(y, x) /= globalHistogram.at<float>(y, x);
		}
	}
	normalize(objectHistogram, objectHistogram, 0, 255, NORM_MINMAX);
}

void backProjection(const Mat &frame, const Mat &objectHistogram, Mat &bp) {
	const int channels[] = { 0, 1 };
	float range[] = { 0, 256 };
	const float *ranges[] = { range, range };
	calcBackProject(&frame, 2, channels, objectHistogram, bp, ranges);
}

int colormain(VideoCapture& cap)
{
	Scalar red(0, 0, 255);
	Mat img_result;
	int numOfLables = 0;
	clock_t begin, end;
	//Mat rgb_color = Mat(1, 1, CV_8UC3, Scalar(176, 103, 17)); //color bgr
	Mat rgb_color = Mat(1, 1, CV_8UC3, Scalar(159, 38, 106)); //color bgr
	Mat hsv_color;
	Mat img_gray, img_canny;
	cvtColor(rgb_color, hsv_color, COLOR_BGR2HSV);

	hue = (int)hsv_color.at<Vec3b>(0, 0)[0];

	int rangeH = 10;
	int LowS = 130;
	int LowV = 50;

	callback(rangeH);

	cvNamedWindow("Result Video", 1);
	cvCreateTrackbar("rangeH(0~90)", "Result Video", &rangeH, 179, callback); //Hue (0 - 179)
	cvCreateTrackbar("LowS(0~255)", "Result Video", &LowS, 255); //Saturation (0 - 255)
	cvCreateTrackbar("LowV(0~255)", "Result Video", &LowV, 255); //Value (0 - 255)

	//VideoCapture cap;
	Mat img_frame, img_hsv;

	cap.open(0);

	Rect trackingWindow(0, 0, 30, 30);
	int frame_index = 0;

	Mat globalHistogram;

	Mat * objectHistogram;
	Rect *prev_rect;

	static Point pt[5];

	begin = clock();
	for (;;)
	{
		cap.read(img_frame);

		if (img_frame.empty()) {
			cerr << "ERROR! blank frame grabbed\n";
			break;
		}

		frame_index++;

		cvtColor(img_frame, img_hsv, COLOR_BGR2HSV);
		

		Mat img_mask1, img_mask2;
		inRange(img_hsv, Scalar(low_hue1, LowS, LowV), Scalar(high_hue1, 255, 255), img_mask1);
		if (range_count == 2) {
			inRange(img_hsv, Scalar(low_hue2, LowS, LowV), Scalar(high_hue2, 255, 255), img_mask2);
			img_mask1 |= img_mask2;
		}

		erode(img_mask1, img_mask1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(img_mask1, img_mask1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		dilate(img_mask1, img_mask1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(img_mask1, img_mask1, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

		if (frame_index < 100) {
			Mat img_labels, stats, centroids;
			numOfLables = connectedComponentsWithStats(img_mask1, img_labels, stats, centroids, 8, CV_32S);
			cout << "num: " << numOfLables << endl;
			cout << "fnum: " << frame_index << endl;
			if (numOfLables == 1)
			{
				frame_index = 0;
			}

			objectHistogram = new Mat[numOfLables];
			Rect *object_region = new Rect[numOfLables];
			prev_rect = new Rect[numOfLables];
			for (int i = 1; i < numOfLables; i++)
			{
				int left = stats.at<int>(i, CC_STAT_LEFT);
				int top = stats.at<int>(i, CC_STAT_TOP);
				int width = stats.at<int>(i, CC_STAT_WIDTH);
				int height = stats.at<int>(i, CC_STAT_HEIGHT);
				object_region[i - 1] = Rect(left, top, width, height);
				getObjectHistogram(img_hsv, object_region[i - 1], globalHistogram, objectHistogram[i - 1]);
				trackingWindow = object_region[i - 1];

				rectangle(img_frame, Point(left, top), Point(left + width, top + height),
					Scalar(0, 0, 255), 1);

				prev_rect[i - 1] = object_region[i - 1];
			}
		}
		else {
			RotatedRect *ar_Rect = new RotatedRect[numOfLables];
			xy = new Point2d[numOfLables];
			for (int i = 1; i < numOfLables; i++)
			{
				Mat bp;
				char number[3];
				backProjection(img_hsv, objectHistogram[i - 1], bp);
				bitwise_and(bp, img_mask1, bp);

				RotatedRect rect = CamShift(bp, prev_rect[i - 1], cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 1));
				ar_Rect[i] = rect;
				Rect t_rect = rect.boundingRect();
				sprintf(number, "%d", i);
				xy[i] = rect.center;

				printf("id: %d, x = %lf, y =%lf\n", (double)(i), xy[i].x, xy[i].y);
				
				Point curr_pt(xy[i].x, xy[i].y);
				//cout << "circle: " << curr_pt << endl;
				pt[i] = curr_pt;
				circle(img_frame, pt[i], 5, red, 3);
				end = clock();
				cout << "s: " << state << endl;
				if ((end - begin)/CLOCKS_PER_SEC > 13) state = true;
				cout << "ss: " << state << endl;
			}
		}

		cvtColor(img_mask1, img_mask1, COLOR_GRAY2BGR);
		hconcat(img_frame, img_mask1, img_result);

		imshow("Result Video", img_result);
		if (waitKey(5) >= 0) break;
	}
	return 0;
}