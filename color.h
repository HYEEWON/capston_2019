#ifndef COLOR_H
#define COLOR_H

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "config.h"

#define PORT     8888 
#define MAXLINE 1024 
using namespace std;
using namespace cv;

extern cv::Point2d * xy;

void callback(int range);
void getObjectHistogram(cv::Mat &frame, cv::Rect object_region, cv::Mat &globalHistogram, cv::Mat &objectHistogram);
void backProjection(const cv::Mat &frame, const cv::Mat &objectHistogram, cv::Mat &bp);
int colormain(VideoCapture& cap);

#endif
