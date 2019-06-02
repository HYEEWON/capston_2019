#include <iostream>
#include "opencv2/opencv.hpp"
#include "color.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[])
{
	VideoCapture cap;
	colormain(cap);
	return 0;
}
