#ifndef GAUSSIANNOISE_H
#define GAUSSIANNOISE_H

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

inline uchar Clamp(int n);
bool AddGaussianNoise(const Mat mSrc, Mat &mDst,double Mean=0.0, double StdDev=10.0);
bool AddGaussianNoise_Opencv(const Mat mSrc, Mat &mDst,double Mean=0.0, double StdDev=10.0);

#endif
