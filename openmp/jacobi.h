#ifndef JACOBI_H
#define JACOBI_H

#include <opencv2/opencv.hpp>

void jacobi_sequential(cv::Mat& A, cv::Mat& A_new, int iterations);
void jacobi_parallel_cpu(cv::Mat& A, cv::Mat& A_new, int iterations);
void jacobi_parallel_gpu(cv::Mat& A, cv::Mat& A_new, int iterations);

#endif