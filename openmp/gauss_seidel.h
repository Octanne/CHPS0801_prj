#ifndef GAUSS_SEIDEL_H
#define GAUSS_SEIDEL_H

#include <opencv2/opencv.hpp>

void gauss_seidel_sequential(cv::Mat& A, cv::Mat& A_new, int iterations);
void gauss_seidel_parallel_fronts_cpu(cv::Mat& A, cv::Mat& A_new, int iterations);
void gauss_seidel_parallel_fronts_gpu(cv::Mat& A, cv::Mat& A_new, int iterations);
void gauss_seidel_rb_parallel_cpu(cv::Mat& A, cv::Mat& A_new, int iterations);

#endif