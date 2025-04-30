#ifndef FILTERS_H
#define FILTERS_H

#include <opencv2/opencv.hpp>

// Applies a 3x3 box blur using CUDA
void apply_blur(const cv::Mat& input, cv::Mat& output);

#endif  // FILTERS_H
