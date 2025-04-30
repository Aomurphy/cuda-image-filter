#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "filters.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: ./image_filter <image_path>" << std::endl;
        return -1;
    }

    // Load input image
    cv::Mat input = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (input.empty()) {
        std::cerr << "Error: Cannot load image " << argv[1] << std::endl;
        return -1;
    }

    std::cout << "Image size: " << input.cols << " x " << input.rows << std::endl;

    // --- CPU Baseline (OpenCV blur) ---
    cv::Mat cpu_output;
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cv::blur(input, cpu_output, cv::Size(3, 3));  // 3x3 box blur
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_duration = cpu_end - cpu_start;
    std::cout << "CPU OpenCV blur time: " << cpu_duration.count() << " ms" << std::endl;

    // Save CPU output
    cv::imwrite("output_cpu.jpg", cpu_output);

    // --- GPU Blur (CUDA) ---
    cv::Mat gpu_output;
    apply_blur(input, gpu_output);  // this prints GPU time internally

    // Save GPU output
    cv::imwrite("output_gpu.jpg", gpu_output);

    std::cout << "Output images saved as 'output_cpu.jpg' and 'output_gpu.jpg'" << std::endl;

    return 0;
}
