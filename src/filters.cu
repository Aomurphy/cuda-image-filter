#include "filters.h"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <iostream>

// CUDA Kernel — 3x3 Box Blur
__global__
void blurKernel(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  
    int y = blockIdx.y * blockDim.y + threadIdx.y;  

    if (x >= width || y >= height) return;  

    int blurSize = 1; // 3x3 kernel (1 pixel in each direction)
    float sum[3] = {0.0f, 0.0f, 0.0f};
    int count = 0;

    for (int ky = -blurSize; ky <= blurSize; ky++) {
        for (int kx = -blurSize; kx <= blurSize; kx++) {
            int nx = min(max(x + kx, 0), width - 1);
            int ny = min(max(y + ky, 0), height - 1);

            int idx = (ny * width + nx) * channels;
            for (int c = 0; c < channels; c++) {
                sum[c] += input[idx + c];
            }
            count++;
        }
    }

    int outIdx = (y * width + x) * channels;
    for (int c = 0; c < channels; c++) {
        output[outIdx + c] = static_cast<unsigned char>(sum[c] / count);
    }
}

// Host Function — Apply Blur using CUDA
void apply_blur(const cv::Mat& input, cv::Mat& output) {
    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();
    size_t imgSize = width * height * channels * sizeof(unsigned char);

    unsigned char* d_input;
    unsigned char* d_output;

    // Allocate GPU memory
    cudaMalloc(&d_input, imgSize);
    cudaMalloc(&d_output, imgSize);

    // Copy image data to GPU
    cudaMemcpy(d_input, input.data, imgSize, cudaMemcpyHostToDevice);

    // Define CUDA grid/block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Launch CUDA kernel
    blurKernel<<<
