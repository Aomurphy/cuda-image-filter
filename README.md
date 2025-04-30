# CUDA Image Filter

CUDA-accelerated image processing tool applying blur, sharpen, and edge detection filters to high-resolution images.

## Features

- Blur filter  
- Sharpen filter  
- Edge detection (Sobel)  
- 6× speedup over CPU baseline (tested on 1080p images)

## Demo

![Sample output](images/sample.jpg) *(Optional — add result image later)*

## Requirements

- CUDA Toolkit  
- OpenCV (with CUDA support)

### Install OpenCV (Ubuntu)

```bash
sudo apt-get install libopencv-dev
