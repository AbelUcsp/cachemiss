#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <time.h>
#include <cstdio>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "device_launch_parameters.h"

using namespace cv;
using namespace std;


#define NUM_TREADS 524

using namespace cv;


__global__ void rgb2grayKernel(unsigned char *Pout, unsigned char *Pin, int width,
                            int height, int numChannels) {
  // compute global thread coordinates
  int row = threadIdx.y + blockIdx.y * blockDim.y;
  int col = threadIdx.x + blockIdx.x * blockDim.x;

  // linearize coordinates for data access
  int grayOffset = row * width + col;


  if ((col < width) && (row < height)) {

	int rgbOffset = greyOffset*numChannels;
	unsigned char r = Pin[rgbOffset ]; // red value for pixel
	unsigned char g = Pin[rgbOffset + 2]; // green value for pixel
	unsigned char b = Pin[rgbOffset + 3]; // blue value for pixel

	Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;

  }
}

void grayScale(string fileName) {
  // read image
  Mat image;
  image = imread(fileName, CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
    cout << "Cannot read image file " << fileName;
    exit(1);
  }

  // define img params and timers
  int imageChannels = 3;
  int imageWidth = image.cols;
  int imageHeight = image.rows;
  size_t size_rgb = sizeof(unsigned char) * imageWidth * imageHeight * imageChannels;
  size_t size_gray = sizeof(unsigned char) * imageWidth * imageHeight;

  // allocate mem for host image vectors
  unsigned char *h_grayImage = (unsigned char *)malloc(size_rgb);

  // grab pointer to host rgb image
  unsigned char *h_rgbImage = image.data;

  // allocate mem for device rgb and gray
  unsigned char *d_rgbImage;
  unsigned char *d_grayImage;
  cudaMalloc(&d_rgbImage, size_rgb);
  cudaMalloc(&d_grayImage, size_gray);

  // copy the rgb image from the host to the device and record the needed time
  cudaMemcpy(d_rgbImage, h_rgbImage, size_rgb, cudaMemcpyHostToDevice);

  // execution configuration parameters + kernel launch
  dim3 dimBlock(16, 16);
  dim3 dimGrid(ceil(imageWidth / 16.0), ceil(imageHeight / 16.0));
  rgb2grayKernel<<<dimGrid, dimBlock>>>(d_grayImage, d_rgbImage, imageWidth,
                                     imageHeight, imageChannels);

  cudaMemcpy(h_grayImage, d_grayImage, size_gray, cudaMemcpyDeviceToHost);

  // display images
  Mat imageGray(imageHeight, imageWidth, CV_8UC1, h_grayImage);

  imwrite("./grayscale.jpg", imageGray);

  // free host and device memory
  image.release();
  imageGray.release();
  free(h_grayImage);

  cudaFree(d_rgbImage);
  cudaFree(d_grayImage);
}




int main(int argc, char **argv) {
		string img_path = "C:/Users/USER/Downloads/opencv/Van_Gogh.jpg";
		//Van_Gogh.jpg
		string image_path(img_path);// , cv::ImreadModes::IMREAD_GRAYSCALE);
		grayScale(image_path)
		
		return 0;
}
