#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
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

const size_t FILTER_SIZE = 11;
const size_t BLOCK_SIZE = 16;

__global__ void imgBlurKernel(unsigned char *outImg, unsigned  char *inImg,
	int width, int height) {
	int filterRow, filterCol;
	int cornerRow, cornerCol;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int filterSize = 2 * FILTER_SIZE + 1;

	// compute global thread coordinates
	int row = by * blockDim.y + ty;
	int col = bx * blockDim.x + tx;

	// make sure thread is within image boundaries
	if ((row < height) && (col < width)) {
		// instantiate accumulator
		int numPixels = 0;
		int cumSum = 0;

		// top-left corner coordinates
		cornerRow = row - FILTER_SIZE;
		cornerCol = col - FILTER_SIZE;

		// accumulate values inside filter
		for (int i = 0; i < filterSize; i++) {
			for (int j = 0; j < filterSize; j++) {
				// filter coordinates
				filterRow = cornerRow + i;
				filterCol = cornerCol + j;

				// accumulate sum
				if ((filterRow >= 0) && (filterRow <= height) && (filterCol >= 0) &&
					(filterCol <= width)) {
					cumSum += inImg[filterRow * width + filterCol];
					numPixels++;
				}
			}
		}
		// set the value of output
		outImg[row * width + col] = (unsigned char)(cumSum / numPixels);
	}
}

void imageBlur(string imageName) {
	// read image
	Mat img;
											  ///IMREAD_COLOR);
	img = imread(imageName, cv::ImreadModes::IMREAD_GRAYSCALE);

	// define img params
	int imgWidth = img.cols;
	int imgHeight = img.rows;
	size_t imgSize = sizeof(unsigned char) * imgWidth * imgHeight;

	// allocate mem for host output image vectors
	unsigned char *h_outImg = (unsigned char *)malloc(imgSize);

	// grab pointer to host input image
	unsigned char *h_inImg = img.data;

	// allocate mem for device input and output
	void *d_inImg;	///char
	void *d_outImg;
	cudaMalloc(&d_inImg, imgSize);
	cudaMalloc(&d_outImg, imgSize);

	// copy the input image from the host to the device
	cudaMemcpy(d_inImg, h_inImg, imgSize, cudaMemcpyHostToDevice);

	// execution configuration parameters + kernel launch
	dim3 dimBlock(32, 32);
	dim3 dimGrid(ceil(imgWidth / 32.0), ceil(imgHeight / 32.0));

	imgBlurKernel <<<dimGrid, dimBlock >>>( (unsigned char*)d_outImg, (unsigned char*)d_inImg, imgWidth, imgHeight);

	// copy output image from device to host
	cudaMemcpy(h_outImg, d_outImg, imgSize, cudaMemcpyDeviceToHost);

	// display images
	Mat imgBlur(imgHeight, imgWidth, CV_8UC1, h_outImg);

	namedWindow("blur", WINDOW_NORMAL);
	imshow("blur", imgBlur);
	waitKey(0);

	imwrite("./blur.jpg", imgBlur);

	// free host and device memory
	img.release();
	imgBlur.release();
	free(h_outImg);
	cudaFree(d_outImg);
	cudaFree(d_inImg);
}


int main(int argc, char **argv) {
	string img_path = "C:/Users/USER/Downloads/opencv/Van_Gogh.jpg";
	//Van_Gogh.jpg
	string image_path(img_path);// , cv::ImreadModes::IMREAD_GRAYSCALE);
	imageBlur(image_path);
	
	return 0;
}