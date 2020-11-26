#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <time.h>
#include <cstdio>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>1
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "device_launch_parameters.h"

using namespace cv;
using namespace std;


#define NUM_TREADS 524

__global__
void cudaGrayScale(float *R, float *G, float *B, float* gray, int n) {
	int i = blockDim.x*blockIdx.x + threadIdx.x;
	if (i < n) {
		gray[i] = static_cast<float>((R[i] * 0.21 + G[i] * 0.71 + B[i] * 0.07) / 350.0);
	}
}

void grayscale(float* R, float* G, float* B, float* grayscale, int n) {
	int size = n * sizeof(float);
	float *d_R, *d_G, *d_B, *d_gray;
	cudaMalloc((void **)&d_R, size);
	cudaMemcpy(d_R, R, size, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_G, size);
	cudaMemcpy(d_G, G, size, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_B, size);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&d_gray, size);

	cudaGrayScale << <ceil(n / NUM_TREADS), NUM_TREADS >> >(d_R, d_G, d_B, d_gray, n);
	cudaMemcpy(grayscale, d_gray, size, cudaMemcpyDeviceToHost);

	cudaFree(d_R);
	cudaFree(d_G);
	cudaFree(d_B);
	cudaFree(d_gray);
}

using namespace std;
using namespace cv;


int main() {
	string image_path = "C:/Users/USER/Downloads/opencv/Van_Gogh.jpg";

	Mat matrix_image;
	Mat matrix_filtered;
	string result_image_path;

	matrix_image = imread(image_path, cv::IMREAD_COLOR);// CV_LOAD_IMAGE_COLOR);
	if (!(matrix_image).data) {
		cout << "Error reading image" << endl;
		return 0;
	}

	int cols = matrix_image.cols;
	int rows = matrix_image.rows;
	float* Blue = new float[cols * rows];
	float* Green = new float[cols * rows];
	float* Red = new float[cols * rows];
	float* GrayScaleMatrix = new float[cols * rows];
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			int pos = cols * i + j;
			Blue[pos] = (float)matrix_image.at<cv::Vec3b>(i, j)[0];
			Green[pos] = (float)matrix_image.at<cv::Vec3b>(i, j)[1];
			Red[pos] = (float)matrix_image.at<cv::Vec3b>(i, j)[2];
		}
	}
	grayscale(Red, Green, Blue, GrayScaleMatrix, cols * rows);
	Mat gray = Mat(rows, cols, CV_32FC1, GrayScaleMatrix);
	gray.convertTo(gray, CV_8UC3, 255.0);
	imwrite("./Van_Gogh_grays.jpg", gray);

	return 0;
}

