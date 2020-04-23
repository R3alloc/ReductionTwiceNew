
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"	//__syncthreads()函数需要这个头文件

#include <iostream>
#include <stdio.h>
#include <vector>

#define THREAD_PER_BLOCK 512	//与cuthunder.cu当中不一致
#define NUM_STREAM_PER_DEVICE 3	
#define IMAGE_TOTAL_NUM 128		//在function.h中也有定义 图片的总数本身不要设置为一个太大的值
#define IMAGE_BATCH (IMAGE_TOTAL_NUM/4)	//在function.h中也有定义 这是在CPU上分割的batch大小 假定batch的大小为总数的四分之一
#define BATCH_SIZE (IMAGE_BATCH/4)	//这是在GPU上分割的batch大小，每次将数据从host memory拷贝到device memory的大小就是这个

#define RFLOAT float

using namespace std;

void cudaResultCheck(cudaError_t result, char* fileName, char* functionName, int lineNum);

void cudaInit(vector<int>& iGPU, vector<void*>& stream);

void cudaEndUp(vector<int>& iGPU, vector<void*>& stream);


void devMalloc(RFLOAT** devData, int dataSize);

void hostRegister(RFLOAT* imgData, int dataSizeByte);

void hostFree(RFLOAT* imgData);

void substract(vector<void*>& stream, vector<int>& iGPU, RFLOAT* imgData, int nRow, int nCol, RFLOAT radius, int nImg, int nGPU);



__global__ void kernel_reductionSum(RFLOAT* out, const RFLOAT* in, size_t N);

__global__ void kernel_reductionSumOfSquareVar(RFLOAT* out, const RFLOAT* in, const RFLOAT mean, size_t N);

void reductionStddev(RFLOAT* answer, RFLOAT* partial, const RFLOAT* in, const RFLOAT mean, size_t N, size_t bgSize, int numBlocks, int numThreads, cudaStream_t& stream);


void reductionMean(RFLOAT* answer, RFLOAT* partial, const RFLOAT* in, size_t N, size_t bgSize, int numBlocks, int numThreads, cudaStream_t& stream);

__global__ void kernel_writeDevBuffer(RFLOAT* dev_image_buf, size_t dataSize, const RFLOAT mean, const RFLOAT stddev);

void hostTmpltGen(int* host_template, int nRow, int nCol, RFLOAT radius, size_t* bgSize);

__global__ void kernel_templateMask(RFLOAT* dev_src, int* dev_template, RFLOAT* dev_tmpImgMean, int nRow, int nCol);

__global__ void kernel_templateMaskStddev(RFLOAT* dev_tmpImg, int* dev_template, int nRow, int nCol, RFLOAT mean);

void showSingleImgInt(int* img, size_t nRow, size_t nCol);

void showSingleImg(RFLOAT* img, size_t nRow, size_t nCol);







