
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <vector>

#define THREAD_PER_BLOCK 512	//与cuthunder.cu当中一致
#define NUM_STREAM_PER_DEVICE 3	
#define IMAGE_TOTAL_NUM 64		//在function.h中也有定义 图片的总数本身不要设置为一个太大的值
#define IMAGE_BATCH (IMAGE_TOTAL_NUM/4)	//在function.h中也有定义 这是在CPU上分割的batch大小 假定batch的大小为总数的四分之一
#define BATCH_SIZE (IMAGE_BATCH/2)	//这是在GPU上分割的batch大小，每次将数据从host memory拷贝到device memory的大小就是这个

void cudaResultCheck(cudaError_t result, char* fileName, char* functionName, int lineNum);

void cudaInit(vector<int>& iGPU, vector<void*>& stream);

void cudaEndUp(vector<int>& iGPU, vector<void*>& stream);


void devMalloc(int** devData, int dataSize);

void hostRegister(int* imgData, int dataSizeByte);

void hostFree(int* imgData);

void substract(vector<void*>& stream, vector<int>& iGPU, int* imgData, int idim, int batch, int nGPU);

