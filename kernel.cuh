
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"	//__syncthreads()������Ҫ���ͷ�ļ�

#include <iostream>
#include <stdio.h>
#include <vector>

#define THREAD_PER_BLOCK 512	//��cuthunder.cu���в�һ��
#define NUM_STREAM_PER_DEVICE 3	
#define IMAGE_TOTAL_NUM 256		//��function.h��Ҳ�ж��� ͼƬ����������Ҫ����Ϊһ��̫���ֵ
#define IMAGE_BATCH (IMAGE_TOTAL_NUM/4)	//��function.h��Ҳ�ж��� ������CPU�Ϸָ��batch��С �ٶ�batch�Ĵ�СΪ�������ķ�֮һ
#define BATCH_SIZE (IMAGE_BATCH/4)	//������GPU�Ϸָ��batch��С��ÿ�ν����ݴ�host memory������device memory�Ĵ�С�������
using namespace std;

void cudaResultCheck(cudaError_t result, char* fileName, char* functionName, int lineNum);

void cudaInit(vector<int>& iGPU, vector<void*>& stream);

void cudaEndUp(vector<int>& iGPU, vector<void*>& stream);


void devMalloc(int** devData, int dataSize);

void hostRegister(int* imgData, int dataSizeByte);

void hostFree(int* imgData);

void substract(vector<void*>& stream, vector<int>& iGPU, int* imgData, int idim, int batch, int nGPU);

__global__ void Reduction1_kernel(int* out, const int* in, size_t N);

void Reduction_mean(int* answer, int* partial, const int* in, size_t N, int numBlocks, int numThreads, cudaStream_t& stream);

