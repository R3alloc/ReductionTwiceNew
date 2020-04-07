#include "kernel.cuh"


void cudaResultCheck(cudaError_t result,char* fileName, char* functionName, int lineNum)
{
	if (result != cudaSuccess)
	{
		cudaError_t error = cudaGetLastError();
		printf("*CUDA error in file %s, \n*function %s, \n*line %d: %s\n",fileName, functionName, lineNum, cudaGetErrorString(error));
	}
	return;
}

void cudaInit()
{
	//cudaError_t result = cudaSetDevice(1);
	cudaError_t result = cudaSetDevice(0);
	cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);
	return;
}

void devMalloc(int** devData, int dataSize)
{
	cudaError_t result = cudaMalloc((void**)devData, dataSize);
	cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);
	return;
}

