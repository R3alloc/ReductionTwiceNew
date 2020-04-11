
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


void cudaResultCheck(cudaError_t result, char* fileName, char* functionName, int lineNum);

void cudaInit();

void devMalloc(int** devData, int dataSize);

void hostRegister(int* imgData, int dataSizeByte);

void hostFree(int* imgData);
