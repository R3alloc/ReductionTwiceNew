#include "kernel.cuh"
#include "functions.h"

#define IMAGE_SIZE (1024*1024)
#define IMAGE_WIDTH 1024
#define PIXEL_VALUE_MAX 256
int main()
{
	cudaInit();
	
	return 0;
}