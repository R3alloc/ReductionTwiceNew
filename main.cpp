#include "kernel.cuh"
#include "functions.h"
#include <vector>
using namespace std;


int main()
{
	cudaInit();
	vector<int*> imgVec;
	int imageNum = 32;
	imgVecInit(imgVec, imageNum);
	imgVecRandomGen(imgVec);
	imgVecShow(imgVec);
	imgVecFree(imgVec, imageNum);

	return 0;
}