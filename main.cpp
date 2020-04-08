#include "kernel.cuh"
#include "functions.h"
#include <vector>
using namespace std;


int main()
{
	cudaInit();
	vector<int*> imgVec;
	vector<int*> oriImgVec;
	int imageNum = 32;

	imgVecInit(imgVec, imageNum);
	imgVecInit(oriImgVec, imageNum);

	imgVecRandomGen(imgVec);
	imgVecShow(imgVec);

	imgVecCpy(imgVec, oriImgVec, imageNum);
	imgVecShow(oriImgVec);

	substractImg(imgVec);
	imgVecShow(imgVec);

	imgVecFree(imgVec, imageNum);
	imgVecFree(oriImgVec, imageNum);

	return 0;
}