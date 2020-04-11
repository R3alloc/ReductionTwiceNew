#include "kernel.cuh"
#include "functions.h"
#include <vector>
using namespace std;


int main()
{
	
	vector<int*> imgVec;
	vector<int*> oriImgVec;
	vector<int*> cudaImgVec;
	int imageNum = IMAGE_TOTAL_NUM;

	imgVecInit(imgVec, imageNum);
	imgVecInit(oriImgVec, imageNum);

	imgVecRandomGen(oriImgVec);
	imgVecShow(imgVec);

	imgVecCpy( oriImgVec, imgVec, imageNum);
	//imgVecShow(oriImgVec);
	imgVecCpy(oriImgVec, cudaImgVec, imageNum);
	substractImg(cudaImgVec);

	imgVecShow(imgVec);

	imgVecFree(imgVec, imageNum);
	imgVecFree(oriImgVec, imageNum);

	return 0;
}