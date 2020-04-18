#include "kernel.cuh"
#include "functions.h"
#include <vector>
using namespace std;


RFLOAT main()
{
	
	vector<RFLOAT*> imgVec;
	vector<RFLOAT*> oriImgVec;
	vector<RFLOAT*> cudaImgVec;
	RFLOAT imageNum = IMAGE_TOTAL_NUM;

	imgVecInit(imgVec, imageNum);
	imgVecInit(oriImgVec, imageNum);
	imgVecInit(cudaImgVec, imageNum);

	imgVecRandomGen(oriImgVec);
	imgVecShow(oriImgVec);

	imgVecCpy( oriImgVec, imgVec, imageNum);
	//imgVecShow(oriImgVec);
	imgVecCpy(oriImgVec, cudaImgVec, imageNum);
	substractImg(imgVec);
	imgVecShow(imgVec);

	substractImgG(cudaImgVec);

	imgVecFree(imgVec, imageNum);
	imgVecFree(oriImgVec, imageNum);

	return 0;
}