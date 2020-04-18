#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <time.h>
#include "kernel.cuh"	//为了调用一些在GPU上执行的代码


//#define IMAGE_SIZE (1024*1024)
//#define IMAGE_WIDTH 1024

#define IMAGE_SIZE (64*64)
#define IMAGE_WIDTH 64

#define PIXEL_VALUE_MAX 256
#define SHOW_IMAGE_NUM 16
#define SHOW_IMAGE_PIXEL 16
#define RADIUS (IMAGE_WIDTH/4)
#define IMAGE_TOTAL_NUM 256		//图片的总数本身不要设置为一个太大的值
#define IMAGE_BATCH (IMAGE_TOTAL_NUM/4)	//这是在CPU上分割的batch大小 假定batch的大小为总数的四分之一

#define RFLOAT float

using namespace std;

void imgVecInit(vector<RFLOAT*>& imgVec, int imageNum);

void imgVecFree(vector<RFLOAT*>& imgVec, int imageNum);

void imgVecRandomGen(vector<RFLOAT*>& imgVec);

void imgVecShow(vector<RFLOAT*>& imgVec);

void imgVecCpy(vector<RFLOAT*>& srcImgVec, vector<RFLOAT*>& dstImgVec, int imageNum);

void substractImg(vector<RFLOAT*>& imgVec);

void bgMeanStddev(vector<RFLOAT>& bg, RFLOAT& mean, RFLOAT& stddev, int imageNum);

void substractImgG(vector<RFLOAT*>& imgVec);


