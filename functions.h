#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <time.h>
#include "kernel.cuh"	//为了调用一些在GPU上执行的代码


#define IMAGE_SIZE (1024*1024)
#define IMAGE_WIDTH 1024
#define PIXEL_VALUE_MAX 256
#define SHOW_IMAGE_NUM 16
#define SHOW_IMAGE_PIXEL 16
#define RADIUS (IMAGE_WIDTH/4)
#define IMAGE_TOTAL_NUM 64		//图片的总数本身不要设置为一个太大的值
#define IMAGE_BATCH (IMAGE_TOTAL_NUM/4)	//这是在CPU上分割的batch大小 假定batch的大小为总数的四分之一


using namespace std;

void imgVecInit(vector<int*>& imgVec, int imageNum);

void imgVecFree(vector<int*>& imgVec, int imageNum);

void imgVecRandomGen(vector<int*>& imgVec);

void imgVecShow(vector<int*>& imgVec);

void imgVecCpy(vector<int*>& srcImgVec, vector<int*>& dstImgVec, int imageNum);

void substractImg(vector<int*>& imgVec);

void bgMeanStddev(vector<int>& bg, int& mean, int& stddev, int imageNum);


