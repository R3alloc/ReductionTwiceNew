#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <time.h>
#include "kernel.cuh"	//Ϊ�˵���һЩ��GPU��ִ�еĴ���


//#define IMAGE_SIZE (1024*1024)
//#define IMAGE_WIDTH 1024

#define IMAGE_SIZE (32*32)
#define IMAGE_WIDTH 32
#define BG_RADIUS (IMAGE_WIDTH/2)	//ʵ����_para.maskRadius / _para.pixelSize
#define PIXEL_VALUE_MAX 256
#define SHOW_IMAGE_NUM 16
#define SHOW_IMAGE_PIXEL 16
#define RADIUS (IMAGE_WIDTH/2)
#define IMAGE_TOTAL_NUM 128		//ͼƬ����������Ҫ����Ϊһ��̫���ֵ
#define IMAGE_BATCH (IMAGE_TOTAL_NUM/4)	//������CPU�Ϸָ��batch��С �ٶ�batch�Ĵ�СΪ�������ķ�֮һ

#define RFLOAT float

using namespace std;

void imgVecInit(vector<RFLOAT*>& imgVec, int imageNum);

void imgVecFree(vector<RFLOAT*>& imgVec, int imageNum);

void imgVecRandomGen(vector<RFLOAT*>& imgVec);

void imgVecShow(vector<RFLOAT*>& imgVec);

void imgVecCpy(vector<RFLOAT*>& srcImgVec, vector<RFLOAT*>& dstImgVec, int imageNum);

void substractImgBg(vector<RFLOAT*>& imgVec);

void bgMeanStddev(vector<RFLOAT>& bg, RFLOAT& mean, RFLOAT& stddev, int imageNum);

void substractImgBgG(vector<RFLOAT*>& imgVec);


