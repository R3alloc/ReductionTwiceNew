#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <time.h>
#include "kernel.cuh"	//Ϊ�˵���һЩ��GPU��ִ�еĴ���


#define IMAGE_SIZE (64*64)
#define IMAGE_WIDTH 64
#define PIXEL_VALUE_MAX 256
#define SHOW_IMAGE_NUM 16
#define SHOW_IMAGE_PIXEL 16
#define RADIUS (IMAGE_WIDTH/4)
#define IMAGE_TOTAL_NUM 256		//ͼƬ����������Ҫ����Ϊһ��̫���ֵ
#define IMAGE_BATCH (IMAGE_TOTAL_NUM/4)	//������CPU�Ϸָ��batch��С �ٶ�batch�Ĵ�СΪ�������ķ�֮һ


using namespace std;

void imgVecInit(vector<int*>& imgVec, int imageNum);

void imgVecFree(vector<int*>& imgVec, int imageNum);

void imgVecRandomGen(vector<int*>& imgVec);

void imgVecShow(vector<int*>& imgVec);

void imgVecCpy(vector<int*>& srcImgVec, vector<int*>& dstImgVec, int imageNum);

void substractImg(vector<int*>& imgVec);

void bgMeanStddev(vector<int>& bg, int& mean, int& stddev, int imageNum);

void substractImgG(vector<int*>& imgVec);


