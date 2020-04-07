
#include "functions.h"
//似乎只需要在头文件当中using namespace std即可。
//using namespace std;

void imgVecInit(vector<int*>& imgVec, int imageNum) 
{
	for (int i = 0; i < imageNum; i++)
	{
		int* imgData = new int[IMAGE_SIZE];
		imgVec.push_back(imgData);
	}
}

void imgVecFree(vector<int*>& imgVec, int imageNum)
{
	if (imgVec.size() != imageNum) 
	{
		cout << "vector size is not equal with image number" << endl;
	}
	for (int i = 0; i < imageNum; i++)
	{
		int* imgData = imgVec.back();
		delete[] imgData;
		imgVec.pop_back();
	}
}

void imgVecRandomGen(vector<int*>& imgVec)
{
	srand((unsigned int)(time(NULL)));
	//注意迭代器的定义方法
	for (vector<int*>::iterator it = imgVec.begin(); it != imgVec.end(); it++)
	{
		for (int i = 0; i < IMAGE_SIZE; i++)
		{
			(*it)[i] = rand() % PIXEL_VALUE_MAX;
		}
	}
}

void imgVecShow(vector<int*>& imgVec)
{
	for (int imgIdx=0;imgIdx<SHOW_IMAGE_NUM;imgIdx++)
	{
		for (int i = 0; i < SHOW_IMAGE_PIXEL; i++)
		{
			cout << imgVec[imgIdx][i] << " ";
		}
		cout << endl;
	}
}