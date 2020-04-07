
#include "functions.h"
//似乎只需要在头文件当中using namespace std即可。
//using namespace std;

//初始化一个存储多个image的vector，为每个image分配空间
void imgVecInit(
	vector<int*>& imgVec,	/**< [inout] 存储多个image指针的vector引用  */
	int imageNum			/**< [in] image数量，也是vector的大小  */
) 
{
	for (int i = 0; i < imageNum; i++)
	{
		int* imgData = new int[IMAGE_SIZE];
		imgVec.push_back(imgData);
	}
}

//释放imgVec当中所有指针所指向的空间
void imgVecFree(
	vector<int*>& imgVec,	/**< [inout] 存储多个image指针的vector引用  */
	int imageNum			/**< [in] image数量，也是vector的大小  */
)
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

//为imgVec中所有的图片，生成随机像素值
void imgVecRandomGen(vector<int*>& imgVec /**< [inout] 存储多个image指针的vector引用  */ )
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

//直接显示一部分图片的部分像素值
void imgVecShow(vector<int*>& imgVec /**< [inout] 存储多个image指针的vector引用  */)
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