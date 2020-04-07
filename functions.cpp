
#include "functions.h"
//�ƺ�ֻ��Ҫ��ͷ�ļ�����using namespace std���ɡ�
//using namespace std;

//��ʼ��һ���洢���image��vector��Ϊÿ��image����ռ�
void imgVecInit(
	vector<int*>& imgVec,	/**< [inout] �洢���imageָ���vector����  */
	int imageNum			/**< [in] image������Ҳ��vector�Ĵ�С  */
) 
{
	for (int i = 0; i < imageNum; i++)
	{
		int* imgData = new int[IMAGE_SIZE];
		imgVec.push_back(imgData);
	}
}

//�ͷ�imgVec��������ָ����ָ��Ŀռ�
void imgVecFree(
	vector<int*>& imgVec,	/**< [inout] �洢���imageָ���vector����  */
	int imageNum			/**< [in] image������Ҳ��vector�Ĵ�С  */
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

//ΪimgVec�����е�ͼƬ�������������ֵ
void imgVecRandomGen(vector<int*>& imgVec /**< [inout] �洢���imageָ���vector����  */ )
{
	srand((unsigned int)(time(NULL)));
	//ע��������Ķ��巽��
	for (vector<int*>::iterator it = imgVec.begin(); it != imgVec.end(); it++)
	{
		for (int i = 0; i < IMAGE_SIZE; i++)
		{
			(*it)[i] = rand() % PIXEL_VALUE_MAX;
		}
	}
}

//ֱ����ʾһ����ͼƬ�Ĳ�������ֵ
void imgVecShow(vector<int*>& imgVec /**< [inout] �洢���imageָ���vector����  */)
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