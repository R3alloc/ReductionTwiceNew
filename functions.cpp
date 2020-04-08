
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
	cout << "image vector init finished!" << endl;
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
	cout << "image vector free finished!" << endl;
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
	cout << "image vector generate finished!" << endl;
}

//ֱ����ʾһ����ͼƬ�Ĳ�������ֵ
void imgVecShow(vector<int*>& imgVec /**< [inout] �洢���imageָ���vector����  */)
{
	cout << "Show part of image in image vector" << endl;
	for (int imgIdx=0;imgIdx<SHOW_IMAGE_NUM;imgIdx++)
	{
		for (int i = 0; i < SHOW_IMAGE_PIXEL; i++)
		{
			cout << imgVec[imgIdx][i] << " ";
		}
		cout << endl;
	}
}

void imgVecCpy(
	vector<int*>& srcImgVec, /**< [in] �洢���imageָ���vector����  */
	vector<int*>& dstImgVec, /**< [out] �洢���imageָ���vector����  */
	int imageNum				/**< [in] ͼƬ����  */
)
{
	for (int imgIdx = 0; imgIdx < imageNum; imgIdx++)
	{
		for (int pxlIdx = 0; pxlIdx < IMAGE_SIZE; pxlIdx++)
		{
			dstImgVec[imgIdx][pxlIdx] = srcImgVec[imgIdx][pxlIdx];
		}
	}
	cout << "image vector copy finished" << endl;
	
}
void substractImg(vector<int*>& imgVec)
{
	int radius = RADIUS;
	int imageNum = imgVec.size();

	//����ÿһ��ͼƬ
	for (int i = 0; i < imgVec.size(); i++)
	{
		int* image = imgVec[i];
		int mean;
		int stddev;

		//�˴�ģ����ͼ������ϽǶ���Ϊԭ�������뾶 �뾶֮�ⶼ�㱳��
		vector<int> bg;
		for (int pxlIdx = 0; pxlIdx < IMAGE_SIZE; pxlIdx++)
		{
			int row = pxlIdx / IMAGE_WIDTH;
			int col = pxlIdx % IMAGE_SIZE;
			if (col * col + row * row > radius * radius)
			{
				bg.push_back(image[pxlIdx]);
			}
		}

		bgMeanStddev(bg,mean,stddev,imageNum);
		cout << "image " << i << ": mean=" << mean << " stddev=" << stddev << endl;
		for (int j = 0; j < IMAGE_SIZE; j++)
		{
			image[j] -= mean;
			image[j] /= stddev;
		}

	}
	
}

void bgMeanStddev(vector<int>& bg, int& mean, int& stddev, int imageNum)
{
	int sum = 0;
	for (vector<int>::iterator it = bg.begin(); it != bg.end(); it++)
	{
		sum += (*it);
	}
	mean = sum / bg.size();

	//��׼���׼ֵ����ƽ�������ƽ��������ƽ������ƽ����
	//long int quadSum = 0;
	//ע������Ҫ��long long int�Ų������
	long long int quadSum = 0;
	for (vector<int>::iterator it = bg.begin(); it != bg.end(); it++)
	{
		int curPixel = *it;
		quadSum += pow( (curPixel - mean),2);
	}
	stddev = sqrt(quadSum / bg.size());

}

void substractImgG(vector<int*>& imgVec)
{

}