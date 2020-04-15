
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
	//��������̣߳����˳����Ǽ����̣߳��ٲ�����㡣
	//TODO

	//ģ��
	vector<void*> _stream;
	vector<int> _iGPU;
	int _nGPU = 1;	//GPU������
	cudaInit(_iGPU, _stream);
	//ģ��

	int* image;
	int mean;
	int stddev;
	int nImg = IMAGE_TOTAL_NUM;	//Ҫ�����ͼƬ����
	int batch = IMAGE_BATCH;	//һ�δ���ͼƬ������
	int dimSizeRL = IMAGE_SIZE;	//һ��ͼƬ��ʵ�ռ䵱�е���������
	int* imgData = (int*)malloc(sizeof(int) * IMAGE_BATCH * dimSizeRL);
	hostRegister(imgData, IMAGE_BATCH * dimSizeRL*sizeof(int));
	
	//l�൱��ÿһ��ѭ���е�һ��base��ƫ�ƻ�׼��
	for (int l = 0; l < nImg;)
	{
		if (l >= nImg)
		{
			break;
		}
		//����ÿһ�ֵ�batch��С����󲻳���IMAGE_BATCH �����һ�ֻ�С�ڵ���IMAGE_BATCH�������ִζ��ǵ���IMAGE_BATCH
		batch = (l + IMAGE_BATCH < nImg) ? IMAGE_BATCH : (nImg - l);

		//��ʼ����ǰbatch�е�ÿһ��ͼƬ
		//����ǰbatch�е����ݶ����浽imgData���һά������ ע�⣬�����ڶ����ݵĹ������ƺ���ʹ�õ�memoryBazaar
		for (int i = 0; i < batch; i++)
		{
			for (int n = 0; n < dimSizeRL; n++)
			{
				imgData[i * dimSizeRL + n] = imgVec[l + i][n];
			}
		}

		//�����batch �е����ݽ���GPUȥ����
		//��Ҫ����mean��stddev��ҲҪ�޸Ķ�Ӧ�����ݣ�����д��imgData
		substract(_stream,
			_iGPU,
			imgData,
			IMAGE_WIDTH,		//reMask�����ֵ�������_para.size,��ͼ��һ���ߵĳ��ȡ��������ó�IMAGE_WIDTH
			batch,
			_nGPU
		);

		//�������������д��imgVec
		//TODO

		l += batch;
	}

	//��ȫ��������ɺ��ͷ�CPU�ϵ���ҳ�ڴ�
	hostFree(imgData);
	free(imgData);

	cudaEndUp(_iGPU, _stream);
}