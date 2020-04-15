
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
	cout << "image vector init finished!" << endl;
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
	cout << "image vector free finished!" << endl;
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
	cout << "image vector generate finished!" << endl;
}

//直接显示一部分图片的部分像素值
void imgVecShow(vector<int*>& imgVec /**< [inout] 存储多个image指针的vector引用  */)
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
	vector<int*>& srcImgVec, /**< [in] 存储多个image指针的vector引用  */
	vector<int*>& dstImgVec, /**< [out] 存储多个image指针的vector引用  */
	int imageNum				/**< [in] 图片数量  */
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

	//处理每一张图片
	for (int i = 0; i < imgVec.size(); i++)
	{
		int* image = imgVec[i];
		int mean;
		int stddev;

		//此处模拟以图像的左上角顶点为原点来画半径 半径之外都算背景
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

	//标准差：标准值与其平均数离差平方的算术平均数的平方根
	//long int quadSum = 0;
	//注意这里要用long long int才不会溢出
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
	//如果是主线程，就退出。是计算线程，再参与计算。
	//TODO

	//模拟
	vector<void*> _stream;
	vector<int> _iGPU;
	int _nGPU = 1;	//GPU的数量
	cudaInit(_iGPU, _stream);
	//模拟

	int* image;
	int mean;
	int stddev;
	int nImg = IMAGE_TOTAL_NUM;	//要处理的图片总数
	int batch = IMAGE_BATCH;	//一次处理图片的数量
	int dimSizeRL = IMAGE_SIZE;	//一张图片在实空间当中的像素数量
	int* imgData = (int*)malloc(sizeof(int) * IMAGE_BATCH * dimSizeRL);
	hostRegister(imgData, IMAGE_BATCH * dimSizeRL*sizeof(int));
	
	//l相当于每一轮循环中的一个base，偏移基准。
	for (int l = 0; l < nImg;)
	{
		if (l >= nImg)
		{
			break;
		}
		//设置每一轮的batch大小，最大不超过IMAGE_BATCH 在最后一轮会小于等于IMAGE_BATCH；其他轮次都是等于IMAGE_BATCH
		batch = (l + IMAGE_BATCH < nImg) ? IMAGE_BATCH : (nImg - l);

		//开始处理当前batch中的每一张图片
		//将当前batch中的数据都保存到imgData这个一维数组中 注意，这里在读数据的过程中似乎会使用到memoryBazaar
		for (int i = 0; i < batch; i++)
		{
			for (int n = 0; n < dimSizeRL; n++)
			{
				imgData[i * dimSizeRL + n] = imgVec[l + i][n];
			}
		}

		//将这个batch 中的数据交给GPU去处理
		//既要计算mean，stddev，也要修改对应的数据，并且写回imgData
		substract(_stream,
			_iGPU,
			imgData,
			IMAGE_WIDTH,		//reMask里这个值传入的是_para.size,是图像一条边的长度。这里设置成IMAGE_WIDTH
			batch,
			_nGPU
		);

		//将处理完的数据写回imgVec
		//TODO

		l += batch;
	}

	//在全部计算完成后，释放CPU上的锁页内存
	hostFree(imgData);
	free(imgData);

	cudaEndUp(_iGPU, _stream);
}