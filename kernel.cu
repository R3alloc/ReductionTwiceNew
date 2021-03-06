#include "kernel.cuh"
//using namespace std;


void cudaResultCheck(cudaError_t result,char* fileName, char* functionName, int lineNum)
{
	if (result != cudaSuccess)
	{
		cudaError_t error = cudaGetLastError();
		printf("*CUDA error in file %s, \n*function %s, \n*line %d: %s\n",fileName, functionName, lineNum, cudaGetErrorString(error));
	}
	return;
}

void cudaInit(vector<int>& iGPU,
	vector<void*>& stream)
{
	cudaDeviceReset();
	//cudaError_t result = cudaSetDevice(1);
	int deviceNum;
	cudaGetDeviceCount(&deviceNum);
	cout << "Device Number: " << deviceNum << endl;
	for (int i = 0; i < deviceNum; i++)
	{
		iGPU.push_back(i);
		for (int j = 0; j < NUM_STREAM_PER_DEVICE; j++)
		{
			cudaStream_t* newStream = new cudaStream_t;
			cudaStreamCreate(newStream);
			stream.push_back((void*)newStream);
		}
	}
	//cudaError_t result = cudaSetDevice(0);
	//cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);
	return;
}

void cudaEndUp(vector<int>& iGPU,
	vector<void*>& stream)
{
	
	for (int j = 0; j < stream.size(); j++)
	{
		cudaStreamDestroy(*((cudaStream_t*)(stream[j])));
		delete stream[j];
	}	

	/*关于cudaDeviceReset的使用还需要再研究一下 与current device和current process都有关
	for (int i = 0; i < iGPU.size(); i++)
	{
		cudaDeviceReset();
	}
	*/
}

void devMalloc(RFLOAT** devData, int dataSize)
{
	cudaError_t result = cudaMalloc((void**)devData, dataSize);
	cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);
	return;
}

void hostRegister(RFLOAT* imgData, int dataSizeByte)
{
	//注意使用cudaMemcpyAsync的时候用到了stream，要求使用锁页内存。
	//下面的两种方法应该都可以，一个是将已经分配的一段内存register为锁页内存
	//另一个是直接分配新的内存为锁页内存
	cudaError_t result = cudaHostRegister(imgData,dataSizeByte,cudaHostRegisterDefault);
	//cudaError_t result = cudaHostAlloc((void**)&imgData, dataSizeByte, cudaHostAllocDefault);
	cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);
	return;
}

void hostFree(RFLOAT* imgData)
{
	cudaError_t result =  cudaHostUnregister(imgData);
	cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);
	return;
}


void substract(
	vector<void*>& stream,	//这里使用vector，要在头文件中包括进来
	vector<int>& iGPU,
	RFLOAT* imgData,	//imgData当中存储了IMAGE_TOTAL_NUM张照片，但是会在多GPU，多stream当中，再次拆分一次
	int nRow,		//reMask里这个值传入的是_para.size,是图像一条边的长度。
	int nCol,
	RFLOAT radius,	//背景的半径 实际是_para.maskRadius / _para.pixelSize
	int nImg,		//交给substract函数处理的图片总数，在function.cpp当中也是一个batch，一般大小为IMAGE_BATCH
	int nGPU
)
{
	//LOG(INFO) << "Subtract begin.";

	//每个GPU对应一个RFLOAT*
	//已写delete空间代码
	//这一段是不需要的 在reMask当中有这么一段是为了给每台GPU一个mask以方便计算
	//RFLOAT** devSubstract = new RFLOAT*[nGPU];

	//一个image的像素点数量
	size_t imgSizeRL = nRow * nCol;

	//定义stream的总数
	int nStream = nGPU * NUM_STREAM_PER_DEVICE;

	//这个数组当中存储的是指针，每个指针都指向一整段空间地址（能存储BATCH_SIZE张image）
	//int** dev_image_buf = new int*[nStream];
	//int* dev_image_buf[nStream];
	RFLOAT** dev_image_buf = (RFLOAT**)malloc(sizeof(RFLOAT*)*nStream);

	//存储每台GPU上的模板指针
	int** dev_template = (int**)malloc(sizeof(int*) * nStream);

	//用于临时存储
	RFLOAT** dev_tmpImgMean = (RFLOAT**)malloc(sizeof(RFLOAT*) * nStream);
	RFLOAT** dev_tmpImgStddev = (RFLOAT**)malloc(sizeof(RFLOAT*) * nStream);

	int threadInBlock = (nRow > THREAD_PER_BLOCK) ? THREAD_PER_BLOCK : nRow;

	//base Stream
	int baseS;

	int nImgBatch = 0;
	int smidx = 0;

	//为每台GPU上的每个流分配空间，分配BATCH_SIZE张照片的图片
	//可以在这里为每个GPU配备一个背景模板（应该每个GPU配一个还是每个stream配一个？）
	int* host_template = new int[imgSizeRL];
	size_t bgSize;
	hostTmpltGen(host_template, nRow, nCol, radius, &bgSize);

	cout << "bgSize = " << bgSize << endl;
	for (int n = 0; n < nGPU; n++)
	{
		baseS = n * NUM_STREAM_PER_DEVICE;
		cudaSetDevice(iGPU[n]);

		for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
		{
			cout << "Allocate memory for GPU[" << n << "],stream[" << i << "]" << endl;
			cudaError_t result = cudaMalloc((void**)&dev_image_buf[i + baseS], BATCH_SIZE * imgSizeRL * sizeof(RFLOAT));
			cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);

			//每个stream一个模板 分配空间
			result = cudaMalloc((void**)&dev_template[i+baseS], imgSizeRL*sizeof(int));
			cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);
			result = cudaMalloc((void**)&dev_tmpImgMean[i + baseS], imgSizeRL * sizeof(RFLOAT));
			cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);
			result = cudaMalloc((void**)&dev_tmpImgStddev[i + baseS], imgSizeRL * sizeof(RFLOAT));
			cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);

			//将CPU上生成的模板拷贝到GPU上
			result = cudaMemcpy(dev_template[i + baseS], host_template, imgSizeRL * sizeof(int), cudaMemcpyHostToDevice);

		}
	}

	//LOG(INFO) << "alloc memory done, begin to calculate...";

	//已释放
	RFLOAT* partial;
	cudaMalloc((void**)&partial, THREAD_PER_BLOCK * sizeof(RFLOAT));

	for (int i = 0; i < nImg;)
	{

		//多GPU并行 不过本台机器只有一台GPU
		for (int n = 0; n < nGPU; n++)
		{
			//注意这里的跳脱循环不要漏了
			//实际上跳脱出这一层循环之后，也就进入了外层循环i<nImg的判断，也就跳出了外层循环。
			//注意这里大于等于的条件 等于号不要漏了，否则会出现i=nImg的情况还在继续循环，导致访问不到数据。
			if (i >= nImg)
			{
				break;
			}

			//设定device并且开辟空间，检查错误
			cudaSetDevice(iGPU[n]);

			//对于当前GPU的base stream
			baseS = n * NUM_STREAM_PER_DEVICE;
			nImgBatch = (i + BATCH_SIZE < nImg) ? BATCH_SIZE : (nImg - i);
			
			cout << "Start from image " << i <<" smidx="<<smidx<<" baseS="<<baseS<< endl;
			
			//以开始忘了cudaStreamCreate了，所以这里的stream根本没法使用。
			//将数据从host拷贝到device上
			//异步拷贝
			cudaError_t result = cudaMemcpyAsync(
				dev_image_buf[smidx + baseS],
				imgData + i * imgSizeRL,	//注意指针的偏移量，不用去加sizeof(RFLOAT)
				//&(imgData[i * imgSizeRL]),	//这两种指针的用法都是正确的
				nImgBatch * imgSizeRL * sizeof(RFLOAT),
				cudaMemcpyHostToDevice,
				*((cudaStream_t*)(stream[smidx + baseS]))	//由于stream当中的存储类型为void*，这里需要先转换指针类型再解引用。
			);
			cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);
			
			for (int r = 0; r < nImgBatch; r++)
			{
				//计算偏移量
				long long shiftRL = (long long)r * imgSizeRL;
				
				//计算均值
				RFLOAT mean;
				RFLOAT stddev;

				//可以在这里做图片(dev_image_buf[smidx + baseS])[r*imgSizeRL] 与模板进行与操作的 得到一个临时的只有背景的图片
				//然后输入给reductionMean的图片是这个临时图片
				kernel_templateMask << < nCol,
					threadInBlock,
					0,
					*((cudaStream_t*)stream[smidx + baseS]) >> > (
						&((dev_image_buf[smidx + baseS])[r * imgSizeRL]),	//src data
						dev_template[smidx + baseS],
						dev_tmpImgMean[smidx + baseS],	//out
						nRow,
						nCol
						);

				

				//一次只处理一张照片,处理完了之后将数据直接写回dev_image_buf当中
				reductionMean(&mean,	//最后要求的结果：均值
					partial,			//用于存放中间结果的一段device memory
					//&((dev_image_buf[smidx + baseS])[r*imgSizeRL]),	//src data
					dev_tmpImgMean[smidx + baseS],
					imgSizeRL,			//data size
					bgSize,				//背景像素点数量
					nCol,				//grid size 一维
					threadInBlock,	//block size 一维
					*((cudaStream_t*)stream[smidx + baseS]));	//异步拷贝相关的流
				
				

				//计算标准差则使用另一张临时图片 这张临时图片中所有的0都被填成上面计算出来的mean值
				//这样计算标准差的时候非背景的像素点平方差就是0，不会对标准差的计算产生影响
				kernel_templateMaskStddev <<<nCol,
					threadInBlock,
					0,
					*((cudaStream_t*)stream[smidx + baseS]) >>> (
						dev_tmpImgMean[smidx + baseS],	//既是src也是dst
						dev_template[smidx + baseS],
						nRow,
						nCol,
						mean
						);

				

				//计算标准差
				reductionStddev(&stddev,
					partial,
					dev_tmpImgMean[smidx + baseS],	//src data
					//&((dev_image_buf[smidx + baseS])[r*imgSizeRL]),	//src data
					mean,
					imgSizeRL,			//data size
					bgSize,
					//imgSizeRL,			//bg size
					nCol,				//grid size 一维
					threadInBlock,	//block size 一维
					*((cudaStream_t*)stream[smidx + baseS]));	//异步拷贝相关的流

				/*
				//处理dev_image_buf当中的数据
				writeDevBuffer(mean,
					stddev,
					&((dev_image_buf[smidx + baseS])[r * imgSizeRL]),
					imgSizeRL,
					idim, 
					THREAD_PER_BLOCK, 
					*((cudaStream_t*)stream[smidx + baseS]));
					*/

				cout <<"Image "<<r<<": mean = " << mean <<" stddev= "<<stddev<<" bgSize=" <<bgSize<<endl;
			}

			//一个batch里所有的照片处理完毕之后，应该写回imgData当中
			//TODO

			i += nImgBatch;
		}
		smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
	}

	
	cudaFree(partial);
	//delete[] devSubstract;
	
	//为每台GPU上的每个流分配空间，分配BATCH_SIZE张照片的图片
	for (int n = 0; n < nGPU; n++)
	{
		baseS = n * NUM_STREAM_PER_DEVICE;
		cudaSetDevice(iGPU[n]);

		for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
		{
			//这一步的同步也很重要
			cudaError_t result = cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
			cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);

			cout << "Free memory for GPU[" << n << "],stream[" << i << "]" << endl;
			result = cudaFree(dev_image_buf[i + baseS]);
			cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);
		}
	}
}

/*一次只处理一张照片,处理完了之后将数据直接写回dev_image_buf当中*/
//计算均值
__global__ void
kernel_reductionSum(RFLOAT* out, const RFLOAT* in, size_t N)
{
	//这个数组的大小与blockSize有关（也就是blockDim.x）
	//注意这个数组在这里定义的时候虽然没有指定大小，但是在调用这个kernel的时候有一个核函数参数就是用来控制kernel内部使用共享内存的大小。
	extern __shared__ RFLOAT sPartials[];
	RFLOAT sum = 0;
	//tid是当前线程在当前block中的索引
	const int tid = threadIdx.x;
	//i是当前线程在所有线程中的索引
	//i的步长是grid当中的block数量*block中线程的数量
	//in[]存储在全局内存中 输入指针被恰当地对齐，由这段代码发起的全部内存事务将被合并，这将最大限度地提高内存带宽。
	//也就是说一个cuda线程要去多次访问全局内存，然后把这些值加起来
	//这个循环实际上也处理了如果N很小，甚至小于tid的情况：sum有初值为0
	for (size_t i = blockIdx.x * blockDim.x + tid;
		i < N;
		i += blockDim.x * gridDim.x)
	{
		sum += in[i];
	}

	//每个线程把它得到的累计值写入共享内存
	sPartials[tid] = sum;
	//在执行对数步长的规约前进行同步操作
	__syncthreads();

	//blockSize必须是2的整数次方的原因在这里：每一轮都只有上一次一半的线程还在工作
	//对于共享内存中的值 执行对数步长的规约操作
	//共享内存中后半部分的值被添加到前半部分的值上，
	//假设blockDim.x == 1024，则第一轮activeThreads=512
	for (int activeThreads = blockDim.x >> 1;
		activeThreads;
		activeThreads >>= 1) //>>是二进制右移运算符 等价于整除2
							  //>>=是右移且赋值运算符 也就是activeThreads = activeThreads>>1
	{
		if (tid < activeThreads)
		{
			sPartials[tid] += sPartials[tid + activeThreads];
		}
		//每一轮加完之后要线程同步
		__syncthreads();
	}

	//每个block的0号线程存储一个结果，一共有numBlocks个线程，所以存储了这么多个结果。
	if (tid == 0)
	{
		out[blockIdx.x] = sPartials[0];
	}
}


//计算用于计算平方差之和的kernel 
//Sum Of Square Variances
__global__ void
kernel_reductionSumOfSquareVar(RFLOAT* out, const RFLOAT* in, const RFLOAT mean, size_t N)
{

	extern __shared__ RFLOAT sPartials[];

	//这里使用long long RFLOAT是为了避免模拟的时候值溢出
	RFLOAT sum = 0;
	//tid是当前线程在当前block中的索引
	const int tid = threadIdx.x;

	for (size_t i = blockIdx.x * blockDim.x + tid;
		i < N;
		i += blockDim.x * gridDim.x)
	{
		sum += (in[i] - mean)* (in[i] - mean);
	}

	//每个线程把它得到的累计值写入共享内存
	sPartials[tid] = sum;
	//在执行对数步长的规约前进行同步操作
	__syncthreads();

	//blockSize必须是2的整数次方的原因在这里：每一轮都只有上一次一半的线程还在工作
	//对于共享内存中的值 执行对数步长的规约操作
	//共享内存中后半部分的值被添加到前半部分的值上，
	//假设blockDim.x == 1024，则第一轮activeThreads=512
	for (int activeThreads = blockDim.x >> 1;
		activeThreads;
		activeThreads >>= 1) //>>是二进制右移运算符 等价于整除2
							  //>>=是右移且赋值运算符 也就是activeThreads = activeThreads>>1
	{
		if (tid < activeThreads)
		{
			sPartials[tid] += sPartials[tid + activeThreads];
		}
		//每一轮加完之后要线程同步
		__syncthreads();
	}

	//每个block的0号线程存储一个结果，一共有numBlocks个线程，所以存储了这么多个结果。
	if (tid == 0)
	{
		out[blockIdx.x] = sPartials[0];
	}
}

void reductionStddev(RFLOAT* answer,
	RFLOAT* partial,
	const RFLOAT* in,
	const RFLOAT mean,
	size_t N,
	size_t bgSize,
	int numBlocks,
	int numThreads,
	cudaStream_t& stream)
{
	size_t sharedSize = numThreads * sizeof(RFLOAT);
	RFLOAT* dev_sumOfSquareVar;	//用于存储平方差之和

	cudaMalloc(&dev_sumOfSquareVar, sizeof(RFLOAT));

	//第一次的结果partial只是一个中间结果，中间数组保存的是一些平方差的和
	kernel_reductionSumOfSquareVar << <
		numBlocks,
		numThreads,
		sharedSize,
		stream >> > (
			partial,	//长度等于numThreads，中间结果partial的长度跟numThreads有关。
			in,			//长度为N
			mean,
			N
			);

	//第二次结果answer才是最终的计算结果。
	kernel_reductionSum <<<
		1,
		numThreads,
		sharedSize,
		stream >> > (
			//answer,		//长度为1
			dev_sumOfSquareVar,
			partial,	//长度为numBlocks
			numBlocks);


	//经过了这一步，才真正得到了结果
	cudaMemcpyAsync(answer, dev_sumOfSquareVar, sizeof(RFLOAT), cudaMemcpyDeviceToHost, stream);

	//求和之后计算标准差
	*answer = sqrt(*answer / bgSize);

	cudaFree(dev_sumOfSquareVar);
}

//这里调用两遍kernel函数是必须的
//非常重要 注意这里kernel函数的参数 ：block中threads的数量==Reduction1_kernel第二个输入参数（一个数组）的长度，也就是共享内存sharedSize
void
reductionMean(RFLOAT* answer,		//<out> 指向最终结果的指针
	RFLOAT* partial,	//指向存储临时数据 中间数组的指针，应该已经开辟好了空间。数组的长度应该是blockDim.x
	const RFLOAT* in, //存储输入数据的指针
	size_t N,	//输入数据的数量 这里是imgSizeRL
	size_t bgSize,	//背景像素点的数量
	int numBlocks, 
	int numThreads,
	cudaStream_t& stream)
{
	size_t sharedSize = numThreads * sizeof(RFLOAT);
	RFLOAT* dev_mean;

	cudaMalloc(&dev_mean, sizeof(RFLOAT));

	//第一次的结果partial只是一个中间结果，并未完全做和
	kernel_reductionSum <<<
		numBlocks, 
		numThreads, 
		sharedSize,
		stream>>> (
			partial,	//长度等于numThreads，中间结果partial的长度跟numThreads有关。
			in,			//长度为N
			N);

	//第二次结果answer才是最终的计算结果。
	kernel_reductionSum <<<
		1,
		numThreads,
		sharedSize,
		stream >> > (
			//answer,		//长度为1
			dev_mean,
			partial,	//长度为numBlocks
			numBlocks);


	//经过了这一步，才真正得到了结果
	cudaMemcpyAsync(answer, dev_mean, sizeof(RFLOAT), cudaMemcpyDeviceToHost, stream);

	//求和之后计算均值 只要在累加之后除以背景像素点数量即可，非背景的像素点都被置为0了
	*answer = *answer / bgSize;

	cudaFree(dev_mean);

}

//处理dev_image_buf当中的数据
void writeDevBuffer(const RFLOAT mean,
	const RFLOAT stddev,
	RFLOAT* dev_image_buf,
	size_t dataSize,
	size_t numBlocks,		//idim
	size_t numThreads,		//THREAD_PER_BLOCK,
	cudaStream_t& stream)
{
	kernel_writeDevBuffer << <numBlocks,
		numThreads,
		0,
		stream >> > (
			dev_image_buf,
			dataSize,
			mean,
			stddev
			);

}

//这里传递进来的数据是一维的，kernel的启动使用的也是一维的grid，一维的block
__global__ void kernel_writeDevBuffer(
	RFLOAT* dev_image_buf,
	size_t dataSize,
	const RFLOAT mean,
	const RFLOAT stddev
)
{
	const int tid = threadIdx.x;

	for (size_t i = blockIdx.x * blockDim.x + tid;
		i < dataSize;
		i += blockDim.x * gridDim.x)	//就是numBlocks*numThreads
	{
		RFLOAT tmp = dev_image_buf[i];
		tmp -= mean;
		tmp /= stddev;
		dev_image_buf[i] = tmp;
	}
}


/*
__global__ void kernel_templateGen(
	RFLOAT* tmplt,	//输出模板 是个关键字，不能以template作为参数名
	RFLOAT radius,	//确定背景的半径 _para.maskRadius / _para.pixelSize
	int nRow,	//图片的行数
	int nCol	//图片的列数
)
{
	int imgSizeRL = nRow * nCol;

	const int tid = threadIdx.x;

	for (size_t i = blockIdx.x * blockDim.x + tid;
		i < imgSizeRL;
		i += blockDim.x*gridDim.x)
	{
		if()
	}

}
*/

//在host端生成背景模板的函数
void hostTmpltGen(int* host_template, int nRow, int nCol, RFLOAT radius, size_t* bgSize)
{
	int imgSizeRL = nRow * nCol;
	int pixelIndex;
	size_t tmpBgSize = 0;
	for (long j = -nRow / 2; j < nRow / 2; j++)
	{
		for (long i = -nCol / 2; i < nCol / 2; i++)
		{
			pixelIndex = (j >= 0 ? j : j + nRow) * nCol
				+ (i >= 0 ? i : i + nCol);

			//如果对应的像素在radius之外，则是背景，设为1
			if (pow(i, 2) + pow(j, 2) > pow(radius, 2))
			{
				host_template[pixelIndex] = 1;
				tmpBgSize++;
			}
			//对应像素在radius内部，不是背景，不参与计算mean和stddev，所以设置为0
			else
			{
				host_template[pixelIndex] = 0;
			}
		}
	}
	*bgSize = tmpBgSize;

	//showSingleImgInt(host_template, nRow, nCol);
}


__global__ void kernel_templateMask(
	RFLOAT* dev_src,	//src data
	int* dev_template,
	RFLOAT* dev_tmpImgMean,
	int nRow,
	int nCol
)
{
	int imgSizeRL = nRow * nCol;
	int tid = threadIdx.x;
	for (size_t i = blockDim.x * blockIdx.x + tid;
		i < imgSizeRL;
		i += gridDim.x * blockDim.x)
	{
		dev_tmpImgMean[i] = dev_src[i] * dev_template[i];
	}
}

__global__ void kernel_templateMaskStddev(
	RFLOAT* dev_tmpImg,	//既是src也是dst
	int* dev_template,
	int nRow,
	int nCol,
	RFLOAT mean
) 
{
	//变量一定要记得初始化！！！！！！！！
	int imgSizeRL = nCol*nRow;
	int tid = threadIdx.x;

	for (size_t i = blockDim.x * blockIdx.x + tid;
		i < imgSizeRL;
		i += gridDim.x * blockDim.x)
	{
		//这里是如果不是背景，就设置为mean。如果是背景，就加上0
		dev_tmpImg[i] += ( 1 - dev_template[i]) * mean;
		//dev_tmpImg[i] = mean;
		//dev_tmpImg[i] = 0;
	}
}

void showSingleImgInt(int* img, size_t nRow, size_t nCol)
{
	//再遍历行 先从第一行的1，2，3 。。。列开始
	for (size_t i = 0; i < nRow; i++)
	{
		//先遍历列
		for (size_t j = 0; j < nCol; j++)
		{
			cout << img[i*nRow+j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}

void showSingleImg(RFLOAT* img, size_t nRow, size_t nCol)
{
	for (size_t i = 0; i < nRow; i++)
	{
		for (size_t j = 0; j < nCol; j++)
		{
			cout << img[i*nRow+j] << " ";
		}
		cout << endl;
	}
	cout << endl;
}