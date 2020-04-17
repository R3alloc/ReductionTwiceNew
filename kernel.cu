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

void devMalloc(int** devData, int dataSize)
{
	cudaError_t result = cudaMalloc((void**)devData, dataSize);
	cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);
	return;
}

void hostRegister(int* imgData, int dataSizeByte)
{
	//注意使用cudaMemcpyAsync的时候用到了stream，要求使用锁页内存。
	//下面的两种方法应该都可以，一个是将已经分配的一段内存register为锁页内存
	//另一个是直接分配新的内存为锁页内存
	cudaError_t result = cudaHostRegister(imgData,dataSizeByte,cudaHostRegisterDefault);
	//cudaError_t result = cudaHostAlloc((void**)&imgData, dataSizeByte, cudaHostAllocDefault);
	cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);
	return;
}

void hostFree(int* imgData)
{
	cudaError_t result =  cudaHostUnregister(imgData);
	cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);
	return;
}


void substract(
	vector<void*>& stream,	//这里使用vector，要在头文件中包括进来
	vector<int>& iGPU,
	int* imgData,	//imgData当中存储了IMAGE_TOTAL_NUM张照片，但是会在多GPU，多stream当中，再次拆分一次
	int idim,		//reMask里这个值传入的是_para.size,是图像一条边的长度。
	int nImg,		//交给substract函数处理的图片总数，在function.cpp当中也是一个batch，一般大小为IMAGE_BATCH
	int nGPU
)
{
	//LOG(INFO) << "Subtract begin.";

	//每个GPU对应一个int*
	//已写delete空间代码
	//这一段是不需要的 在reMask当中有这么一段是为了给每台GPU一个mask以方便计算
	//int** devSubstract = new int*[nGPU];

	//一个image的像素点数量
	size_t imgSizeRL = idim * idim;

	//定义stream的总数
	int nStream = nGPU * NUM_STREAM_PER_DEVICE;

	//这个数组当中存储的是指针，每个指针都指向一整段空间地址（能存储BATCH_SIZE张image）
	//int** dev_image_buf = new int*[nStream];
	//int* dev_image_buf[nStream];
	int** dev_image_buf = (int**)malloc(sizeof(int*)*nStream);

	int threadInBlock = (idim > THREAD_PER_BLOCK) ? THREAD_PER_BLOCK : idim;

	//base Stream
	int baseS;

	int nImgBatch = 0;
	int smidx = 0;

	//为每台GPU上的每个流分配空间，分配BATCH_SIZE张照片的图片
	for (int n = 0; n < nGPU; n++)
	{
		baseS = n * NUM_STREAM_PER_DEVICE;
		cudaSetDevice(iGPU[n]);

		for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
		{
			cout << "Allocate memory for GPU[" << n << "],stream[" << i << "]" << endl;
			cudaError_t result = cudaMalloc((void**)&dev_image_buf[i + baseS], BATCH_SIZE * imgSizeRL * sizeof(int));
			cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);
		}
	}

	//LOG(INFO) << "alloc memory done, begin to calculate...";

	//已释放
	int* partial;
	cudaMalloc((void**)&partial, THREAD_PER_BLOCK * sizeof(int));

	for (int i = 0; i < nImg;)
	{
		//注意这里的跳脱循环不要漏了
		//注意这里大于等于的条件 等于号不要漏了，否则会出现i=nImg的情况还在继续循环，导致访问不到数据。
		if (i >= nImg)
		{
			break;
		}

		//多GPU并行 不过本台机器只有一台GPU
		for (int n = 0; n < nGPU; n++)
		{
			//注意这里的跳脱循环不要漏了
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
			cout << "nImgBatch=" << nImgBatch << endl;
			cudaError_t result1 = cudaMemcpy(
				dev_image_buf[smidx + baseS],
				//imgData + i * imgSizeRL,	//注意指针的偏移量，不用去加sizeof(int)
				&(imgData[i * imgSizeRL]),
				nImgBatch * imgSizeRL * sizeof(int),
				cudaMemcpyHostToDevice	//由于stream当中的存储类型为void*，这里需要先转换指针类型再解引用。
			);
			cudaResultCheck(result1, __FILE__, __FUNCTION__, __LINE__);
			for (int ii = 0; ii < nImgBatch; ii++)
			{
				int* test = new int[imgSizeRL];
				cudaMemcpy(test, &dev_image_buf[smidx + baseS][ii * imgSizeRL], imgSizeRL * sizeof(int), cudaMemcpyDeviceToHost);
				cout << "Image " << ii<<": ";
				for (int iii = 0; iii < 16; iii++)
				{
					cout << test[iii] << " ";
				}
				cout << endl;
				delete[] test;
			}
			
			//尝试直接获取stream[]
			cudaStream_t testStream = *((cudaStream_t*)stream[smidx + baseS]);
			cout << "Success to access stream[" << smidx + baseS << "]" << endl;


			//以开始忘了cudaStreamCreate了，所以这里的stream根本没法使用。
			//将数据从host拷贝到device上
			//异步拷贝
			cudaError_t result = cudaMemcpyAsync(
				dev_image_buf[smidx + baseS],
				imgData + i * imgSizeRL,	//注意指针的偏移量，不用去加sizeof(int)
				//&(imgData[i * imgSizeRL]),	//这两种指针的用法都是正确的
				nImgBatch * imgSizeRL * sizeof(int),
				cudaMemcpyHostToDevice,
				*((cudaStream_t*)(stream[smidx + baseS]))	//由于stream当中的存储类型为void*，这里需要先转换指针类型再解引用。
			);
			cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);

			//这里的下标一开始写错了。。。
			result = cudaStreamSynchronize(*((cudaStream_t*)stream[smidx + baseS]));
			cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);

			//测试memory copy
			//这一段代码验证了数据的拷贝是没有问题的
			
			for (int ii = 0; ii < nImgBatch; ii++)
			{
				int* test = new int[imgSizeRL];
				result = cudaMemcpy(test, &dev_image_buf[smidx + baseS][ii*imgSizeRL],  imgSizeRL * sizeof(int), cudaMemcpyDeviceToHost);
				for (int iii = 0; iii < 16; iii++)
				{
					cout << test[iii] << " ";
				}
				cout << endl;
				delete[] test;
			}
			

			for (int r = 0; r < nImgBatch; r++)
			{
				//计算偏移量
				long long shiftRL = (long long)r * imgSizeRL;
				

				//计算均值
				int mean;
				int stddev;

				//一次只处理一张照片,处理完了之后将数据直接写回dev_image_buf当中
				Reduction_mean(&mean,	//最后要求的结果：均值
					partial,			//用于存放中间结果的一段device memory
					&((dev_image_buf[smidx + baseS])[r*imgSizeRL]),	//src data
					imgSizeRL,			//data size
					idim,				//grid size 一维
					THREAD_PER_BLOCK,	//block size 一维
					*((cudaStream_t*)stream[smidx + baseS]));	//异步拷贝相关的流
				//计算标准差
				//TODO

				//处理dev_image_buf当中的数据
				//TODO

				cout <<"Image "<<r<<": mean = " << mean << endl;


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

/*一次只处理一张照片,处理完了之后将数据直接写回dev_image_buf当中
	kernel_substract << <
		idim,			//分为idim个block
		threadInBlock,	//一个block中启动threadInBlock个线程
		0,
		*((cudaStream_t*)stream[smidx + baseS]) >> > (
			dev_image_buf[smidx + baseS],	//这里保存了从imgData里面拷贝过来的数据，做了修改之后记得写回去
			r,								//处理这个batch当中的第r张照片
			idim,							//一张照片的长度/宽度
			imgSizeRL						//一张照片在实空间当中所占据的像素点数
			);
*/
/*
__global__ void kernel_substract(
	int* dev_image,
	int imgIdx,
	int dim,
	size_t imgSizeRL
	)
{
	//grid中的block是一维组织，block中的线程也是一维组织
	int tid = threadIdx.x + blockDim.x * blockIdx.x;



}
*/

__global__ void
Reduction1_kernel(int* out, const int* in, size_t N)
{
	//这个数组的大小与blockSize有关（也就是blockDim.x）
	//注意这个数组在这里定义的时候虽然没有指定大小，但是在调用这个kernel的时候有一个核函数参数就是用来控制kernel内部使用共享内存的大小。
	extern __shared__ int sPartials[];
	int sum = 0;
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

//这里调用两遍kernel函数是必须的
//非常重要 注意这里kernel函数的参数 ：block中threads的数量==Reduction1_kernel第二个输入参数（一个数组）的长度，也就是共享内存sharedSize
void
Reduction_mean(int* answer,		//<out> 指向最终结果的指针
	int* partial,	//指向存储临时数据 中间数组的指针，应该已经开辟好了空间。数组的长度应该是blockDim.x
	const int* in, //存储输入数据的指针
	size_t N,	//输入数据的数量 这里是imgSizeRL
	int numBlocks, 
	int numThreads,
	cudaStream_t& stream)
{
	unsigned int sharedSize = numThreads * sizeof(int);

	//先同步这个流，再做测试
	cudaError_t result = cudaStreamSynchronize(stream);
	cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);

	cout << "In function Reduction_mean: Try to verify memory access" << endl;
	int* test = new int[N];
	cudaMemcpy(test, in, N * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 32; i++)
	{
		cout << test[i] << " ";
	}
	cout << endl;
	delete[] test;

	//第一次的结果partial只是一个中间结果，并未完全做和
	Reduction1_kernel <<<
		numBlocks, 
		numThreads, 
		sharedSize,
		stream>>> (
			partial,	//长度等于numThreads，中间结果partial的长度跟numThreads有关。
			in,			//长度为N
			N);

	result = cudaStreamSynchronize(stream);
	cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);

	int* dev_mean;
	cudaMalloc(&dev_mean, sizeof(int));
	//第二次结果answer才是最终的计算结果。
	Reduction1_kernel << <
		1,
		numThreads,
		sharedSize,
		stream >> > (
			//answer,		//长度为1
			dev_mean,
			partial,	//长度为numBlocks
			numBlocks);

	result = cudaStreamSynchronize(stream);
	cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);

	//经过了这一步，才真正得到了结果
	cudaMemcpyAsync(answer, dev_mean, sizeof(int), cudaMemcpyDeviceToHost, stream);

	//求和之后计算均值
	*answer = *answer / N;

	cudaFree(dev_mean);
}