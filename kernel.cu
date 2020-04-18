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

	/*����cudaDeviceReset��ʹ�û���Ҫ���о�һ�� ��current device��current process���й�
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
	//ע��ʹ��cudaMemcpyAsync��ʱ���õ���stream��Ҫ��ʹ����ҳ�ڴ档
	//��������ַ���Ӧ�ö����ԣ�һ���ǽ��Ѿ������һ���ڴ�registerΪ��ҳ�ڴ�
	//��һ����ֱ�ӷ����µ��ڴ�Ϊ��ҳ�ڴ�
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
	vector<void*>& stream,	//����ʹ��vector��Ҫ��ͷ�ļ��а�������
	vector<int>& iGPU,
	RFLOAT* imgData,	//imgData���д洢��IMAGE_TOTAL_NUM����Ƭ�����ǻ��ڶ�GPU����stream���У��ٴβ��һ��
	int idim,		//reMask�����ֵ�������_para.size,��ͼ��һ���ߵĳ��ȡ�
	int nImg,		//����substract���������ͼƬ��������function.cpp����Ҳ��һ��batch��һ���СΪIMAGE_BATCH
	int nGPU
)
{
	//LOG(INFO) << "Subtract begin.";

	//ÿ��GPU��Ӧһ��RFLOAT*
	//��дdelete�ռ����
	//��һ���ǲ���Ҫ�� ��reMask��������ôһ����Ϊ�˸�ÿ̨GPUһ��mask�Է������
	//RFLOAT** devSubstract = new RFLOAT*[nGPU];

	//һ��image�����ص�����
	size_t imgSizeRL = idim * idim;

	//����stream������
	int nStream = nGPU * NUM_STREAM_PER_DEVICE;

	//������鵱�д洢����ָ�룬ÿ��ָ�붼ָ��һ���οռ��ַ���ܴ洢BATCH_SIZE��image��
	//int** dev_image_buf = new int*[nStream];
	//int* dev_image_buf[nStream];
	RFLOAT** dev_image_buf = (RFLOAT**)malloc(sizeof(RFLOAT*)*nStream);

	int threadInBlock = (idim > THREAD_PER_BLOCK) ? THREAD_PER_BLOCK : idim;

	//base Stream
	int baseS;

	int nImgBatch = 0;
	int smidx = 0;

	//Ϊÿ̨GPU�ϵ�ÿ��������ռ䣬����BATCH_SIZE����Ƭ��ͼƬ
	for (int n = 0; n < nGPU; n++)
	{
		baseS = n * NUM_STREAM_PER_DEVICE;
		cudaSetDevice(iGPU[n]);

		for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
		{
			cout << "Allocate memory for GPU[" << n << "],stream[" << i << "]" << endl;
			cudaError_t result = cudaMalloc((void**)&dev_image_buf[i + baseS], BATCH_SIZE * imgSizeRL * sizeof(RFLOAT));
			cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);
		}
	}

	//LOG(INFO) << "alloc memory done, begin to calculate...";

	//���ͷ�
	RFLOAT* partial;
	cudaMalloc((void**)&partial, THREAD_PER_BLOCK * sizeof(RFLOAT));

	for (int i = 0; i < nImg;)
	{

		//��GPU���� ������̨����ֻ��һ̨GPU
		for (int n = 0; n < nGPU; n++)
		{
			//ע�����������ѭ����Ҫ©��
			//ʵ�������ѳ���һ��ѭ��֮��Ҳ�ͽ��������ѭ��i<nImg���жϣ�Ҳ�����������ѭ����
			//ע��������ڵ��ڵ����� ���ںŲ�Ҫ©�ˣ���������i=nImg��������ڼ���ѭ�������·��ʲ������ݡ�
			if (i >= nImg)
			{
				break;
			}

			//�趨device���ҿ��ٿռ䣬������
			cudaSetDevice(iGPU[n]);

			//���ڵ�ǰGPU��base stream
			baseS = n * NUM_STREAM_PER_DEVICE;
			nImgBatch = (i + BATCH_SIZE < nImg) ? BATCH_SIZE : (nImg - i);
			
			cout << "Start from image " << i <<" smidx="<<smidx<<" baseS="<<baseS<< endl;
			
			//�Կ�ʼ����cudaStreamCreate�ˣ����������stream����û��ʹ�á�
			//�����ݴ�host������device��
			//�첽����
			cudaError_t result = cudaMemcpyAsync(
				dev_image_buf[smidx + baseS],
				imgData + i * imgSizeRL,	//ע��ָ���ƫ����������ȥ��sizeof(RFLOAT)
				//&(imgData[i * imgSizeRL]),	//������ָ����÷�������ȷ��
				nImgBatch * imgSizeRL * sizeof(RFLOAT),
				cudaMemcpyHostToDevice,
				*((cudaStream_t*)(stream[smidx + baseS]))	//����stream���еĴ洢����Ϊvoid*��������Ҫ��ת��ָ�������ٽ����á�
			);
			cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);
			
			for (int r = 0; r < nImgBatch; r++)
			{
				//����ƫ����
				long long shiftRL = (long long)r * imgSizeRL;
				
				//�����ֵ
				RFLOAT mean;
				RFLOAT stddev;

				//һ��ֻ����һ����Ƭ,��������֮������ֱ��д��dev_image_buf����
				Reduction_mean(&mean,	//���Ҫ��Ľ������ֵ
					partial,			//���ڴ���м�����һ��device memory
					&((dev_image_buf[smidx + baseS])[r*imgSizeRL]),	//src data
					imgSizeRL,			//data size
					idim,				//grid size һά
					THREAD_PER_BLOCK,	//block size һά
					*((cudaStream_t*)stream[smidx + baseS]));	//�첽������ص���
				//�����׼��
				//TODO

				Reduction_stddev(&stddev,
					partial,
					&((dev_image_buf[smidx + baseS])[r * imgSizeRL]),	//src data
					mean,
					imgSizeRL,			//data size
					idim,				//grid size һά
					THREAD_PER_BLOCK,	//block size һά
					*((cudaStream_t*)stream[smidx + baseS]));	//�첽������ص���

				//����dev_image_buf���е�����
				//TODO

				cout <<"Image "<<r<<": mean = " << mean <<" stddev= "<<stddev<< endl;
			}

			//һ��batch�����е���Ƭ�������֮��Ӧ��д��imgData����
			//TODO

			i += nImgBatch;
		}
		smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
	}

	
	cudaFree(partial);
	//delete[] devSubstract;
	
	//Ϊÿ̨GPU�ϵ�ÿ��������ռ䣬����BATCH_SIZE����Ƭ��ͼƬ
	for (int n = 0; n < nGPU; n++)
	{
		baseS = n * NUM_STREAM_PER_DEVICE;
		cudaSetDevice(iGPU[n]);

		for (int i = 0; i < NUM_STREAM_PER_DEVICE; i++)
		{
			//��һ����ͬ��Ҳ����Ҫ
			cudaError_t result = cudaStreamSynchronize(*((cudaStream_t*)stream[i + baseS]));
			cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);

			cout << "Free memory for GPU[" << n << "],stream[" << i << "]" << endl;
			result = cudaFree(dev_image_buf[i + baseS]);
			cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);
		}
	}
}

/*һ��ֻ����һ����Ƭ,��������֮������ֱ��д��dev_image_buf����*/
//�����ֵ
__global__ void
kernel_reductionSum(RFLOAT* out, const RFLOAT* in, size_t N)
{
	//�������Ĵ�С��blockSize�йأ�Ҳ����blockDim.x��
	//ע��������������ﶨ���ʱ����Ȼû��ָ����С�������ڵ������kernel��ʱ����һ���˺�������������������kernel�ڲ�ʹ�ù����ڴ�Ĵ�С��
	extern __shared__ RFLOAT sPartials[];
	RFLOAT sum = 0;
	//tid�ǵ�ǰ�߳��ڵ�ǰblock�е�����
	const int tid = threadIdx.x;
	//i�ǵ�ǰ�߳��������߳��е�����
	//i�Ĳ�����grid���е�block����*block���̵߳�����
	//in[]�洢��ȫ���ڴ��� ����ָ�뱻ǡ���ض��룬����δ��뷢���ȫ���ڴ����񽫱��ϲ����⽫����޶ȵ�����ڴ����
	//Ҳ����˵һ��cuda�߳�Ҫȥ��η���ȫ���ڴ棬Ȼ�����Щֵ������
	//���ѭ��ʵ����Ҳ���������N��С������С��tid�������sum�г�ֵΪ0
	for (size_t i = blockIdx.x * blockDim.x + tid;
		i < N;
		i += blockDim.x * gridDim.x)
	{
		sum += in[i];
	}

	//ÿ���̰߳����õ����ۼ�ֵд�빲���ڴ�
	sPartials[tid] = sum;
	//��ִ�ж��������Ĺ�Լǰ����ͬ������
	__syncthreads();

	//blockSize������2�������η���ԭ�������ÿһ�ֶ�ֻ����һ��һ����̻߳��ڹ���
	//���ڹ����ڴ��е�ֵ ִ�ж��������Ĺ�Լ����
	//�����ڴ��к�벿�ֵ�ֵ����ӵ�ǰ�벿�ֵ�ֵ�ϣ�
	//����blockDim.x == 1024�����һ��activeThreads=512
	for (int activeThreads = blockDim.x >> 1;
		activeThreads;
		activeThreads >>= 1) //>>�Ƕ�������������� �ȼ�������2
							  //>>=�������Ҹ�ֵ����� Ҳ����activeThreads = activeThreads>>1
	{
		if (tid < activeThreads)
		{
			sPartials[tid] += sPartials[tid + activeThreads];
		}
		//ÿһ�ּ���֮��Ҫ�߳�ͬ��
		__syncthreads();
	}

	//ÿ��block��0���̴߳洢һ�������һ����numBlocks���̣߳����Դ洢����ô��������
	if (tid == 0)
	{
		out[blockIdx.x] = sPartials[0];
	}
}


//�������ڼ���ƽ����֮�͵�kernel 
//Sum Of Square Variances
__global__ void
kernel_reductionSumOfSquareVar(RFLOAT* out, const RFLOAT* in, const RFLOAT mean, size_t N)
{

	extern __shared__ RFLOAT sPartials[];

	//����ʹ��long long RFLOAT��Ϊ�˱���ģ���ʱ��ֵ���
	RFLOAT sum = 0;
	//tid�ǵ�ǰ�߳��ڵ�ǰblock�е�����
	const int tid = threadIdx.x;

	for (size_t i = blockIdx.x * blockDim.x + tid;
		i < N;
		i += blockDim.x * gridDim.x)
	{
		sum += (in[i] - mean)* (in[i] - mean);
	}

	//ÿ���̰߳����õ����ۼ�ֵд�빲���ڴ�
	sPartials[tid] = sum;
	//��ִ�ж��������Ĺ�Լǰ����ͬ������
	__syncthreads();

	//blockSize������2�������η���ԭ�������ÿһ�ֶ�ֻ����һ��һ����̻߳��ڹ���
	//���ڹ����ڴ��е�ֵ ִ�ж��������Ĺ�Լ����
	//�����ڴ��к�벿�ֵ�ֵ����ӵ�ǰ�벿�ֵ�ֵ�ϣ�
	//����blockDim.x == 1024�����һ��activeThreads=512
	for (int activeThreads = blockDim.x >> 1;
		activeThreads;
		activeThreads >>= 1) //>>�Ƕ�������������� �ȼ�������2
							  //>>=�������Ҹ�ֵ����� Ҳ����activeThreads = activeThreads>>1
	{
		if (tid < activeThreads)
		{
			sPartials[tid] += sPartials[tid + activeThreads];
		}
		//ÿһ�ּ���֮��Ҫ�߳�ͬ��
		__syncthreads();
	}

	//ÿ��block��0���̴߳洢һ�������һ����numBlocks���̣߳����Դ洢����ô��������
	if (tid == 0)
	{
		out[blockIdx.x] = sPartials[0];
	}
}

void Reduction_stddev(RFLOAT* answer,
	RFLOAT* partial,
	const RFLOAT* in,
	const RFLOAT mean,
	size_t N,
	int numBlocks,
	int numThreads,
	cudaStream_t& stream)
{
	size_t sharedSize = numThreads * sizeof(RFLOAT);
	RFLOAT* dev_sumOfSquareVar;	//���ڴ洢ƽ����֮��

	cudaMalloc(&dev_sumOfSquareVar, sizeof(RFLOAT));

	//��һ�εĽ��partialֻ��һ���м������м����鱣�����һЩƽ����ĺ�
	kernel_reductionSumOfSquareVar << <
		numBlocks,
		numThreads,
		sharedSize,
		stream >> > (
			partial,	//���ȵ���numThreads���м���partial�ĳ��ȸ�numThreads�йء�
			in,			//����ΪN
			mean,
			N
			);

	//�ڶ��ν��answer�������յļ�������
	kernel_reductionSum << <
		1,
		numThreads,
		sharedSize,
		stream >> > (
			//answer,		//����Ϊ1
			dev_sumOfSquareVar,
			partial,	//����ΪnumBlocks
			numBlocks);


	//��������һ�����������õ��˽��
	cudaMemcpyAsync(answer, dev_sumOfSquareVar, sizeof(RFLOAT), cudaMemcpyDeviceToHost, stream);

	//���֮������׼��
	*answer = sqrt(*answer / N);

	cudaFree(dev_sumOfSquareVar);
}

//�����������kernel�����Ǳ����
//�ǳ���Ҫ ע������kernel�����Ĳ��� ��block��threads������==Reduction1_kernel�ڶ������������һ�����飩�ĳ��ȣ�Ҳ���ǹ����ڴ�sharedSize
void
Reduction_mean(RFLOAT* answer,		//<out> ָ�����ս����ָ��
	RFLOAT* partial,	//ָ��洢��ʱ���� �м������ָ�룬Ӧ���Ѿ����ٺ��˿ռ䡣����ĳ���Ӧ����blockDim.x
	const RFLOAT* in, //�洢�������ݵ�ָ��
	size_t N,	//�������ݵ����� ������imgSizeRL
	int numBlocks, 
	int numThreads,
	cudaStream_t& stream)
{
	size_t sharedSize = numThreads * sizeof(RFLOAT);
	RFLOAT* dev_mean;

	cudaMalloc(&dev_mean, sizeof(RFLOAT));

	//��һ�εĽ��partialֻ��һ���м�������δ��ȫ����
	kernel_reductionSum <<<
		numBlocks, 
		numThreads, 
		sharedSize,
		stream>>> (
			partial,	//���ȵ���numThreads���м���partial�ĳ��ȸ�numThreads�йء�
			in,			//����ΪN
			N);

	//�ڶ��ν��answer�������յļ�������
	kernel_reductionSum <<<
		1,
		numThreads,
		sharedSize,
		stream >> > (
			//answer,		//����Ϊ1
			dev_mean,
			partial,	//����ΪnumBlocks
			numBlocks);


	//��������һ�����������õ��˽��
	cudaMemcpyAsync(answer, dev_mean, sizeof(RFLOAT), cudaMemcpyDeviceToHost, stream);

	//���֮������ֵ
	*answer = *answer / N;

	cudaFree(dev_mean);

}
