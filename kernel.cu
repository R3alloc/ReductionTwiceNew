#include "kernel.cuh"
using namespace std;


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
	//cudaError_t result = cudaSetDevice(1);
	int deviceNum;
	cudaGetDeviceCount(&deviceNum);
	for (int i = 0; i < deviceNum; i++)
	{
		iGPU.push_back(i);
		for (int j = 0; j < NUM_STREAM_PER_DEVICE; j++)
		{
			cudaStream_t* newStream = new cudaStream_t;
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
		delete stream[j];
	}	

	/*����cudaDeviceReset��ʹ�û���Ҫ���о�һ�� ��current device��current process���й�
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
	cudaError_t result = cudaHostRegister(imgData,dataSizeByte,cudaHostRegisterDefault);
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
	vector<void*>& stream,	//����ʹ��vector��Ҫ��ͷ�ļ��а�������
	vector<int>& iGPU,
	int* imgData,	//imgData���д洢��IMAGE_TOTAL_NUM����Ƭ�����ǻ��ڶ�GPU����stream���У��ٴβ��һ��
	int idim,		//reMask�����ֵ�������_para.size,��ͼ��һ���ߵĳ��ȡ�
	int nImg,		//����substract���������ͼƬ��������function.cpp����Ҳ��һ��batch��һ���СΪIMAGE_BATCH
	int nGPU
)
{
	//LOG(INFO) << "Subtract begin.";

	//ÿ��GPU��Ӧһ��int*
	//��дdelete�ռ����
	//��һ���ǲ���Ҫ�� ��reMask��������ôһ����Ϊ�˸�ÿ̨GPUһ��mask�Է������
	//int** devSubstract = new int*[nGPU];

	//һ��image�����ص�����
	size_t imgSizeRL = idim * idim;

	//����stream������
	int nStream = nGPU * NUM_STREAM_PER_DEVICE;

	//������鵱�д洢����ָ�룬ÿ��ָ�붼ָ��һ���οռ��ַ���ܴ洢BATCH_SIZE��image��
	int** dev_image_buf = new int*[nStream];

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
			cudaError_t result = cudaMalloc(&dev_image_buf[i + baseS], BATCH_SIZE * imgSizeRL * sizeof(int));
			cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);
		}
	}

	//LOG(INFO) << "alloc memory done, begin to calculate...";

	for (int i = 0; i < nImg;)
	{

		//��GPU���� ������̨����ֻ��һ̨GPU
		for (int n = 0; n < nGPU; n++)
		{
			//�趨device���ҿ��ٿռ䣬������
			cudaSetDevice(iGPU[n]);
			cudaError_t result = cudaMalloc((void**)&devSubstract[n], imgSizeRL * sizeof(int));
			cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);

			//���ڵ�ǰGPU��base stream
			baseS = n * NUM_STREAM_PER_DEVICE;
			nImgBatch = (i + BATCH_SIZE < nImg) ? BATCH_SIZE : (nImg - i);
			
			//�����ݴ�host������device��
			//�첽����
			result = cudaMemcpyAsync(dev_image_buf[smidx + baseS],
				imgData + i * imgSizeRL,	//ע��ָ���ƫ����������ȥ��sizeof(int)
				nImgBatch * imgSizeRL * sizeof(int),
				cudaMemcpyHostToDevice,
				*((cudaStream_t*)(stream[smidx + baseS]))	//����stream���еĴ洢����Ϊvoid*��������Ҫ��ת��ָ�������ٽ����á�
			);
			cudaResultCheck(result, __FILE__, __FUNCTION__, __LINE__);

			for (int r = 0; r < nImgBatch; n++)
			{
				//����ƫ����
				long long shiftRL = (long long)r * imgSizeRL;
				//һ��ֻ����һ����Ƭ,��������֮������ֱ��д��dev_image_buf����
				kernel_substract <<<
					idim,
					threadInBlock,
					0,
					*((cudaStream_t*)stream[smidx + baseS])>>>(
						dev_image_buf[smidx + baseS],	//���ﱣ���˴�imgData���濽�����������ݣ������޸�֮��ǵ�д��ȥ
						r,								//�������batch���еĵ�r����Ƭ
						idim,							//һ����Ƭ�ĳ���/���
						imgSizeRL						//һ����Ƭ��ʵ�ռ䵱����ռ�ݵ����ص���
						);
			}

			//һ��batch�����е���Ƭ�������֮��Ӧ��д��imgData����
			//TODO


		}

		smidx = (smidx + 1) % NUM_STREAM_PER_DEVICE;
	}
	//delete[] devSubstract;
}