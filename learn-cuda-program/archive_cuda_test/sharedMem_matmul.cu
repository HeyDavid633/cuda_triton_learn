// 计算矩阵乘法：C = A * B (M*K，K*N)
#include<cuda.h>
#include<iostream>
#include<sys/time.h>


#define M 512
#define K 512
#define N 512

void initial(float *array, int size)
{
	for (int i = 0; i < size; i++){
		array[i] = (float)(rand() % 10 + 1);
	}
}

//核函数（传入显存ABC以及维度信息MNK）
__global__ void multiplicateMatrix(float *array_A, float *array_B, float *array_C, int M_p, int K_p, int N_p)
{
	//这里我们划分的lblock和grid是二维的，分别计算线程的二维索引（x方向和y方向的索引）
	int col = threadIdx.x + blockDim.x*blockIdx.x;  //col 
	int row = threadIdx.y + blockDim.y*blockIdx.y;  //row

	if ( col >= N_p && row >= M_p) return;

    float sum = 0;
    for (int k = 0; k < K_p; k++) {
        sum += array_A[row*K_p + k] * array_B[k*N_p + col];
    }
    array_C[row*N_p + col] = sum;
	
}

//核函数（静态共享内存版）
__global__ void matrixMultiplyShared(float *A, float *B, float *C)
{
	//分配共享内存
	__shared__ float sharedM[32][32];
	__shared__ float sharedN[32][32];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
    if ( col >= N && row >= M) return;

	float Csub = 0.0;
    int numBlocks = (K + 32 - 1) / 32;

	
	//核心：下面将保存在全局内存中的矩阵M&N分块存放到共享内存中
	for (int i = 0; i < numBlocks; i++)//如上图，将一个红框矩形分成多个正方形
	{
		int x_A = i * 32 + threadIdx.x;
        int y_A = blockIdx.y * 32 + threadIdx.y;
        sharedM[ty][tx] = A[y_A * K + x_A];

        int x_B = blockIdx.x * 32 + threadIdx.x;
        int y_B = i * 32 + threadIdx.y;
        sharedN[ty][tx] = B[y_B * N + x_B];
        __syncthreads();
        //同一线程块中所有线程必须到达运行 __syncthreads()之后才可以做其余操作
		//此操作可以防止当只有部分数据拷贝到共享内存后就提前进行下列计算。


		for (int j = 0; j < 32; j++)//分块后的矩阵相乘
			Csub += sharedM[ty][j] * sharedN[j][tx];
		__syncthreads();
	}

	C[row*N + col] = Csub;
}


//主函数
int main(int argc, char **argv)
{
	int Axy = M * K;
	int Bxy = K * N;
	int Cxy = M * N;

	float *h_A, *h_B, *h_C, *deviceRef;
	
	//在CPU上分配内存
	h_A = (float*)malloc(Axy * sizeof(float)); 
	h_B = (float*)malloc(Bxy * sizeof(float));
	h_C = (float*)malloc(Cxy * sizeof(float));
    deviceRef = (float*)malloc(Cxy * sizeof(float));

	initial(h_A, Axy);
	initial(h_B, Bxy);
    // for (int i = 0; i < Cxy; ++i)h_C[i] = 0.0;
	
	//在GPU上分配显存
	float *d_A, *d_B, *d_C;
	cudaMalloc((void**)&d_A, Axy * sizeof(float));
	cudaMalloc((void**)&d_B, Bxy * sizeof(float));
	cudaMalloc((void**)&d_C, Cxy * sizeof(float));
	
	//将CPU上初始化的a b值拷贝到GPU上
	cudaMemcpy(d_A, h_A, Axy * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, Bxy * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(2, 2);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    struct timeval t1,t2;
    gettimeofday(&t1,NULL);
    multiplicateMatrix<<<grid,block>>> (d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
    gettimeofday(&t2,NULL);
    float time_use = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000;
    std::cout<<"Normal Time use: "<<time_use<<" ms"<<std::endl;

    cudaMemcpy(h_C, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);



    dim3 block2(32, 32);
    dim3 grid2((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    gettimeofday(&t1,NULL);
	matrixMultiplyShared << < grid2, block2 >> > (d_A, d_B, d_C);
    cudaDeviceSynchronize();
    gettimeofday(&t2,NULL);
    time_use = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000;
    std::cout<<"Shared Time use: "<<time_use<<" ms"<<std::endl;

    cudaMemcpy(deviceRef, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < Cxy; ++i){
    //     if(i % (512*128) == 0)std::cout <<" "<<deviceRef[i];
    // }
        
    float maxError = 0.0;
    for (int i = 0; i < Cxy; ++i)
        maxError = fmax(maxError, fabs(deviceRef[i] - h_C[i]));
    std::cout << "最大误差: " << maxError << std::endl;

	//释放GPU显存资源
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
	//释放CPU内存资源
    free(h_A);
    free(h_B);
    free(h_C);
    free(deviceRef);

return (0);
}
