#include<cuda.h>
#include<iostream>
#include <sys/time.h>

__global__ void add(float* x, float * y, float* z, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        z[i] = x[i] + y[i];
    }
}

int main(int argc, char **argv)
{
    if(argc != 2){
        std::cout<<"input error ! please input a int number"<<std::endl;
    }
    int zhishu = atoi(argv[1]);
    int N = 1<<zhishu; 

    struct timeval t1,t2;

    int nBytes = N * sizeof(float);

    // 申请托管内存
    float *x, *y, *z;
    cudaMallocManaged((void**)&x, nBytes);
    cudaMallocManaged((void**)&y, nBytes);
    cudaMallocManaged((void**)&z, nBytes);

    // 初始化数据
    for (int i = 0; i < N; ++i){
        x[i] = 10.0;
        y[i] = 20.0;
    }

    // 定义kernel的执行配置
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    // 执行kernel

    gettimeofday(&t1,NULL);
    add << < gridSize, blockSize >> >(x, y, z, N);
    cudaDeviceSynchronize();
    gettimeofday(&t2,NULL);

    float time_use = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000;
    std::cout<<"Time use: "<<time_use<<" ms"<<std::endl;

    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(z[i] - 30.0));
    std::cout << "最大误差: " << maxError << std::endl;

    // 释放内存
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return 0;
}