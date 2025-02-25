// 2025.1.18  
// 初步上手样例1，来自于https://zhuanlan.zhihu.com/p/34587739

#include <iostream>
#include <cuda.h>
#include <cstdio>

__global__ void vector_add(float *x, float *y, float *z, int n)
{
    int index  = threadIdx.x + blockIdx.x * blockDim.x;
    
    // int stride = blockDim.x * gridDim.x;
    // for(int i=index; i<n; i+=stride){
    //     z[i] = x[i] + y[i];
        
    //     if(i % 256 == 0){
    //         printf("stride = %d | n = %d\n", stride,n);
    //         printf("i = %d: z[i] = %f\n", i, z[i]);
    //     }
    // }
    z[index] = x[index] + y[index];
}

int main()
{
    int N = 1<<12;
    int nBytes = N * sizeof(float);
    
    // 在host上开内存
    float *x, *y, *z; 
    x = (float*)malloc(nBytes);
    y = (float*)malloc(nBytes);
    z = (float*)malloc(nBytes);

    for(int i=0; i<N; i++){
        x[i] = 10.0;
        y[i] = 20.0;
    }

    // 在device上开内存
    float *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, nBytes);
    cudaMalloc((void**)&d_y, nBytes);
    cudaMalloc((void**)&d_z, nBytes);

    // 拷贝数据到device
    cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);
    
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x -1)/blockSize.x);

    vector_add<< < gridSize, blockSize >> >(d_x, d_y, d_z, N);
    cudaDeviceSynchronize();

    cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyDeviceToHost);

    float maxError = 0.0;
    for(int i = 0; i< N; i++){
        maxError = fmax(maxError, fabs(z[i] - 30.0));
    }
    std::cout << "最大误差: " << maxError << std::endl;

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    free(x);
    free(y);
    free(z);

    return 0;
}