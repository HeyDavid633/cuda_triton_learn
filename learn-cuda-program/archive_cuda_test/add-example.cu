#include<cuda.h>
#include<iostream>
#include <sys/time.h>

// 两个向量加法kernel，grid和block均为一维
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
    float *x, *y, *z;
    x = (float*)malloc(nBytes);
    y = (float*)malloc(nBytes);
    z = (float*)malloc(nBytes);

    for(int i = 0; i<N; i++){
        x[i] = 10.0;
        y[i] = 20.0;
    }

    float *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, nBytes);
    cudaMalloc((void**)&d_y, nBytes);
    cudaMalloc((void**)&d_z, nBytes);

    cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);

    dim3 blockSize(512);
    dim3 gridSize((N + blockSize.x -1)/blockSize.x); //Attention here is blockSize ; 而不是blockDim 
    //blockSize.x = blockDim.x  但是 blockDim在Host上面不能获取
    //std::cout<<"blockSize.x:"<<blockSize.x<<"   blockDim.x:"<<blockDim.x<<std::endl;

    gettimeofday(&t1,NULL);
    add<<<gridSize, blockSize>>>(d_x, d_y, d_z, N);
    cudaDeviceSynchronize();
    gettimeofday(&t2,NULL);
    
    float time_use = (t2.tv_sec - t1.tv_sec) * 1000 + (t2.tv_usec - t1.tv_usec) / 1000;
    std::cout<<"Time use: "<<time_use<<" ms"<<std::endl;


    cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyDeviceToHost);
    
    int error = 0;
    for(int i =0; i<N; i++){
        if(z[i] - 30.0f)error++;
    }

    std::cout<<"Error num : "<<error<<std::endl;

    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    free(x);
    free(y);
    free(z);


    return 0;

}
