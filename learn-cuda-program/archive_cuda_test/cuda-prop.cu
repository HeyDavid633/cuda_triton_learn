#include <iostream>
#include <cuda.h>
#include <cstdio>
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

__device__ void print_arch(){
  const char my_compile_time_arch[] = STR(__CUDA_ARCH__);
  printf("__CUDA_ARCH__: %s\n", my_compile_time_arch);
}
//nvcc cuda-prop.cu -arch=sm_89 -o cuda_prop否则 __CUDA_ARCH__ = 520

__global__ void example()
{
   print_arch();
}

int main(void){
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
    std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个SM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;

    example<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}

