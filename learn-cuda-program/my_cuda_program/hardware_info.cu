// 2025.1.17
//
// 复习CUDA 获取硬件信息来看看
//
// nvcc cuda-prop.cu -arch=sm_89 -o cuda_prop 否则 __CUDA_ARCH__ = 520
#include <iostream>
#include <cuda.h>
#include <cstdio>
#include <stdio.h>

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

void checkCudaError(cudaError_t err, const char *message)
{
    if (err != cudaSuccess)
    {
        printf("%s: %s\n", message, cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void print_arch()
{
    const char my_compile_time_arch[] = STR(__CUDA_ARCH__);
    printf("__CUDA_ARCH__: %s\n", my_compile_time_arch);
}

int main(void)
{
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);

    // 输出 GPU 设备信息
    std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << ">>>  SM的数量: " << devProp.multiProcessorCount << std::endl;
    std::cout << "每个 block 的最大共享内存大小: " << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个 SM 的最大共享内存大小: " << devProp.sharedMemPerMultiprocessor / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块的最大线程数:   " << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "每个SM的最大线程数:      " << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个SM的最大线程束数:    " << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;

    // 获取 L2 Cache 大小（单位为字节）
    size_t l2CacheSize = devProp.l2CacheSize;
    std::cout << ">>>  L2 Cache 大小: " << l2CacheSize / (1024.0 * 1024.0) << " MB" << std::endl;

    // 计算 L1 Cache 大小（单位为 MB）
    // L1 Cache 大小 = SM 数量 * 每个 SM 的 Shared Memory 大小
    size_t l1CacheSize = devProp.multiProcessorCount * devProp.sharedMemPerMultiprocessor;
    std::cout << ">>>  L1 Cache 大小: " << l1CacheSize / (1024.0 * 1024.0) << " MB" << std::endl;

    cudaSharedMemConfig config;
    cudaError_t err = cudaDeviceGetSharedMemConfig(&config);
    if (err == cudaSuccess)
    {
        printf("Current shared memory bank size: ");
        switch (config)
        {
        case cudaSharedMemBankSizeFourByte:
            printf("4 bytes\n");
            break;
        case cudaSharedMemBankSizeEightByte:
            printf("8 bytes\n");
            break;
        default:
            printf("Unknown\n");
            break;
        }
    }

    // 配置片上的共享内存
    cudaFuncCache cacheConfig;
    err = cudaDeviceSetCacheConfig(cacheConfig);
    std::cout << "Shared mem & L1 Cache Config : " << cacheConfig << std::endl;

    // 调用 CUDA Kernel
    print_arch<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0;
}