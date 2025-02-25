// 2025.1.19  
// 矩阵乘样例，来自于https://zhuanlan.zhihu.com/p/34587739
// 改进于 matmul1 
// 分别分配和拷贝了结构体Matrix的内存

#include <iostream>
#include <cuda.h>
#include <cstdio>

// row-major存储，M(row, col) = M.elements + row*M.width + col
struct Matrix
{
    int width;
    int height;
    float *elements;
};

__device__ float getElement(Matrix *mat_A, int row, int col)
{
    return mat_A->elements[row * mat_A->width + col];
}

__device__ void setElement(Matrix *mat_A, int row, int col, float value)
{
    mat_A->elements[row * mat_A->width + col] = value;
}

__global__ void matmul_kernel(Matrix *mat_A, Matrix *mat_B, Matrix *mat_C)
{
    float Cvalue = 0.0;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row < mat_A->height && col < mat_B->width) {
        for (int i = 0; i < mat_A->width; i++) {
            Cvalue += getElement(mat_A, row, i) * getElement(mat_B, i, col);
        }
        setElement(mat_C, row, col, Cvalue);
    }
}

int main()
{
    int width = 1 << 10;  // 1024
    int height = 1 << 10; // 1024
    size_t nBytes = sizeof(float) * width * height;

    // 分配主机内存=-
    Matrix h_A, h_B, h_C;
    h_A.width = width;
    h_A.height = height;
    h_B.width = width;
    h_B.height = height;
    h_C.width = width;
    h_C.height = height;

    h_A.elements = (float *)malloc(nBytes);
    h_B.elements = (float *)malloc(nBytes);
    h_C.elements = (float *)malloc(nBytes);

    // 初始化主机数据
    for (int i = 0; i < width * height; i++) {
        h_A.elements[i] = 1.0;
        h_B.elements[i] = 1.0;
    }

    // 分配设备内存
    Matrix *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeof(Matrix));
    cudaMalloc((void **)&d_B, sizeof(Matrix));
    cudaMalloc((void **)&d_C, sizeof(Matrix));

    float *d_A_elements, *d_B_elements, *d_C_elements;
    cudaMalloc((void **)&d_A_elements, nBytes);
    cudaMalloc((void **)&d_B_elements, nBytes);
    cudaMalloc((void **)&d_C_elements, nBytes);

    // 将主机数据拷贝到设备
    cudaMemcpy(d_A_elements, h_A.elements, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_elements, h_B.elements, nBytes, cudaMemcpyHostToDevice);

    // 设置设备上的 Matrix 结构体
    Matrix temp_A = h_A, temp_B = h_B, temp_C = h_C;
    temp_A.elements = d_A_elements;
    temp_B.elements = d_B_elements;
    temp_C.elements = d_C_elements;

    cudaMemcpy(d_A, &temp_A, sizeof(Matrix), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, &temp_B, sizeof(Matrix), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, &temp_C, sizeof(Matrix), cudaMemcpyHostToDevice);

    // 定义 kernel 的执行配置
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // 执行 kernel
    matmul_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    // 将设备数据拷贝回主机
    cudaMemcpy(h_C.elements, d_C_elements, nBytes, cudaMemcpyDeviceToHost);

    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < width * height; i++) {
        maxError = fmax(maxError, fabs(h_C.elements[i] - width));
    }
    std::cout << "最大误差: " << maxError << std::endl;

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_elements);
    cudaFree(d_B_elements);
    cudaFree(d_C_elements);

    // 释放主机内存
    free(h_A.elements);
    free(h_B.elements);
    free(h_C.elements);

    return 0;
}