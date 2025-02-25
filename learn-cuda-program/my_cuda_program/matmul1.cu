// 2025.1.18  
// 矩阵乘样例，来自于https://zhuanlan.zhihu.com/p/34587739

// 在此处，因为结构体的定义搞不清楚应该如何分配内存，不做探讨
// 核心问题在于 结构体 Matrix 和其中的元素 elements应该单独分配、拷贝

#include <iostream>
#include <cuda.h>
#include <cstdio>

//row-major存储，M(row, col) = M.elements + row*M.width + col 
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
}q

__global__ void matmul_kernel(Matrix *mat_A, Matrix *mat_B, Matrix *mat_C)
{
    float Cvalue = 0.0;
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    
    if(row % 32 == 0)printf("row = %d\n", row);

    for(int i = 0; i < mat_A->width; i++){
        Cvalue += getElement(mat_A, row, i) * getElement(mat_B, i, col); 
        
        // if(row%32 == 0 && col%1024 == 0){
        //     printf("Cvalue  = %f\n", Cvalue);
        // }
    }
    setElement(mat_C, row, col, Cvalue);
}

int main()
{
    int width = 1<<10;
    int height = 1<<10;
    Matrix A, B, C;
    int nBytes = sizeof(float) * width * height;
    
    A = (Matrix*)malloc(sizeof(Matrix));
    B = (Matrix*)malloc(sizeof(Matrix));
    C = (Matrix*)malloc(sizeof(Matrix));
    A->elements = (float*)malloc(nBytes);
    B->elements = (float*)malloc(nBytes);
    C->elements = (float*)malloc(nBytes);
    A->height = height;
    A->width = width;
    B->height = height;
    B->width = width;
    C->height = height;
    C->width = width;
    for(int i = 0; i<width*height; i++){
        A->elements[i] = 1.0;
        B->elements[i] = 1.0;
    }

    // 分配设备内存
    Matrix *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeof(Matrix));
    cudaMalloc((void **)&d_B, sizeof(Matrix));
    cudaMalloc((void **)&d_C, sizeof(Matrix));
    // cudaMalloc((void **)&d_A->elements, nBytes); //不能直接指向它
    // cudaMalloc((void **)&d_B->elements, nBytes);
    // cudaMalloc((void **)&d_C->elements, nBytes);

    float *d_A_elements, *d_B_elements, *d_C_elements;
    cudaMalloc((void **)&d_A_elements, nBytes);
    cudaMalloc((void **)&d_B_elements, nBytes);
    cudaMalloc((void **)&d_C_elements, nBytes);

    // 将主机数据拷贝到设备
    cudaMemcpy(d_A_elements, A.elements, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_elements, B.elements, nBytes, cudaMemcpyHostToDevice);

    // 设置设备上的 Matrix 结构体
    Matrix temp_A = A, temp_B = B, temp_C = C;
    temp_A.elements = d_A_elements;
    temp_B.elements = d_B_elements;
    temp_C.elements = d_C_elements;

    cudaMemcpy(d_A, &temp_A, sizeof(Matrix), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, &temp_B, sizeof(Matrix), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, &temp_C, sizeof(Matrix), cudaMemcpyHostToDevice);


    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x -1)/blockSize.x, (height + blockSize.y -1)/blockSize.y);

    matmul_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(C->elements, d_C_elements, nBytes, cudaMemcpyDeviceToHost);

    float maxError = 0.0;
    for(int i = 0; i< width * height; i++){
        maxError = fmax(maxError, fabs(C->elements[i] - width));
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
    free(A);
    free(B);
    free(C);
    free(A.elements);
    free(B.elements);
    free(C.elements);

    return 0;
}
