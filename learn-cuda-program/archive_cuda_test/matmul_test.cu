#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matrixMultiply(const float *A, const float *B, float *C, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < height) {
        float sum = 0.0f;
        for (int i = 0; i < width; ++i) {
            sum += A[row * width + i] * B[col * width + i];  // 注意：这里假设B已经转置
        }
        C[row * height + col] = sum;
    }
}

int main() {
    const int width = 3;
    const int height = 4;
    float h_A[height * width] = {1, 2, 3, 
                                4, 5, 6, 
                                7, 8, 9,
                                10, 11, 12};
    float h_B[height * width] = {1, 2, 3, 
                                4, 5, 6, 
                                7, 8, 9,
                                10, 11, 12};
    float h_C[height * height] = {0};

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, height * width * sizeof(float));
    cudaMalloc((void **)&d_B, height * width * sizeof(float));
    cudaMalloc((void **)&d_C, height * height * sizeof(float));

    cudaMemcpy(d_A, h_A, height * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, height * width * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((height + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width, height);

    cudaMemcpy(h_C, d_C, height * height * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < height; ++j) {
            printf("%f ", h_C[i * height + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
