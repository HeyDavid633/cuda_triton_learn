#include <iostream>
#include <cmath>
#include <cstdlib>

__global__ void matrixMultiply(const float* A, const float* B, float* C, int widthA, int widthB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;
    if (row < 3 && col < 4) {
        for (int k = 0; k < widthA; ++k) {
            sum += A[row * widthA + k] * B[col * widthB + k];
        }
        C[row * 4 + col] = sum;
    }
}

// 得到的结果应该是（3，4）, 传入参数(5, 5)
void matrixMultiplyCPU(float* A, float* B, float* C, int widthA, int widthB) {
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            float sum = 0;
            for (int k = 0; k < widthA; ++k) {
                sum += A[row * widthA + k] * B[col * widthB + k];
            }
            C[row * 4 + col] = sum;
        }
    }
}

int main() {
    // Initialize random matrices A (3x5) and B (4x5)
    float A[3][5], B[4][5];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 5; ++j)
            A[i][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 5; ++j)
            B[i][j] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

    // Allocate and copy matrices to the GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeof(float) * 3 * 5);
    cudaMalloc((void **)&d_B, sizeof(float) * 4 * 5);
    cudaMalloc((void **)&d_C, sizeof(float) * 3 * 4);
    cudaMemcpy(d_A, A, sizeof(float) * 3 * 5, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * 4 * 5, cudaMemcpyHostToDevice);

    // Call kernel
    dim3 blockSize(2, 2);
    dim3 gridSize((4 + blockSize.x - 1) / blockSize.x, (3 + blockSize.y - 1) / blockSize.y);
    matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, 5, 5);

    // Copy result back to host
    float C[3][4];
    cudaMemcpy(C, d_C, sizeof(float) * 3 * 4, cudaMemcpyDeviceToHost);

    // Verify with CPU computation
    float C_cpu[3][4];
    matrixMultiplyCPU((float*)A, (float*)B, (float*)C_cpu, 5, 5);

    // Compare results
    bool correct = true;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (fabs(C_cpu[i][j] - C[i][j]) > 1e-5) {
                correct = false;
                break;
            }
        }
    }

    if (correct)
        std::cout << "The matrix multiplication is correct." << std::endl;
    else
        std::cout << "The matrix multiplication is incorrect." << std::endl;

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);



    return 0;
}
