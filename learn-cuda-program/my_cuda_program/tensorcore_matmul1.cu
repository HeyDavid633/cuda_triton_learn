// 2025.02.06 
// tensorcore版本的GEMM 初步使用
// 参考 https://zhuanlan.zhihu.com/p/671312675
// 

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>
#include <mma.h>
using namespace nvcuda;

#define warpSize 32

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void wmma_example(half *a, half *b, float *c, 
                             int M, int N, int K, 
                             float alpha, float beta) 
{
    // Leading dimensions. Packed with no transpositions.
    int lda = M;
    int ldb = K;
    int ldc = M;

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // set 0 in accumulator fragment
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over the K-dimension
    // 每个warp计算输出矩阵的一个tile; MxN的输出tile
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;
        
        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

            // Perform the matrix multiplication 执行的事FMA
            // 原地累积，因此第一个和最后一个参数都是我们之前初始化为零的累加器fragment
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag); 
        }
    }

    // Load in current value of c, scale by beta, and add to result scaled by alpha
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    
    if (cRow < M && cCol < N) {
        wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);
        
        for(int i=0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }
        // Store the output
        wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
    }
}

__global__ void my_Simple_TesnorCoreGemm(half *dA, half *dB, float *dC, int M, int K, int N)
{
    int warpM = (threadIdx.x + blockIdx.x * blockDim.x) / warpSize;
    int warpN = threadIdx.y + blockIdx.y * blockDim.y;

    int ld_a = K;
    int ld_b = K;
    int ld_c = N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_left;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> frag_right;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc;
    wmma::fill_fragment(frag_acc, 0.0f);

    int aRow = warpM * WMMA_M;
    int bCol = warpN * WMMA_N;
    for (int i = 0; i < K; i += WMMA_K)
    {
        int aCol = i;
        int bRow = i;

        wmma::load_matrix_sync(frag_left, dA + aRow * ld_a + aCol, ld_a);
        wmma::load_matrix_sync(frag_right, dB + bCol * ld_b + bRow, ld_b);
        wmma::mma_sync(frag_acc, frag_left, frag_right, frag_acc);
    }

    wmma::store_matrix_sync(dC + aRow * ld_c + bCol, frag_acc, ld_c, wmma::mem_row_major);
}


void initializeMatrix(half* matrix, int size, half value) {
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = value;
    }
}

void initializeMatrix(float* matrix, int size, float value) {
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = value;
    }
}

void printMatrix(const float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}

bool verifyResult(float* C, float* expectedC, int M, int N) {
    for (int i = 0; i < M * N; ++i) {
        if (fabs(C[i] - expectedC[i]) > 1e-5) {
            std::cout << "Mismatch at index " << i << ": " << C[i] << " != " << expectedC[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    const int M = 256;
    const int N = 256;
    const int K = 256;
    const int sizeA = M * K;
    const int sizeB = K * N;
    const int sizeC = M * N;

    // Allocate host memory
    half* h_A = new half[sizeA];
    half* h_B = new half[sizeB];
    float* h_C = new float[sizeC];
    float* h_expectedC = new float[sizeC];

    // Initialize matrices
    initializeMatrix(h_A, M, __float2half(2.0f));
    initializeMatrix(h_B, K, __float2half(2.0f));
    initializeMatrix(h_C, M, 0.0f);
    initializeMatrix(h_expectedC, M, 0.0f);

    // Compute expected result on host
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            for (int k = 0; k < K; ++k) {
                h_expectedC[m * N + n] += static_cast<float>(h_A[m * K + k]) * static_cast<float>(h_B[k * N + n]);
            }
        }
    }

    // Allocate device memory
    half* d_A;
    half* d_B;
    float* d_C;
    cudaMalloc(&d_A, sizeA * sizeof(half));
    cudaMalloc(&d_B, sizeB * sizeof(half));
    cudaMalloc(&d_C, sizeC * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, sizeA * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeC * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 blockSize(warpSize, 1);
    dim3 gridSize((M + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);

    // Launch kernel
    float alpha = 1.0f;
    float beta = 0.0f;
    // wmma_example<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    my_Simple_TesnorCoreGemm<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, K, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify result
    bool isCorrect = verifyResult(h_C, h_expectedC, M, N);
    if (isCorrect) {
        std::cout << "The computation is correct." << std::endl;
    } else {
        std::cout << "The computation is incorrect." << std::endl;
    }

    // // Print result matrix
    // std::cout << "Result Matrix:" << std::endl;
    // printMatrix(h_C, M, N);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_expectedC;

    return 0;
}