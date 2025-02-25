// 2025.2.25 我的TensorCore matmul
//
// 1.理解 行主序 列主序 与 访存取数据的关系
// 2.以warp为单位理解TensorCore的操作 
//
// 该版本已经可以理解并自己写出来了
//

#include <stdio.h>
#include <stdlib.h>
#include <cuda_fp16.h>
#include <sys/time.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <mma.h>
using namespace nvcuda;
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

#define WARPSIZE 32

double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
void init_matrix_half(half *init_mat, int wid, int hight, int init_num)
{
    for (int i = 0; i < wid * hight; i++)
    {
        init_mat[i] = (half)(i % init_num);
    }
}
void init_matrix_float(float *init_mat, int wid, int hight, int init_num)
{
    for (int i = 0; i < wid * hight; i++)
    {
        init_mat[i] = (float)(init_num);
    }
}
float compare(float *hostC, float *serialC, int M, int N)
{
    float error = 0;
    for (int i = 0; i < M * N; i++)
    {
        error = fmax(error, fabs(hostC[i] - serialC[i]));
    }
    // return error / (M*N);
    return error;
}
void cublasMatrix_half(half *dA, half *dB, float *dC, int M, int K, int N)
{
    cublasHandle_t handle; // cublas句柄
    cublasCreate(&handle); // 初始化句柄
    float alpha = 1.0;
    float beta = 0.0;
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
    // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, dA, K, dB, N, &beta, dC, M);
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 dB, CUDA_R_32F, N,
                 dA, CUDA_R_32F, K,
                 &beta,
                 dC, CUDA_R_32F, N,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
void matrixSerial_half_B_colmajor(half *hostA, half *hostB, float *hostC, int M, int K, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float tmp = 0;
            for (int s = 0; s < K; s++)
            {
                tmp += (float)(hostA[i * K + s]) * (float)(hostB[j * K + s]);
            }
            hostC[i * N + j] = tmp;
        }
    }
}
void matrixSerial_half_B_rowmajor(half *hostA, half *hostB, float *hostC, int M, int K, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float tmp = 0;
            for (int s = 0; s < K; s++)
            {
                tmp += (float)(hostA[i * K + s]) * (float)(hostB[s * N + j]);
            }
            hostC[i * N + j] = tmp;
        }
    }
}

//B 是列主序
__global__ void my_Simple_TesnorCoreGemm(half *dA, half *dB, float *dC, int M, int K, int N)
{
    int warpM = (threadIdx.x + blockIdx.x * blockDim.x) / WARPSIZE;
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

//B 是行主序
__global__ void my_Simple_TesnorCoreGemm2(half *dA, half *dB, float *dC, int M, int K, int N)
{
    int warpM = (threadIdx.x + blockIdx.x * blockDim.x) / WARPSIZE;
    int warpN = threadIdx.y + blockIdx.y * blockDim.y;

    int ld_a = K;
    int ld_b = N;
    int ld_c = N;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_left;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> frag_right;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc;
    wmma::fill_fragment(frag_acc, 0.0f);

    int aRow = warpM * WMMA_M;
    int bCol = warpN * WMMA_N;
    for (int i = 0; i < K; i += WMMA_K)
    {
        int aCol = i;
        int bRow = i;

        wmma::load_matrix_sync(frag_left, dA + aRow * ld_a + aCol, ld_a);
        wmma::load_matrix_sync(frag_right, dB + bRow * ld_b + bCol, ld_b);
        wmma::mma_sync(frag_acc, frag_left, frag_right, frag_acc);
    }

    wmma::store_matrix_sync(dC + aRow * ld_c + bCol, frag_acc, ld_c, wmma::mem_row_major);
}

int main()
{
    half *hostA, *hostB;
    float *hostC, *goldenC;
    int repeat_times = 100, warmup_times = 25;
    int M = 1024;
    int N = 1024;
    int K = 1024;

    hostA = (half *)malloc(M * K * sizeof(float));
    hostB = (half *)malloc(K * N * sizeof(float));
    hostC = (float *)malloc(M * N * sizeof(float));
    goldenC = (float *)malloc(M * N * sizeof(float));
    init_matrix_half(hostA, M, K, 10);
    init_matrix_half(hostB, K, N, 5);
    init_matrix_float(hostC, M, N, 0);

    half *dA, *dB;
    float *dC;
    cudaMalloc((void **)&dA, M * K * sizeof(half));
    cudaMalloc((void **)&dB, K * N * sizeof(half));
    cudaMalloc((void **)&dC, M * N * sizeof(float));
    cudaMemcpy(dA, hostA, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hostB, K * N * sizeof(half), cudaMemcpyHostToDevice);

    int num_block_x = M / (WMMA_M * 128 / WARPSIZE);
    int num_block_y = N / (WMMA_N * 4);
    dim3 block_dim(128, 4, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);

    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    my_Simple_TesnorCoreGemm<<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    // my_Simple_TesnorCoreGemm2<<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);

    // for (int i = 0; i < repeat_times + warmup_times; i++){
    //     if(i == warmup_times)cudaEventRecord(start, 0);

    //     my_Simple_TesnorCoreGemm<<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);

    // }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(hostC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    matrixSerial_half_B_colmajor(hostA, hostB, goldenC, M, K, N);
    // matrixSerial_half_B_rowmajor(hostA, hostB, goldenC, M, K, N);

    float sum_error = compare(goldenC, hostC, M, N);

    printf("M-K-N: %d-%d-%d\n", M, K, N);
    printf("Sum Error: %.4f\n", sum_error);
    printf("GPU Kernel   Time: %.4f ms\n", ker_time / repeat_times);
    printf("grid  dim: (%d, %d, %d)\n", grid_dim.x, grid_dim.y, grid_dim.z);
    printf("block dim: (%d, %d, %d)\n", block_dim.x, block_dim.y, block_dim.z);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hostA);
    free(hostB);
    free(hostC);
    free(goldenC);
    return 0;
}