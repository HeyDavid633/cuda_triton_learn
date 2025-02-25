// 2025.2.25 TensorCore matmul
//
// 参考了cudasamples
// https://github.com/NVIDIA/cuda-samples/tree/master/Samples/3_CUDA_Features/cudaTensorCoreGemm
// 使用了优化手段复杂的GEMM --- 该版本不指望自己写出来，理解逻辑拿来用
//
// 对于这个例子还没有计算正确！
//
// 1.TensorCore计算GEMM
// 2.sharedMem存储小块的矩阵AB
// 3.在sharedMem中增加了skew来规避bank冲突
// 4.启动时，最大化寄存器利用率但又规避使用local mem
// 5.使用了int4来加速访存
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

#ifndef SHARED_MEMORY_LIMIT_64K
#define SHARED_MEMORY_LIMIT_64K 1
#endif

#define WARP_SIZE 32

// MMA matrix tile dimensions.
#define M WMMA_M
#define N WMMA_N
#define K WMMA_K

// GEMM configuration.
#define M_TILES 256
#define N_TILES 256
#define K_TILES 256

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

#define C_LAYOUT wmma::mem_row_major

#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
#define CHUNK_K 4
#else
#define CHUNK_K 8
#endif

#define CHUNK_LINE_BYTES (CHUNK_K * K * sizeof(half))
#define WARP_COPY_BYTES (WARP_SIZE * sizeof(int4))
#define CHUNK_COPY_LINES_PER_WARP (WARP_COPY_BYTES / CHUNK_LINE_BYTES)
#define CHUNK_COPY_LINE_LANES (WARP_SIZE / CHUNK_COPY_LINES_PER_WARP)

#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4
#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2
#define BLOCK_ROW_TILES (WARP_ROW_TILES * BLOCK_ROW_WARPS)
#define BLOCK_COL_TILES (WARP_COL_TILES * BLOCK_COL_WARPS)

#define GLOBAL_MEM_STRIDE N_GLOBAL

#define SHMEM_STRIDE (N * BLOCK_ROW_TILES)
#define SHMEM_OFFSET (N * WARP_ROW_TILES)

#define SKEW_HALF 16

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

#define checkKernelErrors(expr)                                   \
    do                                                            \
    {                                                             \
        expr;                                                     \
                                                                  \
        cudaError_t __err = cudaGetLastError();                   \
        if (__err != cudaSuccess)                                 \
        {                                                         \
            printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, \
                   cudaGetErrorString(__err));                    \
            abort();                                              \
        }                                                         \
    } while (0)

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

__global__ void compute_gemm(const half *A, const half *B, float *C, float *D, float alpha, float beta)
{
    extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];

    // Warp and lane identification.
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;

    // Offset in shared memory from which the B matrix is stored.
    const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

    // This pointer is used to access the C and D matrix tiles this warp computes.
    float *shmem_warp_tile_ptr = (float *)&shmem[0][0] +
                                 (warpId / 2) * SHMEM_STRIDE * K * 2 +
                                 (warpId % 2) * SHMEM_OFFSET;

    // This pointer is used to stream the C and D matrices block-wide tile to and
    // from shared memory.
    float *shmem_warp_stream_ptr = (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * K;

    // Each CTA slides along the 128 x 128 tiles from the top left corner of the
    // matrix to the right and down, and selects the next tile to compute. Once
    // there's no such tile, all warps in this CTA exit.
    for (unsigned int block_pos = blockIdx.x; ;block_pos += gridDim.x)
    {
        const unsigned int block_tile_i = ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
        const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

        // Stop when there are no more D matrix tiles to compute in this CTA.
        if (block_tile_i >= M_TILES){
            break;
        }

        // This warp's pointer to the C matrix data to copy memory from to shared memory.
        const size_t gmem_idx = (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
        const float *src_gmem_warp_stream_ptr = &C[gmem_idx];

        // Stream multiple C tiles to shared memory.
        #pragma unroll
        for (int i = 0; i < K; i++){
            typedef int4 copy_t;

            *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
                *((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId);
        }
        __syncthreads();

        // These fragments will accumulate the result of A and B matrix fragment
        // multiplications along the K_GLOBAL dimension.
        wmma::fragment<wmma::accumulator, M, N, K, float> c[WARP_COL_TILES][WARP_ROW_TILES];

        // Load the C matrix tiles into fragments from shared memory.
        #pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++){
            #pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++){
                const float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;
                wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
            }
        }
        __syncthreads();


        // Select what warp copies what matrix to shared memory.
        // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
        const half *warp_ptr = (warpId < 4) ? (&A[block_tile_i * M * K_GLOBAL] +
                                               M * K_GLOBAL * (warpId % 4) * 2)
                                            : (&B[block_tile_j * N * K_GLOBAL] +
                                               N * K_GLOBAL * (warpId % 4) * 2);

        // Go through the global K dimension by a fixed step at a time.
        #pragma unroll
        for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K)
        {
            // Copy slices of the A and B matrices to shared memory.
            // The first half of the warps in the CTA copy the A matrix, the rest copy the B matrix.
            size_t shmem_idx =
                warpId < (WARPS_PER_BLOCK / 2)
                    ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
                    : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

            // First half of the warp copies the first row / column of the matrix,
            // the second half of the warp copies the next.
            int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * K +
                                      (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) +
                                      (laneId % CHUNK_COPY_LINE_LANES);

            // Shift the second half of the warp to the next row / column in the shared memory.
            shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

            #pragma unroll
            for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++)
            {
                // Copy 16 bytes at once in each lane.
                *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) = *lane_ptr;

                // Advance the global memory pointer and the shared memory index.
                lane_ptr = (int4 *)((half *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
                shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }
            __syncthreads();

            // Compute a grid of C matrix tiles in each warp.
            #pragma unroll
            for (int k_step = 0; k_step < CHUNK_K; k_step++){
                wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[WARP_COL_TILES];
                wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b[WARP_ROW_TILES];

                #pragma unroll
                for (int i = 0; i < WARP_COL_TILES; i++)
                {

                    size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
                    const half *tile_ptr = &shmem[shmem_idx_a][k_step * K];

                    wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);

                    #pragma unroll
                    for (int j = 0; j < WARP_ROW_TILES; j++)
                    {
                        if (i == 0)
                        {
                            // Load the B matrix fragment once, because it is going to be
                            // reused against the other A matrix fragments.
                            size_t shmem_idx_b = shmem_idx_b_off + (WARP_ROW_TILES * N) * (warpId % 2) + (j * N);
                            const half *tile_ptr = &shmem[shmem_idx_b][k_step * K];

                            wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_HALF);
                        }

                        wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                    }
                }
            }
            __syncthreads();
        }

        // Store the D fragments to shared memory.
        #pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++)
        {
            #pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++)
            {
                #pragma unroll
                // Uniform, point-wise transformations of ALL fragment elements by ALL
                // threads in the warp are well-defined even though element indices
                // within fragment storage are not defined.
                for (int t = 0; t < c[i][j].num_elements; t++)
                    c[i][j].x[t] *= alpha;

                float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

                wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
            }
        }
        __syncthreads();

        // Now that shared memory contains all the D tiles, stream them to global memory.
        float *dst_gmem_warp_stream_ptr = &D[gmem_idx];

        #pragma unroll
        for (int i = 0; i < K; i++)
        {
            *((int4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
                *((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
        }
        __syncthreads();
    }
}

int main()
{
    half *hostA, *hostB;
    float *hostC, *goldenC;
    int GLOBAL_M = 1024;
    int GLOBAL_N = 1024;
    int GLOBAL_K = 1024;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    hostA = (half *)malloc(GLOBAL_M * GLOBAL_K * sizeof(float));
    hostB = (half *)malloc(GLOBAL_K * GLOBAL_N * sizeof(float));
    hostC = (float *)malloc(GLOBAL_M * GLOBAL_N * sizeof(float));
    goldenC = (float *)malloc(GLOBAL_M * GLOBAL_N * sizeof(float));
    init_matrix_half(hostA, GLOBAL_M, GLOBAL_K, 10);
    init_matrix_half(hostB, GLOBAL_K, GLOBAL_N, 5);
    init_matrix_float(hostC, GLOBAL_M, GLOBAL_N, 0);

    half *dA, *dB;
    float *dC;
    cudaMalloc((void **)&dA, GLOBAL_M * GLOBAL_K * sizeof(half));
    cudaMalloc((void **)&dB, GLOBAL_K * GLOBAL_N * sizeof(half));
    cudaMalloc((void **)&dC, GLOBAL_M * GLOBAL_N * sizeof(float));
    cudaMemcpy(dA, hostA, GLOBAL_M * GLOBAL_K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hostB, GLOBAL_K * GLOBAL_N * sizeof(half), cudaMemcpyHostToDevice);

    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("每个 block 最大共享内存: %.2f KB\n", deviceProp.sharedMemPerBlock / 1024.0);
    printf("每个 SM 最大的共享内存: %.2f KB\n", deviceProp.sharedMemPerMultiprocessor / 1024.0);
    printf("设备 SM 的数量: %d\n", deviceProp.multiProcessorCount);

    enum
    {
        SHMEM_SZ = MAX(
            sizeof(half) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2,
            M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N *
                (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
    };
    printf("Used shared memory size: %lu KB\n", SHMEM_SZ / 1024UL);

    // kernel
    cudaFuncSetAttribute(compute_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ);
    compute_gemm<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK, SHMEM_SZ>>>(dA, dB, dC, dC, alpha, beta);


    cudaMemcpy(hostC, dC, GLOBAL_M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // matrixSerial_half_B_colmajor(hostA, hostB, goldenC, GLOBAL_M, GLOBAL_K, GLOBAL_N);
    matrixSerial_half_B_rowmajor(hostA, hostB, goldenC, M, K, N);

    float sum_error = compare(goldenC, hostC, GLOBAL_M, GLOBAL_N);

    printf("M-K-N: %d-%d-%d\n", GLOBAL_M, GLOBAL_K, GLOBAL_N);
    printf("Sum Error: %.4f\n", sum_error);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hostA);
    free(hostB);
    free(hostC);
    free(goldenC);
    return 0;
}