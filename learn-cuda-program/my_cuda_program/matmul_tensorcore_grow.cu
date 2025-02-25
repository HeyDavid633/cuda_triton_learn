// 2025.2.14 Matmul TensorCore
// 
// 熟悉 TensorCore 的矩阵乘计算
// https://github.com/xgqdut2016/hpc_project/tree/main/cuda/matrix
// https://zhuanlan.zhihu.com/p/671312675
//
// V1 朴素实现的TC matmul
// V2 继承朴素实现的思路，但从shred Mem来取
// V3 blockIdx.x处理B的列，blockIdx.y处理A的行 ，没用shared mem
// V4 block版TC matmul
// V5 利用了shared Mem 的 block版TC matmul
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <mma.h>
using namespace nvcuda;
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 8;
const int BLOCK_DIM_x = 16;
const int BLOCK_DIM_y = 16;
const int warpSize = 32;
const int warpNum = BLOCK_DIM_x * BLOCK_DIM_y / warpSize;
const int warpX = (warpNum == 1 ? 1 : 2);
const int warpY = warpNum / warpX;

double get_walltime()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)(tp.tv_sec + tp.tv_usec * 1e-6);
}
void init_matrix(float *init_mat, int wid, int hight, int init_num){
    for(int i = 0; i<wid*hight; i++){
        init_mat[i] = float(i % init_num);
    }
}
void matrixSerial(float *hostA, float *hostB, float *hostC, int M, int K, int N)
{
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            float tmp = 0;
            for (int s = 0; s < K; s++){
                tmp += hostA[i * K + s] * hostB[s * N + j];
            }
            hostC[i * N + j] = tmp;
        }
    }
}
float compare(float *hostC, float *serialC, int M, int N)
{
    float error = 0;
    for (int i = 0; i < M * N; i++){
        error = fmax(error, fabs(hostC[i] - serialC[i]));
    }
    // return error / (M*N);
    return error;
}
void cublasMatrix(float *dA, float *dB, float *dC, int M, int K, int N)
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


// TensorCore matmul 最朴素实现 V1
__global__ void V1_row_wmma_matmul_kernel(float *dA, float *dB, float *dC, int M, int K, int N)
{
    //行主序 给leading dim 为列数量
    int lda = K; //确定主序[M, K]   行主序(x,y)=x * K + y，列主序(x,y) = y * M + x
    int ldb = N; 
    int ldc = N;

    int index_A = blockIdx.x * warpX * WMMA_M;
    int index_B = blockIdx.y * warpY * WMMA_N;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warpId = tid / warpSize;
    int warpIdx = warpId % warpX; //0, 1
    int warpIdy = warpId / warpX; //0, 1, 2, 3

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> left_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> right_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int aRow = index_A + warpIdx * WMMA_M;
    int bCol = index_B + warpIdy * WMMA_N;
    int width = (K + WMMA_K - 1) /WMMA_K;
    for(int i = 0; i<width; i++)
    {
        int aCol = i * WMMA_K;
        int bRow = i * WMMA_K;

        // 具体读取子矩阵的元素到 frag
        wmma::load_matrix_sync(left_frag, dA + aRow * lda + aCol, lda);
        wmma::load_matrix_sync(right_frag, dB + bRow * ldb + bCol, ldb);
        // 子矩阵乘
        wmma::mma_sync(c_frag, left_frag, right_frag, c_frag);
    }

    //每次算得的都是 WMMA_M * WMMA_N
    wmma::store_matrix_sync(dC + aRow * ldc + bCol, c_frag, ldc, wmma::mem_row_major);
}

// V2 继承V1方式但是放进shared Mem
__global__ void V2_row_wmma_matmul_kernel_shared(float *dA, float *dB, float *dC, int M, int K, int N)
{
    int ldc = N;
    __shared__ float Shared_A[warpNum * WMMA_M * WMMA_K];
    __shared__ float Shared_B[warpNum * WMMA_K * WMMA_N];
    int indA = blockIdx.x * warpX * WMMA_M;
    int indB = blockIdx.y * warpY * WMMA_N;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warpId = tid / warpSize;
    int warpIdx = warpId % warpX;
    int warpIdy = warpId / warpX;

    int laneId = tid % warpSize;
    int smem_a_m = laneId % WMMA_M;
    int smem_a_k = laneId / WMMA_M;
    int stride_a = warpSize / WMMA_M;

    int smem_b_k = laneId % WMMA_K;
    int smem_b_n = laneId / WMMA_K;
    int stride_b = warpSize / WMMA_K;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> left_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> right_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    int aRow = indA + warpIdx * WMMA_M;
    int bCol = indB + warpIdy * WMMA_N;
    int width = (K + WMMA_K - 1) / WMMA_K;
    for(int i = 0; i< width; i++){
        int aCol = i * WMMA_K;
        int bRow = i * WMMA_K;

        for (int id = smem_a_k; id < WMMA_K; id += stride_a){
            Shared_A[warpId * WMMA_M * WMMA_K + smem_a_m * WMMA_K + id] = dA[(aRow + smem_a_m) * K + aCol + id];
        }
        for (int id = smem_b_n; id < WMMA_N; id += stride_b){
            Shared_B[warpId * WMMA_N * WMMA_K + smem_b_k * WMMA_N + id] = dB[(bRow + smem_b_k) * N + bCol + id];
        }
        __syncthreads();

        wmma::load_matrix_sync(left_frag, Shared_A + warpId * WMMA_M * WMMA_K, WMMA_K);
        wmma::load_matrix_sync(right_frag, Shared_B + warpId * WMMA_K * WMMA_N, WMMA_N);
        wmma::mma_sync(c_frag, left_frag, right_frag, c_frag);
    }

    wmma::store_matrix_sync(dC + aRow * ldc + bCol, c_frag, ldc, wmma::mem_row_major);
}

// V3 blockIdx.x处理B的列，blockIdx.y处理A的行 ，没用shared mem
__global__ void V3_row_wmma_kerV2(float *dA, float *dB, float *dC, int M, int K, int N)
{
    int lda = K; // A=[M,K],索引(x,y) = x * K + y，列优先原则索引(x,y) = y * M + x
    int ldb = N;
    int ldc = N;

    int indB = blockIdx.x * warpX * WMMA_M;
    int indA = blockIdx.y * warpY * WMMA_N;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warpId = tid / warpSize;
    int warpIdx = warpId % warpX;
    int warpIdy = warpId / warpX;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> left_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> right_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);
    int aRow = indA + warpIdy * WMMA_N;
    int bCol = indB + warpIdx * WMMA_M;
    int width = (K + WMMA_K - 1) / WMMA_K;
    for (int i = 0; i < width; i++)
    {
        int aCol = i * WMMA_K;
        int bRow = i * WMMA_K;
        // 读取A,B矩阵里面子矩阵的元素
        wmma::load_matrix_sync(left_frag, dA + aRow * lda + aCol, lda);
        wmma::load_matrix_sync(right_frag, dB + bRow * ldb + bCol, ldb);
        // 子矩阵做乘法
        wmma::mma_sync(c_frag, left_frag, right_frag, c_frag);
    }
    int cRow = aRow;
    int cCol = bCol;
 
    wmma::store_matrix_sync(dC + cRow * ldc + cCol, c_frag, ldc, wmma::mem_row_major);
}

// V4 wmma Block
__device__ void wmmaBlock(float *dA, float *dB, float *dC, int indA, int indB, int M, int K, int N)
{
    int lda = K; // A=[M,K],索引(x,y) = x * K + y，列优先原则索引(x,y) = y * M + x
    int ldb = N;
    int ldc = N;

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warpId = tid / warpSize;
    int warpIdx = warpId % warpX;
    int warpIdy = warpId / warpX;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> left_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> right_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);
    int bCol = indB + warpIdx * WMMA_M;
    int aRow = indA + warpIdy * WMMA_N;
    int width = (K + WMMA_K - 1) / WMMA_K;
    for (int i = 0; i < width; i++)
    {
        int aCol = i * WMMA_K;
        int bRow = i * WMMA_K;
       
        wmma::load_matrix_sync(left_frag, dA + aRow * lda + aCol, lda);
        wmma::load_matrix_sync(right_frag, dB + bRow * ldb + bCol, ldb);
        // 子矩阵做乘法
        wmma::mma_sync(c_frag, left_frag, right_frag, c_frag);
    }
    int cRow = aRow;
    int cCol = bCol;
  
    wmma::store_matrix_sync(dC + cRow * ldc + cCol, c_frag, ldc, wmma::mem_row_major);

}
__global__ void V4_wmmaRowMatmul(float *dA, float *dB, float *dC, int M, int K, int N)
{
    int indB = blockIdx.x * warpX * WMMA_M;
    int indA = blockIdx.y * warpY * WMMA_N;
    wmmaBlock(dA, dB, dC, indA, indB, M, K, N);
}

// V5 继承V4但用了 shared Mem
__device__ void wmmashareBlock(float *dA, float *dB, float *shareC, int indA, int indB, int M, int K, int N)
{
    int lda = K;              // 一个线程块内是[warpY * WMMA_N, K]
    int ldb = N;              // 一个线程块内是[K, warpX * WMMA_M]
    int ldc = warpX * WMMA_M; // 一个线程块内是[warpY * WMMA_N, warpX * WMMA_M]

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warpId = tid / warpSize;
    int warpIdx = warpId % warpX;
    int warpIdy = warpId / warpX;

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> left_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> right_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);
    int bCol = indB + warpIdx * WMMA_M;
    int aRow = indA + warpIdy * WMMA_N;
    int width = (K + WMMA_K - 1) / WMMA_K;
    for (int i = 0; i < width; i++)
    {
        int aCol = i * WMMA_K;
        int bRow = i * WMMA_K;
        
        wmma::load_matrix_sync(left_frag, dA + aRow * lda + aCol, lda);
        wmma::load_matrix_sync(right_frag, dB + bRow * ldb + bCol, ldb);
        // 子矩阵做乘法
        wmma::mma_sync(c_frag, left_frag, right_frag, c_frag);
        
    }
    int cRow = warpIdy * WMMA_N;
    int cCol = warpIdx * WMMA_M;
    wmma::store_matrix_sync(shareC + cRow * ldc + cCol, c_frag, ldc, wmma::mem_row_major);
}
__global__ void V5_wmmashareRowMatmul(float *dA, float *dB, float *dC, int M, int K, int N)
{
    int indB = blockIdx.x * warpX * WMMA_M;
    int indA = blockIdx.y * warpY * WMMA_N;
    __shared__ float shareC[warpY * WMMA_N * warpX * WMMA_M]; //[warpY * WMMA_N , warpX * WMMA_M]
    wmmashareBlock(dA, dB, shareC, indA, indB, M, K, N);
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warpId = tid / warpSize;
    int warpIdx = warpId % warpX;
    int warpIdy = warpId / warpX;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    int cRowLocal = warpIdy * WMMA_N;
    int cColLocal = warpIdx * WMMA_M;
    int ldcLocal = warpX * WMMA_M;
    wmma::load_matrix_sync(c_frag, shareC + cRowLocal * ldcLocal + cColLocal, ldcLocal, wmma::mem_row_major);
    int cColGlobal = indB + warpIdx * WMMA_M;
    int cRowGlobal = indA + warpIdy * WMMA_N;
    int ldcGlobal = N;
    wmma::store_matrix_sync(dC + cRowGlobal * ldcGlobal + cColGlobal, c_frag, ldcGlobal, wmma::mem_row_major);
}

int main()
{
    float *hostA, *hostB, *hostC, *goldenC;
    int repeat_times = 100, warmup_times = 25;
    int M = 8192;
    int N = 8192;
    int K = 8192;

    hostA = (float *)malloc(M * K * sizeof(float));
    hostB = (float *)malloc(K * N * sizeof(float));
    hostC = (float *)malloc(M * N * sizeof(float));
    goldenC = (float *)malloc(M * N * sizeof(float));
    init_matrix(hostA, M, K, 10);
    init_matrix(hostB, K, N, 5);

    float *dA, *dB, *dC;
    cudaMalloc((void **)&dA, M * K * sizeof(float));
    cudaMalloc((void **)&dB, K * N * sizeof(float));
    cudaMalloc((void **)&dC, M * N * sizeof(float));
    cudaMemcpy(dA, hostA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hostB, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // matrixSerial(hostA, hostB, goldenC, M, K, N);
    
    int num_block_x = (M + WMMA_M * warpX - 1) / (WMMA_M * warpX);
    int num_block_y = (N + WMMA_N * warpY - 1) / (WMMA_N * warpY);
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_block_x, num_block_y, 1);

    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < repeat_times + warmup_times; i++){
        if(i == warmup_times)cudaEventRecord(start, 0);
        // cublasMatrix(dA, dB, dC, M, K, N);
        // V1_row_wmma_matmul_kernel<<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
        // V2_row_wmma_matmul_kernel_shared<<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
        // V3_row_wmma_kerV2<<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
        // V4_wmmaRowMatmul<<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
        V5_wmmashareRowMatmul<<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);

    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(hostC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);    
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