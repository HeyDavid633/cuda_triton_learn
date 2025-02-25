// 2025.2.24 cudasamples cudaTensorCoreGemm

// D = alpha * A * B + beta * C
// C D [M_GLOBAL, N_GLOBAL]
// A   [M_GLOBAL, K_GLOBAL]（行主序)
// B   [K_GLOBAL, N_GLOBAL]（列主序）
// 每个 CTA 负责计算一个 128 x 128 的tile，并在每次迭代中处理一个这样的块
// 每个 CTA 包含八个 warp，每个 warp 计算八个 16 x 16 subtiles, 这些subtiles被组织成一个 2 x 4 的二维数组
// warp 使用 wmma::mma_sync 操作来计算子块，通过遍历 A 和 B 矩阵的 K_GLOBAL 维来累积中间结果

// 该算法使用了一些简单的优化：
// - CTA 将 C 矩阵的 128 x 128 块从全局内存复制到共享内存。完成后，每个 warp 从共享内存加载 C 矩阵片段，
//   从而避免了随机全局内存访问。
// - 在每次内部迭代时，CTA 将 A 和 B 矩阵的一部分从全局内存复制到共享内存。之后，CTA 中的所有 warp 重用共享内存中的 A 和 B 数据，
//   从而减少了从全局内存的数据拷贝次数。
// - A 和 B 矩阵的部分存储在共享内存中时增加了额外的填充（skew），以减少共享内存访问bank conflicts的数量。
//   （有关详细解释，请参见 SKEW_HALF 宏定义附近的说明。）
// - 当 CTA 完成计算结果矩阵的块时，每个 warp 将其子块存储到共享内存中。然后，CTA 将共享内存内容复制到全局内存中，
//   再次避免了冗余的随机全局内存访问。
// - 注意，CTA 块大小的选择是为了最大化 GPU 寄存器利用率，但又足够谨慎以避免使用local memory。

#include <assert.h>
#include <cuda.h>
#include <mma.h>
#include <stdio.h>

#include <helper_cuda.h>
#include <helper_functions.h>

#ifndef CPU_DEBUG
// Set this to 1 to verify the correctness of the GPU-computed matrix.
#define CPU_DEBUG 0
#endif

#ifndef SHARED_MEMORY_LIMIT_64K
// 如果shared mem的大小大于64KB，则设置这个值是 0
#define SHARED_MEMORY_LIMIT_64K 1
#endif

// GPU configuration.
#define WARP_SIZE 32

// MMA matrix tile dimensions.
#define M 16
#define N 16
#define K 16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// GEMM configuration.
#define M_TILES 256
#define N_TILES 256
#define K_TILES 256

#define M_GLOBAL (M * M_TILES)
#define N_GLOBAL (N * N_TILES)
#define K_GLOBAL (K * K_TILES)

#define C_LAYOUT wmma::mem_row_major

// Implementation constants.
#define WARPS_PER_BLOCK 8
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

#if SHARED_MEMORY_LIMIT_64K
// 由于只有 64 Kb 的共享内存可用，我们可以容纳 A 和 B 矩阵数据的两个 8-tile chunks，
// 每个块大小为 16 * 16 * 8 * 8 * 2(half 2Byte) = 32 Kb（所以每个 CTA 中有2个32KB的half元素的数组
// 但是我们不能忽略总共 8 Kb skew 开销，如果没有skew，性能将受到严重影响。
// 因此，我们选择将块大小减半，即减少缓存在共享内存中的 A 和 B 矩阵数据量。
// 相应地，这会使全局 K 维上的外层迭代次数加倍，但这对性能的影响很小。
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

// 这个宏用来为 矩阵A偏移行，为矩阵B偏移列，以规避 bank conflicts
// 在进行操作 wmma::mma_sync 前，warp 需要用 wmma::load_matrix_sync 载入数据
// warp中的每个lane需要从不同矩阵行或列的读取1个或多个元素 --- 可能导致bank冲突
// 解决bank冲突：用一点的Bytes来偏移每行/列
// 最小的偏移量选择：16 个 2 Bytes 的half元素 --- 使得256bits对齐(wmma::load_matrix_sync)对齐
#define SKEW_HALF 16

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

using namespace nvcuda;

__host__ void init_host_matrices(half *a, half *b, float *c)
{
    for (int i = 0; i < M_GLOBAL; i++)
    {
        for (int j = 0; j < K_GLOBAL; j++)
        {
            a[i * K_GLOBAL + j] = (half)(rand() % 3);
        }
    }

    for (int i = 0; i < N_GLOBAL; i++)
    {
        for (int j = 0; j < K_GLOBAL; j++)
        {
            b[i * K_GLOBAL + j] = (half)(rand() % 3);
        }
    }

    for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++)
    {
        c[t] = static_cast<float>(rand() % 3);
    }
}

__host__ void matMultiplyOnHost(half *A, half *B, float *C, float alpha,
                                float beta, int numARows, int numAColumns,
                                int numBRows, int numBColumns, int numCRows,
                                int numCColumns)
{
    for (int i = 0; i < numCRows; i++)
    {
        for (int j = 0; j < numCColumns; j++)
        {
            float temp = 0.0;

            for (int k = 0; k < numAColumns; k++)
            {
                temp += (float)A[i * numAColumns + k] * (float)B[j * numBRows + k];
            }

            C[i * numCColumns + j] = temp * alpha + beta * C[i * numCColumns + j];
        }
    }
}


__global__ void compute_gemm(const half *A, const half *B, const float *C,
                             float *D, float alpha, float beta)
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
    float *shmem_warp_stream_ptr =
        (float *)&shmem[0][0] + warpId * SHMEM_STRIDE * K;

    // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
    // each tile computation. Technically this is not generally correct (may
    // result in a loss of precision). Zero still needs to be specially handled
    // though.
    beta /= alpha;

    // Each CTA slides along the 128 x 128 tiles from the top left corner of the
    // matrix to the right and down, and selects the next tile to compute. Once
    // there's no such tile, all warps in this CTA exit.

    for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x)
    {
        const unsigned int block_tile_i =
            ((block_pos * BLOCK_ROW_TILES) / N_TILES) * (BLOCK_COL_TILES);
        const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

        // Stop when there are no more D matrix tiles to compute in this CTA.
        if (block_tile_i >= M_TILES){
            break;
        }

        // This warp's pointer to the C matrix data to copy memory from to shared
        // memory.
        const size_t gmem_idx =
            (block_tile_i + warpId) * M * GLOBAL_MEM_STRIDE + block_tile_j * N;
        const float *src_gmem_warp_stream_ptr = &C[gmem_idx];

        // Stream multiple C tiles to shared memory.
        #pragma unroll
        for (int i = 0; i < K; i++)
        {
            typedef int4 copy_t;

            *((copy_t *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
                *((copy_t *)(src_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) +
                  laneId);
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

        // Scale the C matrix.
        #pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++){
            #pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++){
                #pragma unroll
                for (int t = 0; t < c[i][j].num_elements; t++){
                    c[i][j].x[t] *= beta;
                }
            }
        }

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
            // The first half of the warps in the CTA copy the A matrix, the rest copy
            // the B matrix.
            size_t shmem_idx =
                warpId < (WARPS_PER_BLOCK / 2)
                    ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
                    : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

            // First half of the warp copies the first row / column of the matrix,
            // the second half of the warp copies the next.
            int4 *lane_ptr = (int4 *)(warp_ptr + tile_k * K +
                                      (laneId / CHUNK_COPY_LINE_LANES) * K_GLOBAL) +
                             (laneId % CHUNK_COPY_LINE_LANES);

            // Shift the second half of the warp to the next row / column in the
            // shared memory.
            shmem_idx += laneId / CHUNK_COPY_LINE_LANES;

            #pragma unroll
            for (int i = 0; i < ((WARP_SIZE / 2) / CHUNK_COPY_LINES_PER_WARP) * 2; i++){
                // Copy 16 bytes at once in each lane.
                *((int4 *)&shmem[shmem_idx][0] + (laneId % CHUNK_COPY_LINE_LANES)) = *lane_ptr;

                // Advance the global memory pointer and the shared memory index.
                lane_ptr = (int4 *)((half *)lane_ptr + K_GLOBAL * CHUNK_COPY_LINES_PER_WARP);
                shmem_idx += CHUNK_COPY_LINES_PER_WARP;
            }
            __syncthreads();

            // Compute a grid of C matrix tiles in each warp.
            #pragma unroll
            for (int k_step = 0; k_step < CHUNK_K; k_step++)
            {
                wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[WARP_COL_TILES];
                wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b[WARP_ROW_TILES];

                #pragma unroll
                for (int i = 0; i < WARP_COL_TILES; i++){

                    size_t shmem_idx_a = (warpId / 2) * M * 2 + (i * M);
                    const half *tile_ptr = &shmem[shmem_idx_a][k_step * K];

                    wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);

                    #pragma unroll
                    for (int j = 0; j < WARP_ROW_TILES; j++){
                        if (i == 0){
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
        for (int i = 0; i < WARP_COL_TILES; i++){
            #pragma unroll
            for (int j = 0; j < WARP_ROW_TILES; j++){
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
        for (int i = 0; i < K; i++){
            *((int4 *)(dst_gmem_warp_stream_ptr + GLOBAL_MEM_STRIDE * i) + laneId) =
                *((int4 *)(shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
        }
        __syncthreads();
    }
}

// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is a less performant version of the compute_gemm kernel. It is
// designed for
//       demonstration purposes only to show the CUDA WMMA API use without
//       relying on availability of the shared memory.
__global__ void simple_wmma_gemm(half *a, half *b, float *c, float *d, int m_ld,
                                 int n_ld, int k_ld, float alpha, float beta)
{
    // Leading dimensions. Packed with no transpositions.
    int lda = k_ld;
    int ldb = k_ld;
    int ldc = n_ld;

    // Tile using a 2D grid --- tile内通过 (warpM, warpN) 确定是哪个warp
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;  //0-3
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);             //0-3

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over k
    for (int i = 0; i < k_ld; i += WMMA_K)
    {
        int aCol = i;
        int aRow = warpM * WMMA_M;
        int bCol = warpN * N;
        int bRow = i;

        // Bounds checking
        if (aRow < m_ld && aCol < k_ld && bRow < k_ld && bCol < n_ld)
        {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aCol + aRow * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load in the current value of c, scale it by beta, and add this our result
    // scaled by alpha
    int cCol = warpN * WMMA_N;
    int cRow = warpM * WMMA_M;

    if (cRow < m_ld && cCol < n_ld)
    {
        wmma::load_matrix_sync(c_frag, c + cCol + cRow * ldc, ldc,
                               wmma::mem_row_major);

        for (int i = 0; i < c_frag.num_elements; i++)
        {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(d + cCol + cRow * ldc, c_frag, ldc,
                                wmma::mem_row_major);
    }
}

int main(int argc, char **argv)
{
    printf("Initializing...\n");

    int dev = findCudaDevice(argc, (const char **)argv);

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    // Tensor cores require a GPU of Volta (SM7X) architecture or higher.
    if (deviceProp.major < 7)
    {
        printf(
            "cudaTensorCoreGemm requires SM 7.0 or higher to use Tensor "
            "Cores.  Exiting...\n");
        exit(EXIT_WAIVED);
    }

    printf("M: %d (%d x %d)\n", M_GLOBAL, M, M_TILES);
    printf("N: %d (%d x %d)\n", N_GLOBAL, N, N_TILES);
    printf("K: %d (%d x %d)\n", K_GLOBAL, K, K_TILES);

    half *A_h = NULL;
    half *B_h = NULL;
    float *C_h = NULL;
#if CPU_DEBUG
    float *result_hD = NULL;
    float *result_host = NULL;
#endif

    A_h = (half *)malloc(sizeof(half) * M_GLOBAL * K_GLOBAL);
    B_h = (half *)malloc(sizeof(half) * K_GLOBAL * N_GLOBAL);
    C_h = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
#if CPU_DEBUG
    result_hD = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
    result_host = (float *)malloc(sizeof(float) * M_GLOBAL * N_GLOBAL);
#endif

    half *A = NULL;
    half *B = NULL;
    float *C = NULL;
    float *D = NULL;

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&A),
                               sizeof(half) * M_GLOBAL * K_GLOBAL));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&B),
                               sizeof(half) * N_GLOBAL * K_GLOBAL));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&C),
                               sizeof(float) * M_GLOBAL * N_GLOBAL));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&D),
                               sizeof(float) * M_GLOBAL * N_GLOBAL));

    assert(((unsigned long long)A) % 128 == 0);
    assert(((unsigned long long)B) % 128 == 0);
    assert(((unsigned long long)C) % 128 == 0);
    assert(((unsigned long long)D) % 128 == 0);

    init_host_matrices(A_h, B_h, C_h);

    printf("Preparing data for GPU...\n");

    checkCudaErrors(cudaMemcpy(A, A_h, sizeof(half) * M_GLOBAL * K_GLOBAL,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(B, B_h, sizeof(half) * N_GLOBAL * K_GLOBAL,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(C, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL,
                               cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemset(D, 0, sizeof(float) * M_GLOBAL * N_GLOBAL));

    enum
    {
        // Compute the right amount of shared memory to request.
        // We need shared memory to hold per-CTA C and D matrix tiles, and to cache per-CTA chunks
        // of the A and B matrices. 
        // Therefore, the right amount to request is the maximum of those two numbers.
        SHMEM_SZ = MAX(
            sizeof(half) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2,
            M * (BLOCK_ROW_WARPS * WARP_ROW_TILES) * N *
                (BLOCK_COL_WARPS * WARP_COL_TILES) * sizeof(float))
    };

    printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);

    const float alpha = 1.1f;
    const float beta = 1.2f;

    cudaEvent_t start, stop;

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start));

    // If enough shared memory available on the GPU use high performant kernel
    std::cout << "每个 block 的最大共享内存大小: " << deviceProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个 SM 最大的共享内存大小: " << deviceProp.sharedMemPerMultiprocessor / 1024.0 << " KB" << std::endl;
    if (deviceProp.sharedMemPerMultiprocessor >= SHMEM_SZ)
    {
        printf("Computing... using high performance kernel compute_gemm \n");

        checkCudaErrors(cudaFuncSetAttribute(
            compute_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));
        checkKernelErrors(
            (compute_gemm<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK,
                            SHMEM_SZ>>>(A, B, C, D, alpha, beta)));
#if CPU_DEBUG
        checkCudaErrors(cudaMemcpy(result_hD, D,
                                   sizeof(float) * M_GLOBAL * N_GLOBAL,
                                   cudaMemcpyDeviceToHost));
#endif
    }
    else
    {
        dim3 gridDim;
        dim3 blockDim;

        // blockDim.x must be a multple of warpSize
        // 128x4 means we have 16 warps and a block computes a 64x64 output tile
        blockDim.x = 128;
        blockDim.y = 4;

        gridDim.x = (M_GLOBAL + (WMMA_M * blockDim.x / 32 - 1)) /
                    (WMMA_M * blockDim.x / 32);
        gridDim.y = (N_GLOBAL + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

        printf("Computing... using simple_wmma_gemm kernel\n");
        simple_wmma_gemm<<<gridDim, blockDim>>>(A, B, C, D, M_GLOBAL, N_GLOBAL,
                                                K_GLOBAL, alpha, beta);
#if CPU_DEBUG
        checkCudaErrors(cudaMemcpy(result_hD, D,
                                   sizeof(float) * M_GLOBAL * N_GLOBAL,
                                   cudaMemcpyDeviceToHost));
#endif
    }

    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaEventSynchronize(stop));

#if CPU_DEBUG
    printf("Verifying correctness of the computations...\n");

    memcpy(result_host, C_h, sizeof(float) * M_GLOBAL * N_GLOBAL);

    matMultiplyOnHost(A_h, B_h, result_host, alpha, beta, M_GLOBAL, K_GLOBAL,
                      K_GLOBAL, N_GLOBAL, M_GLOBAL, N_GLOBAL);

    for (int i = 0; i < N_GLOBAL * M_GLOBAL; i++)
    {
        if (fabs(result_hD[i] - result_host[i]) > 0.1f)
            printf("mismatch i=%d result_hD=%f result_host=%f\n", i, result_hD[i],
                   result_host[i]);
    }
    free(result_hD);
    free(result_host);
#endif

    float milliseconds = 0;

    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Time: %f ms\n", milliseconds);
    printf("TFLOPS: %.2f\n", static_cast<double>((static_cast<double>(M_GLOBAL) *
                                                  N_GLOBAL * K_GLOBAL * 2) /
                                                 (milliseconds / 1000.)) /
                                 1e12);

    free(A_h);
    free(B_h);
    free(C_h);
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(A)));
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(B)));
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(C)));
    checkCudaErrors(cudaFree(reinterpret_cast<void *>(D)));

    return 0;
}