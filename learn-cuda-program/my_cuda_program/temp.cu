// 2025.2.14 Matmul性能提升之路 float32
// 所有输入都保证了不会有余数的情况
// 索引的形式保持为 基址base + 偏移量offset
// 其中的一些优化代码；针对了分块大小、任务规划不可随便动；但MNK规模还是可变的
// 
//  V1 没有任何优化的矩阵乘 初始版任务规划
//  V2 一个线程处理多个元素; 换任务规划
//  V3 引入了shared Mem 的矩阵乘
//  V4 一个线程处理多个元素 + sharedMem
//  V5 数据重排，优化shared mem的访存
//  V6 float4优化访存；向量化的读数据  换任务规划
//  V7 依据于 shared_A 的尺寸，解决bank冲突
//  V8 降低sharedMem的读取，内积转外积
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>

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

// V1 没有任何优化的GEMM 
// 任务规划上 一个线程计算结果得到一个C中的元素
__global__ void matmulKernel_V1(float *dA, float *dB, float *dC, int M, int K, int N)
{
    // bloc_dim （32，32） 写死 确保把1024个线程给用满
    // grid_dim  MNK维度上关于block_dim的延展
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    float sum = 0.0;

    // #pragma unroll 8 
    for(int i = 0; i < K; i++){
        sum += dA[row * K + i] * dB[i * N + col];
    }
    dC[row * N + col] = sum;
}

//V2 一个线程处理多个元素; 
//对比于V1 每个线程处理 TM * TN 个元素
template <int TM, int TN> //这样传进来才不会出现错误，直接是常量
__global__ void matmulKernel_V2(float *dA, float *dB, float *dC, int M, int K, int N)
{

    int indA = TM * (threadIdx.x + blockIdx.x * blockDim.x);
    int indB = TN * (threadIdx.y + blockIdx.y * blockDim.y);
    float tmp[TM][TN] = {0.0f};

    for (int index_q = 0; index_q < TM; index_q++){
        for (int index_v = 0; index_v < TN; index_v++){
            for (int s = 0; s < K; s++){
                tmp[index_q][index_v] += dA[(indA + index_q) * K + s] * dB[s * N + indB + index_v];
            }
            dC[(indA + index_q) * N + indB + index_v] = tmp[index_q][index_v];
        }
    }
    
    //循环展开: 按照TN来展开 效果比较好
    // for (int index_q = 0; index_q < TM; index_q++){
    //     for (int s = 0; s < K; s++){
    //         tmp[index_q][0] += dA[(indA + index_q) * K + s] * dB[s * N + indB];
    //         tmp[index_q][1] += dA[(indA + index_q) * K + s] * dB[s * N + indB+1];
    //         tmp[index_q][2] += dA[(indA + index_q) * K + s] * dB[s * N + indB+2];
    //         tmp[index_q][3] += dA[(indA + index_q) * K + s] * dB[s * N + indB+3];
    //     }
    //     dC[(indA + index_q) * N + indB] = tmp[index_q][0];
    //     dC[(indA + index_q) * N + indB+1] = tmp[index_q][1];
    //     dC[(indA + index_q) * N + indB+2] = tmp[index_q][2];
    //     dC[(indA + index_q) * N + indB+3] = tmp[index_q][3];
    // }

    //循环展开: 按照TM来展开 等于没展开
    // for (int index_v = 0; index_v < TN; index_v++){
    //     for (int s = 0; s < K; s++){
    //         tmp[0][index_v] += dA[(indA) * K + s] * dB[s * N + indB + index_v];
    //         tmp[1][index_v] += dA[(indA + 1) * K + s] * dB[s * N + indB + index_v];
    //         tmp[2][index_v] += dA[(indA + 2) * K + s] * dB[s * N + indB + index_v];
    //         tmp[3][index_v] += dA[(indA + 3) * K + s] * dB[s * N + indB + index_v];
    //     }
    //     dC[(indA) * N + indB + index_v] = tmp[0][index_v];
    //     dC[(indA + 1) * N + indB + index_v] = tmp[1][index_v];
    //     dC[(indA + 2) * N + indB + index_v] = tmp[2][index_v];
    //     dC[(indA + 3) * N + indB + index_v] = tmp[3][index_v];
    // }
}

// V3 使用到了 shared Mem的GEMM --- 利用内存层次优化了访存
// 任务规划和V1保持一致; 理解上应该结合 https://zhuanlan.zhihu.com/p/12789107689
__global__ void matmulKernel_V3(float *dA, float *dB, float *dC, int M, int K, int N, int BLOCK_DIM)
{
    // BlockSize(32, 32, 1) 对应了--- shared_A[32][32]、shared_B[32][32]
    // 应该把shared_A的空间看作二维的
    //
    // 牢记对于矩阵C来说，一个线程处理一个
    // 而其中具体的 threadIdx.x, threadIdx.y 还决定不到C中的元素，只是在索引shared_mem
    // 
    // 踩坑：在这里不结合图形；只是从小到大分析、结合数值、固定某部分不变 分析不出来;反而会误解
    // 此处的正确分析方式是 shared mem的不要放到整个大矩阵来看；就看做简单的两个 shared_A shared_B运算，算完了以后再放回完整矩阵
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    float sum = 0.0f;

    extern __shared__ float sharedMem[];
    float *shared_A = sharedMem;
    float *shared_B = sharedMem + BLOCK_DIM * BLOCK_DIM;

    int width = K / BLOCK_DIM; //在K维度上有多少个 BLOCK_DIM shared_A 小块 

    //第多少个 shared 小块, 每次循环要刷新在shared Mem中的值 
    //相当于每次循环结束都只算出来了 1/width 的结果矩阵C中值 --- shared块内是内积，shared块间是外积
    //在K维度上有 width 个 BLOCK_DIM 的 shared_A 小块；那么sum就要增加 width 次
    for (int shared_id = 0; shared_id < width; shared_id++) {           // 32 + 31*32 
        shared_A[threadIdx.x * BLOCK_DIM + threadIdx.y] = dA[row * K + (threadIdx.y + shared_id * BLOCK_DIM)]; 
        shared_B[threadIdx.x * BLOCK_DIM + threadIdx.y] = dB[(threadIdx.x + shared_id * BLOCK_DIM) * N + col];
        __syncthreads(); //按block来完整获取即可，暂不考虑计算 

        // 局部的结果；此线程获取的shared_A[this_thread]在此被用到了；更用到了当前这个block获取的其他存在shared_A的结果
        for (int s = 0; s < BLOCK_DIM; s++) {
            sum += shared_A[threadIdx.x * BLOCK_DIM + s] * shared_B[s * BLOCK_DIM + threadIdx.y];
        }
        __syncthreads();
    }
    
    dC[row * N + col] = sum;
}

// V4 一个线程处理多个元素 + sharedMem
template<int BM, int BN, int BK, int TM, int TN>
__global__ void matmulKernel_V4(float *dA, float *dB, float *dC, int M, int K, int N)
{
    __shared__ float shared_A[BM * BK];
    __shared__ float shared_B[BK * BN];
    int row_start = TM * (blockDim.x * blockIdx.x + threadIdx.x);
    int col_start = TN * (blockDim.y * blockIdx.y + threadIdx.y);
    int width = (K + BK - 1)/BK;
    
    float part_sum[TM * TN] = {0.0f};

    for(int shared_id = 0; shared_id<width; shared_id++){
        
        for (int index_q = 0; index_q < TM; index_q++){
            for (int index_k = 0; index_k < BK; index_k++){
                shared_A[(TM * threadIdx.x + index_q) * BK + index_k] = dA[(row_start + index_q) * K + (shared_id * BK + index_k)];
            }
        }
        // 小小优化了一下dB的访存
        for (int index_v = 0; index_v < TN; index_v++){
            for (int index_k = 0; index_k < BK; index_k++){
                shared_B[index_k * BN + (TN * threadIdx.y + index_v)] = dB[(shared_id * BK + index_k) * N + (col_start + index_v)];
            }
        }
        __syncthreads();
        
        // 一个线程应该处理 TM * TN 个元素; 每个元素都应该沿着BK的维度求部分的和
        for (int index_q = 0; index_q < TM; index_q++){
            for (int index_v = 0; index_v < TN; index_v++){
                for(int index_k = 0; index_k < BK; index_k++){
                    part_sum[index_q * TN + index_v] += shared_A[(TM * threadIdx.x + index_q) * BK + index_k] * shared_B[index_k * BN + (TN * threadIdx.y + index_v)];
                }
            }
        }
        __syncthreads();
    }

    for (int index_q = 0; index_q < TM; index_q++){
        for (int index_v = 0; index_v < TN; index_v++){
            dC[(row_start + index_q) * N + (col_start + index_v)] = part_sum[index_q * TN + index_v];
        }
    }
}

// V5 数据重排，优化shared mem的访存   参考 https://zhuanlan.zhihu.com/p/410278370
//需要保证 BM*BK = BK*BN = BLOCK_DIM_x * BLOCK_DIM_y 每个线程所 处理的元素恰好放进shared_A / shared_B
template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmulKernel_V5(float *dA, float *dB, float *dC, int M, int K, int N)
{
    __shared__ float shared_A[BM * BK]; //[128, 8]
    __shared__ float shared_B[BK * BN];
    int index_A = TM * (blockDim.x * blockIdx.x); //索引到block上
    int index_B = TN * (blockDim.y * blockIdx.y);
    int width = K / BK;
    float part_sum[TM * TN] = {0.0f};

    int tid = threadIdx.x + blockDim.x * threadIdx.y; //使得在 threadIdx.x上连续变化, 列主序
    // int tid = threadIdx.x * blockDim.y + threadIdx.y; //行主序时 -- 也是对的 但速度慢了
    int shared_A_m_idx = tid / 8;  
    int shared_A_k_idx = tid % 8;
    int shared_B_k_idx = tid / 128;
    int shared_B_n_idx = tid % 128; 
    // int shared_A_m_idx = tid % 128;  
    // int shared_A_k_idx = tid / 128;
    // int shared_B_k_idx = tid % 8;
    // int shared_B_n_idx = tid / 8; 

    for(int shared_id = 0; shared_id < width; shared_id++){

        shared_A[shared_A_m_idx * BK + shared_A_k_idx] = dA[(index_A + shared_A_m_idx) * K + (shared_id * BK + shared_A_k_idx)];
        shared_B[shared_B_k_idx * BN + shared_B_n_idx] = dB[(shared_id * BK + shared_B_k_idx) * N + (index_B + shared_B_n_idx)];
        __syncthreads();

        for (int index_q = 0; index_q < TM; index_q++){
            for (int index_v = 0; index_v < TN; index_v++){
                int reg_C_m_idx = threadIdx.x * TM + index_q;
                int reg_C_n_idx = threadIdx.y * TN + index_v;
                for(int index_k = 0; index_k < BK; index_k++){
                    part_sum[index_q * TN + index_v] += shared_A[reg_C_m_idx * BK + index_k] * shared_B[index_k * BN + reg_C_n_idx];
                }
            }
        }
        __syncthreads();
    }

    for (int index_q = 0; index_q < TM; index_q++){
        for (int index_v = 0; index_v < TN; index_v++){
            int reg_C_m_idx = threadIdx.x * TM + index_q;
            int reg_C_n_idx = threadIdx.y * TN + index_v;
            dC[(index_A + reg_C_m_idx) * N + (index_B + reg_C_n_idx)] = part_sum[index_q * TN + index_v];
        }
    }
}

// V6 引入了float4来加速访存
//需要更换任务规划；虽然看起来只是加了float4； 一个线程访问的元素太多会导致寄存器爆炸
//但是需要保持 BM = TM*BLOCK_DIM_X = 1024;  BN = TN*BLOCK_DIM_Y = 1024; TN = TM
template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmulKernel_V6(float *dA, float *dB, float *dC, int M, int K, int N)
{
    __shared__ float shared_A[BM * BK]; 
    __shared__ float shared_B[BK * BN];
    int index_A = TM * (blockDim.x * blockIdx.x); 
    int index_B = TN * (blockDim.y * blockIdx.y);
    int width = K / BK;
    float part_sum[TM * TN] = {0.0f};

    int tid = threadIdx.x + blockDim.x * threadIdx.y; //使得在 threadIdx.x上连续变化, 列主序
    int shared_A_m_idx = tid / 2;   //索引量减少到 1/4
    int shared_A_k_idx = tid % 2;
    int shared_B_k_idx = tid / 32;
    int shared_B_n_idx = tid % 32; 

    for(int shared_id = 0; shared_id < width; shared_id++){
        //float4 只与load的过程有关；计算时就无关了
        (float4 &)shared_A[shared_A_m_idx * BK + 4 * shared_A_k_idx] = (float4 &)dA[(index_A + shared_A_m_idx) * K + (shared_id * BK + 4 * shared_A_k_idx)];
        (float4 &)shared_B[shared_B_k_idx * BN + 4 * shared_B_n_idx] = (float4 &)dB[(shared_id * BK + shared_B_k_idx) * N + (index_B + 4 * shared_B_n_idx)];
        __syncthreads();

        for (int index_q = 0; index_q < TM; index_q++){
            for (int index_v = 0; index_v < TN; index_v++){
                int reg_C_m_idx = threadIdx.x * TM + index_q;
                int reg_C_n_idx = threadIdx.y * TN + index_v;
                for(int index_k = 0; index_k < BK; index_k++){
                    part_sum[index_q * TN + index_v] += shared_A[reg_C_m_idx * BK + index_k] * shared_B[index_k * BN + reg_C_n_idx];
                }
            }
        }
        __syncthreads();
    }

    for (int index_q = 0; index_q < TM; index_q++){
        for (int index_v = 0; index_v < TN; index_v++){
            int reg_C_m_idx = threadIdx.x * TM + index_q;
            int reg_C_n_idx = threadIdx.y * TN + index_v;
            dC[(index_A + reg_C_m_idx) * N + (index_B + reg_C_n_idx)] = part_sum[index_q * TN + index_v];
        }
    }
}

// V7 在sharedMem中解决BankConflict  
// 没有在sharedMem中添加冗余 但是改变了shared_A的数据布局；转置得存了shared_A
template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmulKernel_V7(float *dA, float *dB, float *dC, int M, int K, int N)
{
    __shared__ float shared_A[BM * BK]; 
    __shared__ float shared_B[BK * BN];
    int index_A = TM * (blockDim.x * blockIdx.x); 
    int index_B = TN * (blockDim.y * blockIdx.y);
    int width = K / BK;
    float part_sum[TM * TN] = {0.0f};

    int tid = threadIdx.x + blockDim.x * threadIdx.y; //使得在 threadIdx.x上连续变化, 列主序
    int shared_A_m_idx = tid / 2;   //索引量减少到 1/4
    int shared_A_k_idx = tid % 2;
    int shared_B_k_idx = tid / 32;
    int shared_B_n_idx = tid % 32; 

    float temp_a[4]; //作为填充

    for(int shared_id = 0; shared_id < width; shared_id++){

        (float4 &)temp_a[0] = (float4 &)dA[(index_A + shared_A_m_idx) * K + (shared_id * BK + 4 * shared_A_k_idx)];
        
        # pragma unroll // 提速从0.6427 -- 0.2260
        for(int id = 0; id < 4; id++){     // 在这里添加了填充        
            shared_A[(4 * shared_A_k_idx + id) * BM + shared_A_m_idx] = temp_a[id]; // 相当于A转置了再来存 改变了shared_A的数据布局
        }
        //(float4 &)shared_A[shared_A_m_idx * BK + 4 * shared_A_k_idx] = (float4 &)dA[(index_A + shared_A_m_idx) * K + (shared_id * BK + 4 * shared_A_k_idx)];
        (float4 &)shared_B[shared_B_k_idx * BN + 4 * shared_B_n_idx] = (float4 &)dB[(shared_id * BK + shared_B_k_idx) * N + (index_B + 4 * shared_B_n_idx)];

    
        __syncthreads();

        for (int index_q = 0; index_q < TM; index_q++){
            for (int index_v = 0; index_v < TN; index_v++){
                int reg_C_m_idx = threadIdx.x * TM + index_q;
                int reg_C_n_idx = threadIdx.y * TN + index_v;
                for(int index_k = 0; index_k < BK; index_k++){     //对应在这里 shared_A[] 的获取也变化了； 之前转置得存，现在转置地取
                    part_sum[index_q * TN + index_v] += shared_A[index_k * BM + reg_C_m_idx] * shared_B[index_k * BN + reg_C_n_idx];
                    // part_sum[index_q * TN + index_v] += shared_A[reg_C_m_idx * BK + index_k] * shared_B[index_k * BN + reg_C_n_idx]; // V6的版本
                }
            }
        }
        __syncthreads();
    }
    //存进dC的过程和前面就保持一致了；一个线程处理 TM * TN 个结果
    for (int index_q = 0; index_q < TM; index_q++){
        for (int index_v = 0; index_v < TN; index_v++){
            int reg_C_m_idx = threadIdx.x * TM + index_q;
            int reg_C_n_idx = threadIdx.y * TN + index_v;
            dC[(index_A + reg_C_m_idx) * N + (index_B + reg_C_n_idx)] = part_sum[index_q * TN + index_v];
        }
    }
}

// V8 降低sharedMem的重复读取
// 在load时和V7没有区别，存回dC时也没区别；只有计算局部结果时内积转外积
template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmulKernel_V8(float *dA, float *dB, float *dC, int M, int K, int N)
{
    __shared__ float shared_A[BM * BK]; 
    __shared__ float shared_B[BK * BN];
    int index_A = TM * (blockDim.x * blockIdx.x); 
    int index_B = TN * (blockDim.y * blockIdx.y);
    int width = K / BK;
    float part_sum[TM * TN] = {0.0f};
    int tid = threadIdx.x + blockDim.x * threadIdx.y; 
    int shared_A_m_idx = tid / 2;   
    int shared_A_k_idx = tid % 2;
    int shared_B_k_idx = tid / 32;
    int shared_B_n_idx = tid % 32; 
    float temp_a[4]; 
    float com_A[TM];
    float com_B[TN];

    for(int shared_id = 0; shared_id < width; shared_id++){

        (float4 &)temp_a[0] = (float4 &)dA[(index_A + shared_A_m_idx) * K + (shared_id * BK + 4 * shared_A_k_idx)];
        // # pragma unroll 
        for(int id = 0; id < 4; id++){     
            shared_A[(4 * shared_A_k_idx + id) * BM + shared_A_m_idx] = temp_a[id]; 
        }
        (float4 &)shared_B[shared_B_k_idx * BN + 4 * shared_B_n_idx] = (float4 &)dB[(shared_id * BK + shared_B_k_idx) * N + (index_B + 4 * shared_B_n_idx)];

    
        __syncthreads();


        for(int index_k = 0; index_k < BK; index_k++){ //循环顺序的调换 变成了外积
            //load TM = 8 个元素到 com_A
            (float4 &)com_A[0] = (float4 &)shared_A[index_k * BM + threadIdx.x * TM];
            (float4 &)com_A[4] = (float4 &)shared_A[index_k * BM + threadIdx.x * TM + 4]; 
            //load TN = 8 个元素到 com_B
            (float4 &)com_B[0] = (float4 &)shared_B[index_k * BN + threadIdx.y * TN];
            (float4 &)com_B[4] = (float4 &)shared_B[index_k * BN + threadIdx.y * TN + 4];

            for(int index_q = 0; index_q < TM; index_q++){
                for(int index_v = 0; index_v < TN; index_v++){
                    part_sum[index_q * TN + index_v] += com_A[index_q] * com_B[index_v];
                }
            }
        }
        __syncthreads();
    }


    for (int index_q = 0; index_q < TM; index_q++){
        for (int index_v = 0; index_v < TN; index_v++){
            int reg_C_m_idx = threadIdx.x * TM + index_q;
            int reg_C_n_idx = threadIdx.y * TN + index_v;
            dC[(index_A + reg_C_m_idx) * N + (index_B + reg_C_n_idx)] = part_sum[index_q * TN + index_v];
        }
    }
}


template <int BM, int BN, int BK, int TM, int TN>
__global__ void matmulKernel_V9(float *dA, float *dB, float *dC, int M, int K, int N)
{
    __shared__ float shared_A[BM * BK * 2]; 
    __shared__ float shared_B[BK * BN * 2];
    int index_A = TM * (blockDim.x * blockIdx.x); 
    int index_B = TN * (blockDim.y * blockIdx.y);
    int width = (K + BK - 1) / BK; // 确保宽度计算正确
    float part_sum[TM * TN] = {0.0f};
    int tid = threadIdx.x + blockDim.x * threadIdx.y; 
    int shared_A_m_idx = tid / 2;   
    int shared_A_k_idx = tid % 2;
    int shared_B_k_idx = tid / 32;
    int shared_B_n_idx = tid % 32; 
    float temp_a[4]; 
    float com_A[TM];
    float com_B[TN];

    // Load first block of data into shared memory
    (float4 &)temp_a[0] = (float4 &)dA[(index_A + shared_A_m_idx) * K + (0 * BK + 4 * shared_A_k_idx)];
    for(int id = 0; id < 4; id++){     
        if (index_A + shared_A_m_idx >= M || 0 * BK + 4 * shared_A_k_idx + id >= K) {
            shared_A[(4 * shared_A_k_idx + id) * BM + shared_A_m_idx] = 0.0f;
        } else {
            shared_A[(4 * shared_A_k_idx + id) * BM + shared_A_m_idx] = temp_a[id]; 
        }
    }
    (float4 &)shared_B[shared_B_k_idx * BN + 4 * shared_B_n_idx] = (float4 &)dB[(0 * BK + shared_B_k_idx) * N + (index_B + 4 * shared_B_n_idx)];
    for(int id = 0; id < 4; id++) {
        if (index_B + 4 * shared_B_n_idx + id >= N || 0 * BK + shared_B_k_idx >= K) {
            shared_B[shared_B_k_idx * BN + 4 * shared_B_n_idx + id] = 0.0f;
        }
    }
    __syncthreads();

    for(int shared_id = 1; shared_id < width; shared_id++){
        (float4 &)temp_a[0] = (float4 &)dA[(index_A + shared_A_m_idx) * K + (shared_id * BK + 4 * shared_A_k_idx)];
        for(int id = 0; id < 4; id++){     
            if (index_A + shared_A_m_idx >= M || shared_id * BK + 4 * shared_A_k_idx + id >= K) {
                shared_A[(4 * shared_A_k_idx + id) * BM + shared_A_m_idx + (shared_id % 2) * BM * BK] = 0.0f;
            } else {
                shared_A[(4 * shared_A_k_idx + id) * BM + shared_A_m_idx + (shared_id % 2) * BM * BK] = temp_a[id]; 
            }
        }
        (float4 &)shared_B[shared_B_k_idx * BN + 4 * shared_B_n_idx + (shared_id % 2) * BN * BK] = (float4 &)dB[(shared_id * BK + shared_B_k_idx) * N + (index_B + 4 * shared_B_n_idx)];
        for(int id = 0; id < 4; id++) {
            if (index_B + 4 * shared_B_n_idx + id >= N || shared_id * BK + shared_B_k_idx >= K) {
                shared_B[shared_B_k_idx * BN + 4 * shared_B_n_idx + id + (shared_id % 2) * BN * BK] = 0.0f;
            }
        }
        __syncthreads();

        int pipe_offset_A = (shared_id - 1) % 2 * BM * BK;
        int pipe_offset_B = (shared_id - 1) % 2 * BK * BN;
        for(int index_k = 0; index_k < BK; index_k++){
            (float4 &)com_A[0] = (float4 &)shared_A[index_k * BM + threadIdx.x * TM + pipe_offset_A];
            (float4 &)com_A[4] = (float4 &)shared_A[index_k * BM + threadIdx.x * TM + 4 + pipe_offset_A]; 
            (float4 &)com_B[0] = (float4 &)shared_B[index_k * BN + threadIdx.y * TN + pipe_offset_B];
            (float4 &)com_B[4] = (float4 &)shared_B[index_k * BN + threadIdx.y * TN + 4 + pipe_offset_B];

            for(int index_q = 0; index_q < TM; index_q++){
                for(int index_v = 0; index_v < TN; index_v++){
                    part_sum[index_q * TN + index_v] += com_A[index_q] * com_B[index_v];
                }
            }
        }
        __syncthreads();
    }

    // Process the last block
    int pipe_offset_A = (width - 1) % 2 * BM * BK;
    int pipe_offset_B = (width - 1) % 2 * BK * BN;
    for(int index_k = 0; index_k < BK; index_k++){ 
        (float4 &)com_A[0] = (float4 &)shared_A[index_k * BM + threadIdx.x * TM + pipe_offset_A];
        (float4 &)com_A[4] = (float4 &)shared_A[index_k * BM + threadIdx.x * TM + 4 + pipe_offset_A]; 
        (float4 &)com_B[0] = (float4 &)shared_B[index_k * BN + threadIdx.y * TN + pipe_offset_B];
        (float4 &)com_B[4] = (float4 &)shared_B[index_k * BN + threadIdx.y * TN + 4 + pipe_offset_B];

        for(int index_q = 0; index_q < TM; index_q++){
            for(int index_v = 0; index_v < TN; index_v++){
                part_sum[index_q * TN + index_v] += com_A[index_q] * com_B[index_v];
            }
        }
    }

    // Write results back to global memory
    for (int index_q = 0; index_q < TM; index_q++){
        for (int index_v = 0; index_v < TN; index_v++){
            int reg_C_m_idx = threadIdx.x * TM + index_q;
            int reg_C_n_idx = threadIdx.y * TN + index_v;
            if (index_A + reg_C_m_idx < M && index_B + reg_C_n_idx < N) {
                dC[(index_A + reg_C_m_idx) * N + index_B + reg_C_n_idx] = part_sum[index_q * TN + index_v];
            }
        }
    }
}



int main()
{
    float *hostA, *hostB, *hostC, *serialC;
    int repeat_times = 100, warmup_times = 25;
    int M = 1024;
    int N = 1024;
    int K = 1024;
    // V1 V3 初始版设置
    // const int BLOCK_DIM_x = 32;
    // const int BLOCK_DIM_y = 32;
    // const int BLOCK_DIM = BLOCK_DIM_x;
    // V2 V4 V5 一个线程处理多个元素的设置 
    // const int TM = 4;
    // const int TN = 4;
    // const int BM = TM * BLOCK_DIM_x;
    // const int BN = TN * BLOCK_DIM_y;
    // const int BK = 8;
    // V6 V7
    const int TM = 8;
    const int TN = 8;
    const int BLOCK_DIM_x = 16;
    const int BLOCK_DIM_y = 16;
    const int BM = TM * BLOCK_DIM_x;
    const int BN = TN * BLOCK_DIM_y;
    const int BK = 8;


    hostA = (float *)malloc(M * K * sizeof(float));
    hostB = (float *)malloc(N * K * sizeof(float));
    hostC = (float *)malloc(M * N * sizeof(float));
    serialC = (float *)malloc(M * N * sizeof(float));
    init_matrix(hostA, M, K, 5);
    init_matrix(hostB, K, N, 10);


    float *dA, *dB, *dC;
    cudaMalloc((void **)&dA, M * K * sizeof(float));
    cudaMalloc((void **)&dB, N * K * sizeof(float));
    cudaMalloc((void **)&dC, M * N * sizeof(float));
    cudaMemcpy(dA, hostA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hostB, N * K * sizeof(float), cudaMemcpyHostToDevice);


    int num_blocks_x = (M + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
    int num_blocks_y = (N + BLOCK_DIM_y - 1) / BLOCK_DIM_y;
    dim3 block_dim(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid_dim(num_blocks_x, num_blocks_y, 1);

    int num_blocks_x_TMTN = (M + BM - 1) / BM;
    int num_blocks_y_TMTN = (N + BN - 1) / BN;
    dim3 grid_dim_TMTN(num_blocks_x_TMTN, num_blocks_y_TMTN, 1);

    // int sharedMemSize = 2 * BLOCK_DIM * BLOCK_DIM * sizeof(float);
    int sharedMemSize_TMTN = (BM * BK + BN * BK) * sizeof(float);
    printf("Shared Mem used: %.0f KB\n", sharedMemSize_TMTN / 1024.0);
    
    
    // matmulKernel_V1<<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
    // matmulKernel_V2<TM, TN><<<grid_dim_TMTN, block_dim>>>(dA, dB, dC, M, K, N);
    // matmulKernel_V3<<<grid_dim, block_dim, sharedMemSize>>>(dA, dB, dC, M, K, N, BLOCK_DIM);
    // matmulKernel_V4<BM, BN, BK, TM, TN><<<grid_dim_TMTN, block_dim>>>(dA, dB, dC, M, K, N);
    // matmulKernel_V5<BM, BN, BK, TM, TN><<<grid_dim_TMTN, block_dim>>>(dA, dB, dC, M, K, N);
    // matmulKernel_V6<BM, BN, BK, TM, TN><<<grid_dim_TMTN, block_dim>>>(dA, dB, dC, M, K, N);
    // matmulKernel_V7<BM, BN, BK, TM, TN><<<grid_dim_TMTN, block_dim>>>(dA, dB, dC, M, K, N);
    // matmulKernel_V8<BM, BN, BK, TM, TN><<<grid_dim_TMTN, block_dim>>>(dA, dB, dC, M, K, N);
    matmulKernel_V9<BM, BN, BK, TM, TN><<<grid_dim_TMTN, block_dim>>>(dA, dB, dC, M, K, N);

    cudaMemcpy(hostC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    double start_time_serial, end_time_serial;
    start_time_serial = get_walltime();
    matrixSerial(hostA, hostB, serialC, M, K, N);
    end_time_serial = get_walltime();
    float sum_error = compare(hostC, serialC, M, N);

    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < repeat_times + warmup_times; i++){
        if(i == warmup_times)cudaEventRecord(start, 0);

        // matmulKernel_V1<<<grid_dim, block_dim>>>(dA, dB, dC, M, K, N);
        // matmulKernel_V2<TM, TN><<<grid_dim_TMTN, block_dim>>>(dA, dB, dC, M, K, N);
        // matmulKernel_V3<<<grid_dim, block_dim, sharedMemSize>>>(dA, dB, dC, M, K, N, BLOCK_DIM);
        // matmulKernel_V4<BM, BN, BK, TM, TN><<<grid_dim_TMTN, block_dim>>>(dA, dB, dC, M, K, N);
        // matmulKernel_V5<BM, BN, BK, TM, TN><<<grid_dim_TMTN, block_dim>>>(dA, dB, dC, M, K, N);
        // matmulKernel_V6<BM, BN, BK, TM, TN><<<grid_dim_TMTN, block_dim>>>(dA, dB, dC, M, K, N);
        // matmulKernel_V7<BM, BN, BK, TM, TN><<<grid_dim_TMTN, block_dim>>>(dA, dB, dC, M, K, N);
        // matmulKernel_V8<BM, BN, BK, TM, TN><<<grid_dim_TMTN, block_dim>>>(dA, dB, dC, M, K, N);
        matmulKernel_V9<BM, BN, BK, TM, TN><<<grid_dim_TMTN, block_dim>>>(dA, dB, dC, M, K, N);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("M-K-N: %d-%d-%d\n", M, K, N);
    printf("CPU Runnning Time: %.4f second\n", (end_time_serial - start_time_serial));
    printf("GPU Kernel   Time: %.4f ms\n", ker_time / repeat_times);
    printf("grid  dim: (%d, %d, %d)\n", grid_dim_TMTN.x, grid_dim_TMTN.y, grid_dim_TMTN.z);
    printf("block dim: (%d, %d, %d)\n", block_dim.x, block_dim.y, block_dim.z);
    printf("Sum Error: %.4f\n", sum_error);


    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    free(hostA);
    free(hostB);
    free(hostC);
    free(serialC);
    return 0;
}