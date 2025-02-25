// 2025.2.16
// 在谭升的博客中学习到的 shared Mem以行主序（固定行，列上移动）去访问是最优的
// https://github.com/Tony-Tan/CUDA_Freshman/blob/master/24_shared_memory_read_data/shared_memory_read_data.cu
// https://face2ai.com/CUDA-F-5-2-%E5%85%B1%E4%BA%AB%E5%86%85%E5%AD%98%E7%9A%84%E6%95%B0%E6%8D%AE%E5%B8%83%E5%B1%80/

#include <cuda_runtime.h>
#include <stdio.h>
#define BDIMX 32
#define BDIMY 32

#define BDIMX_RECT 32
#define BDIMY_RECT 16
#define IPAD 1 //填充物

//行主序存，行主序读，没有读写冲突 恰好没有bank冲突
__global__ void setRowReadRow(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.y][threadIdx.x];
}
//列主序存，列主序读 --- 把读写冲突拉满了
__global__ void setColReadCol(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.x][threadIdx.y] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];
}
__global__ void setColReadRow(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.x][threadIdx.y] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.y][threadIdx.x];
}
__global__ void setRowReadCol(int *out)
{
    __shared__ int tile[BDIMY][BDIMX];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

// 动态声明共享内存，在入口的时候给到shared mem尺寸
__global__ void setRowReadColDyn(int *out)
{
    extern __shared__ int tile[];
    unsigned int row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int col_idx = threadIdx.x * blockDim.y + threadIdx.y;
    tile[row_idx] = row_idx;
    __syncthreads();
    out[row_idx] = tile[col_idx];
}

//填充一下，规避bank冲突
__global__ void setRowReadColIpad(int *out)
{
    __shared__ int tile[BDIMY][BDIMX + IPAD];//只在声明的时候声明，使用的时候和以前一样哦～
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[threadIdx.x][threadIdx.y];
}
__global__ void setRowReadColDynIpad(int *out)
{
    extern __shared__ int tile[];
    unsigned int row_idx = threadIdx.y * (blockDim.x + 1) + threadIdx.x;
    unsigned int col_idx = threadIdx.x * (blockDim.x + 1) + threadIdx.y;
    tile[row_idx] = row_idx;
    __syncthreads();
    out[row_idx] = tile[col_idx];
}


//--------------------rectagle---------------------
__global__ void setRowReadColRect(int *out)
{
    __shared__ int tile[BDIMY_RECT][BDIMX_RECT];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int icol = idx % blockDim.y;
    unsigned int irow = idx / blockDim.y;
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[icol][irow];
}
__global__ void setRowReadColRectDyn(int *out)
{
    extern __shared__ int tile[];
    //以idx连续变化的为主，所以是row主序存 --- idx是行主序
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int icol = idx % blockDim.y;
    unsigned int irow = idx / blockDim.y;

    unsigned int col_idx = (icol * blockDim.x) + irow; //方便以列主序读, irow意味着行在动
    tile[idx] = idx;
    __syncthreads();
    out[idx] = tile[col_idx];
}
__global__ void setRowReadColRectPad(int *out)
{
    // BDIMY_RECT是32的1/2 所以填充两列才可错开
    // 然后在一维的时候，就呈现为一列32个的错开
    __shared__ int tile[BDIMY_RECT][BDIMX_RECT + IPAD * 2]; 
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int icol = idx % blockDim.y;
    unsigned int irow = idx / blockDim.y;
    tile[threadIdx.y][threadIdx.x] = idx;
    __syncthreads();
    out[idx] = tile[icol][irow];
}
__global__ void setRowReadColRectDynPad(int *out)
{
    extern __shared__ int tile[];
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int icol = idx % blockDim.y;
    unsigned int irow = idx / blockDim.y;
    unsigned int row_idx = threadIdx.y * (IPAD + blockDim.x) + threadIdx.x;
    unsigned int col_idx = icol * (IPAD + blockDim.x) + irow;
    tile[row_idx] = idx;
    __syncthreads();
    out[idx] = tile[col_idx];
}

int main(int argc, char **argv)
{
    int nElem = BDIMX * BDIMY;
    printf("Vector size:%d\n", nElem);
    int nByte = sizeof(int) * nElem;
    int *out;
    int repeat_times = 100, warmup_times = 25;

    cudaMalloc((int **)&out, nByte);
    cudaSharedMemConfig MemConfig;
    cudaDeviceGetSharedMemConfig(&MemConfig);
    
    printf("--------------------------------------------\n");
    switch (MemConfig)
    {
    case cudaSharedMemBankSizeFourByte:
        printf("the device is cudaSharedMemBankSizeFourByte: 4-Byte\n");
        break;
    case cudaSharedMemBankSizeEightByte:
        printf("the device is cudaSharedMemBankSizeEightByte: 8-Byte\n");
        break;
    }
    printf("--------------------------------------------\n");

    dim3 block(BDIMX, BDIMY);
    dim3 grid(1, 1);
    dim3 block_rect(BDIMX_RECT, BDIMY_RECT);
    dim3 grid_rect(1, 1);

    // setRowReadRow<<<grid, block>>>(out);
    // cudaDeviceSynchronize();
    
    // setColReadCol<<<grid, block>>>(out);
    // cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    float ker_time = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < repeat_times + warmup_times; i++){
        if(i == warmup_times)cudaEventRecord(start, 0);

        setRowReadRow<<<grid, block>>>(out);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("setRowReadRow  ");
    printf("GPU Kernel   Time: %.4f ms\n", ker_time / repeat_times);

    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < repeat_times + warmup_times; i++){
        if(i == warmup_times)cudaEventRecord(start, 0);

        setColReadRow<<<grid, block>>>(out);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("setColReadRow  ");
    printf("GPU Kernel   Time: %.4f ms\n", ker_time / repeat_times);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < repeat_times + warmup_times; i++){
        if(i == warmup_times)cudaEventRecord(start, 0);

        setRowReadCol<<<grid, block>>>(out);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("setRowReadCol  ");
    printf("GPU Kernel   Time: %.4f ms\n", ker_time / repeat_times);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < repeat_times + warmup_times; i++){
        if(i == warmup_times)cudaEventRecord(start, 0);

        setColReadCol<<<grid, block>>>(out);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("setColReadCol  ");
    printf("GPU Kernel   Time: %.4f ms\n", ker_time / repeat_times);


    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int shared_mem_size = BDIMX * BDIMY * sizeof(int);
    for (int i = 0; i < repeat_times + warmup_times; i++){
        if(i == warmup_times)cudaEventRecord(start, 0);

        setRowReadColDyn<<<grid, block, shared_mem_size>>>(out);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("setRowReadColDyn  ");
    printf("GPU Kernel   Time: %.4f ms\n", ker_time / repeat_times);


    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < repeat_times + warmup_times; i++){
        if(i == warmup_times)cudaEventRecord(start, 0);

        setRowReadColIpad<<<grid, block>>>(out);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("setRowReadColIpad ");
    printf("GPU Kernel   Time: %.4f ms\n", ker_time / repeat_times);


    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < repeat_times + warmup_times; i++){
        if(i == warmup_times)cudaEventRecord(start, 0);

        setRowReadColRect<<<grid_rect, block_rect>>>(out);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("setRowReadColRect ");
    printf("GPU Kernel   Time: %.4f ms\n", ker_time / repeat_times);


    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int shared_mem_size_rect = BDIMX_RECT * BDIMY_RECT * sizeof(int);
    for (int i = 0; i < repeat_times + warmup_times; i++){
        if(i == warmup_times)cudaEventRecord(start, 0);

        setRowReadColRectDyn<<<grid, block, shared_mem_size_rect>>>(out);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ker_time, start, stop); // must float ker_time
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("setRowReadColRectDyn  ");
    printf("GPU Kernel   Time: %.4f ms\n", ker_time / repeat_times);


    



    cudaFree(out);
    return 0;
}