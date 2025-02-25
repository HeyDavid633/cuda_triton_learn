#include <stdio.h>
#define WARP_SIZE 32

__global__ void warpid_test() {
    const int warpNums = (blockDim.x >> 5);
    
    const int local_warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = (WARP_SIZE - 1) & threadIdx.x;
    
    printf("warpNums =  %d ,local_warp_id = %d, lane_id = %d\n", warpNums, local_warp_id, lane_id );
}

__global__ void warpid_test2() {
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = thread_id / 32;
    printf("Thread (%d, %d) is in warp %d\n", threadIdx.x, threadIdx.y, warp_id);
}

int main() {
    dim3 block(3, 32);

    // warpid_test<<< 1, 64 >>>();
    warpid_test2<<< 1, block>>>();
    cudaDeviceSynchronize();

    return 0;
}