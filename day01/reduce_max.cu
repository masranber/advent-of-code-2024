#include <limits.h>

#define MAX_THREADS_PER_BLOCK 1024

__forceinline__ __device__ void reduce(volatile int* sdata, int i1, int i2) {
    int v1 = sdata[i1];
    int v2 = sdata[i2];
    if(v2 > v1) sdata[i1] = v2;
}

__device__ void warp_reduce(volatile int* sdata, int tid) {
    reduce(sdata, tid, tid + 32);
    reduce(sdata, tid, tid + 16);
    reduce(sdata, tid, tid + 8);
    reduce(sdata, tid, tid + 4);
    reduce(sdata, tid, tid + 2);
    reduce(sdata, tid, tid + 1);
}

/*
Finds the maximum value in an array of integers using parallel reduction.

The parallel reduction algorithm is based on Nvidia's presentation on parallel reduction:
https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

I only implemented optimizations 3 and 5. I could not get 4 (combining first reduction with global load)
to work with the max reduction operation. 6-7 are super verbose and tedious to implement. Not worth it
for a project like this.

*/
__global__ void reduce_max(int *arr, int *max, int n) {
    __shared__ int sdata[MAX_THREADS_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? arr[i] : INT_MIN;

    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            reduce(sdata, tid, tid + stride);
        }
        __syncthreads();
    }

    // unroll the last warp (32 threads) from the for loop
    // the last warp does the most work so unrolling provides a nice perf boost 
    if(tid < 32) warp_reduce(sdata, tid);

    // at this point, sdata[0] contains the max for the local block (local max)
    // assign a single thread (0 is easiest) to be responsible for writing the local max
    // to the global max if it's greater than the global max
    if (tid == 0) atomicMax(max, sdata[0]);
}
