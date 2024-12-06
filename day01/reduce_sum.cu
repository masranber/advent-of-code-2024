#define MAX_THREADS_PER_BLOCK 1024

__forceinline__ __device__ void reduce(volatile int* sdata, int i1, int i2) {
    sdata[i1] += sdata[i2];
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
Finds the sum of all elements in an array of integers using parallel reduction.

The parallel reduction algorithm is based on Nvidia's presentation on parallel reduction:
https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

I implemented all optimizations through 5. 6-7 are super verbose and tedious to implement. Not worth it
for a project like this.

*/
__global__ void reduce_sum(int *arr, int *sum, int n) {
    __shared__ int sdata[MAX_THREADS_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? arr[i] : 0;

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

    // at this point, sdata[0] contains the sum for the local block (local sum)
    // assign a single thread (0 is easiest) to be responsible for adding the local sum to the global sum
    if (tid == 0) atomicAdd(sum, sdata[0]);
}
