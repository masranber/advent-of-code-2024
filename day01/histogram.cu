
__global__ void histogram(int *arr, int *hist, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    atomicAdd(&hist[arr[i]], 1);
}
