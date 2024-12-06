
__global__ void similarity_score(int *arr, int *hist, int *scores, int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    scores[i] = arr[i] * hist[arr[i]];
}
