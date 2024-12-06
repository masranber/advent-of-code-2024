
__forceinline__ __device__ void swap(int *arr, int i1, int i2) {
    int temp = arr[i1];
    arr[i1] = arr[i2];
    arr[i2] = temp;
}

/*
CUDA C implementation of the Bitonic Merge Sort pseudocode at:
https://en.wikipedia.org/wiki/Bitonic_sorter#Example_code
*/
__global__ void bitonic_merge_sort(int *arr, int n, int k, int j) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x; // this is the index in the array this thread is assigned to
    unsigned int l = i ^ j;

    if(l > i) {
        if(((i & k) == 0 && arr[i] > arr[l]) || ((i & k) != 0 && arr[i] < arr[l])) {
            swap(arr, i, l);
        }
    }
}
