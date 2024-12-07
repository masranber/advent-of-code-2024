
/*
Performs an elementwise multiplication of two arrays.
*/
__global__ void multiply(int *arr1, int *arr2, int *arr_out, int n) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i >= n) return;
    arr_out[i] = arr1[i] * arr2[i];
}
