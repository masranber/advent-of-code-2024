/*
This only works when reports has at least one more column than n and those columns are padded with 0s 
*/
__global__ void safe_report(int *reports, int *result, int n, int skip_col) {
    __shared__ int is_safe_inc;
    __shared__ int is_safe_dec;

    unsigned int tid = threadIdx.x;
    unsigned int row = blockIdx.x;

    if(tid == 0) {
        is_safe_inc = 1;
        is_safe_dec = 1;
    }

    __syncthreads();

    if(tid < (n - 1)) {
        // Rely on boolean logic to avoid if statements. Branching on GPU is expensive and not easily predictable
        // like on CPU
        unsigned int i = row * n + threadIdx.x;
        int level1 = reports[i];
        int level2 = (tid + 1 == skip_col) ? reports[i + 2] : reports[i + 1];
        bool skip = level1 == 0 || level2 == 0 || tid == skip_col;
        int diff = level2 - level1;
        atomicAnd(&is_safe_inc, skip || (diff >= 1 && diff <= 3));
        atomicAnd(&is_safe_dec, skip || (diff <= -1 && diff >= -3));
    }

    __syncthreads();

    if (tid == 0) {
        result[row] |= is_safe_inc || is_safe_dec;
    }
}
