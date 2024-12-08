
__global__ void solve_calibration1(unsigned long long *calibrations, int rows, int cols, int n, unsigned long long *result) {
    unsigned int row = blockIdx.x;  // row in calibrations
    unsigned int i = threadIdx.y + blockIdx.y * blockDim.y; // the combination to test
    if (row < rows && i < n) {
        int cs = row * cols; // calibration start index
        unsigned long long calc = calibrations[cs + 1]; // 1st operand
        for(int c = 2; c < cols; c++) { // skip test value and 1st operand
            int ci = cs + c; // calibration current index
            // use bit masking of i to determine the operator for current operands
            // i = 3 = 011 = add, mult, mult
            int op = (i >> (c - 2)) & 1;
            unsigned long long operand = calibrations[ci];
            calc = ((calc + operand) * (op == 0)) + ((calc * operand) * (op == 1)); // again funky math to avoid branching
        }
        if(calibrations[cs] == calc) {
            unsigned long long old_val = calibrations[cs];
            if (atomicCAS(&calibrations[cs], old_val, 0ULL) == old_val) {
                // only the thread that successfully updates calibrations[cs] will add to the result
                // otherwise rows with more than 1 solution will end up adding duplicates to the result
                atomicAdd(result, calc);
            }
        }
    }
}


// concat two ints together. 19 || 10 = 1910
__device__ unsigned long long concat(unsigned long long a, unsigned long long b) {
    unsigned long long multiplier = 1;
    unsigned long long temp = b;

    while (temp > 0) {
        multiplier *= 10;
        temp /= 10;
    }

    return a * multiplier + b;
}


__global__ void solve_calibration2(unsigned long long *calibrations, int rows, int cols, int n, unsigned long long *result) {
    unsigned int row = blockIdx.x;  // row in calibrations
    unsigned int i = threadIdx.y + blockIdx.y * blockDim.y; // the combination to test
    if (row < rows && i < n) {
        int cs = row * cols; // calibration start index
        unsigned long long calc = calibrations[cs + 1]; // 1st operand
        for(int c = 2; c < cols; c++) { // skip test value and 1st operand
            int ci = cs + c; // calibration current index
            // use bit masking of i and i+1th bits to determine the operator for current operands
            // i = 36 = 100100 = concat, mult, add
            int op = (i >> ((c - 2) * 2)) & 3; // need *2 to skip every other bit since we're anding with 2 bits now, &3 will mask two consecutive bits
            unsigned long long operand = calibrations[ci];
            calc = ((calc + operand) * (op == 0)) + ((calc * operand) * (op == 1)) + (concat(calc, operand) * (op == 2)); // again funky math to avoid branching
        }
        if(calibrations[cs] == calc) {
            unsigned long long old_val = calibrations[cs];
            if (atomicCAS(&calibrations[cs], old_val, 0ULL) == old_val) {
                // only the thread that successfully updates calibrations[cs] will add to the result
                // otherwise rows with more than 1 solution will end up adding duplicates to the result
                atomicAdd(result, calc);
            }
        }
    }
}