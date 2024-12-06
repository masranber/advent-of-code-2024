import numpy as np
from common import cuda_utils
from common.gpu_array import GPUArray


class BitonicMergeSortKernel(object):

    def __init__(self):
        src = cuda_utils.compile_cuda_source_file('bitonic_merge_sort.cu')
        self.kernel = src.get_function("bitonic_merge_sort")

    def __call__(self, array: GPUArray):
        block, grid = cuda_utils.calc_cuda_kernel_layout_1D(array.size)

        original_size = array.size
        if original_size.bit_count() != 1:
            padded_size = cuda_utils.next_pow2(original_size)  # bitonic merge sort requires the input array length be a power of 2
            array = GPUArray.from_cpu(np.pad(array.to_cpu(), (0, padded_size - original_size), mode='constant', constant_values=np.iinfo(np.int32).max))

        k = 2
        while k <= array.size:
            j = k // 2
            while j > 0:
                self.kernel(array.gpudata, np.int32(array.size), np.int32(k), np.int32(j), block=block, grid=grid)
                j //= 2
            k *= 2

        return GPUArray.from_cpu(array.to_cpu()[:original_size]) if original_size != array.size else array

