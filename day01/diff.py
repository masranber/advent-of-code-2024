import numpy as np
from common import cuda_utils
from common.gpu_array import GPUArray


class ElementwiseAbsDiffKernel(object):

    def __init__(self):
        src = cuda_utils.compile_cuda_source_file('diff.cu')
        self.kernel = src.get_function("diff")

    def __call__(self, array1: GPUArray, array2: GPUArray):
        out = GPUArray.empty(array1.size, array1.dtype)

        block, grid = cuda_utils.calc_cuda_kernel_layout_1D(array1.size)
        self.kernel(array1.gpudata, array2.gpudata, out.gpudata, np.int32(array1.size), block=block, grid=grid)

        return out
