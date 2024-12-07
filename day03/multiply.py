import numpy as np
from common import cuda_utils
from common.gpu_array import GPUArray
import os


class ElementwiseMultiplyKernel(object):

    def __init__(self):
        src_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'multiply.cu')
        src = cuda_utils.compile_cuda_source_file(src_file)
        self.kernel = src.get_function("multiply")

    def __call__(self, array1: GPUArray, array2: GPUArray):
        out = GPUArray.empty(array1.size, array1.dtype)

        block, grid = cuda_utils.calc_cuda_kernel_layout_1D(array1.size)
        self.kernel(array1.gpudata, array2.gpudata, out.gpudata, np.int32(array1.size), block=block, grid=grid)

        return out
