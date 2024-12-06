import pycuda.driver as cuda
import numpy as np
from common import cuda_utils
from common.gpu_array import GPUArray
import os


class ReduceSumKernel(object):

    def __init__(self):
        src_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reduce_sum.cu')
        src = cuda_utils.compile_cuda_source_file(src_file)
        self.kernel = src.get_function("reduce_sum")

    def __call__(self, array: GPUArray):
        sum_device = cuda.mem_alloc(np.int32().itemsize)
        cuda.memcpy_htod(sum_device, np.int32(0))

        block, grid = cuda_utils.calc_cuda_kernel_layout_1D(array.size)
        self.kernel(array.gpudata, sum_device, np.int32(array.size), block=block, grid=grid)

        sum_host = np.empty(1, dtype=np.int32)
        cuda.memcpy_dtoh(sum_host, sum_device)
        return sum_host[0]
