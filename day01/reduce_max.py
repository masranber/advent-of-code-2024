import os
import pycuda.driver as cuda
import numpy as np
from common import cuda_utils
from common.gpu_array import GPUArray

class ReduceMaxKernel(object):

    def __init__(self):
        src_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reduce_max.cu')
        src = cuda_utils.compile_cuda_source_file(src_file)
        self.kernel = src.get_function("reduce_max")

    def __call__(self, array: GPUArray):
        max_device = cuda.mem_alloc(np.int32().itemsize)
        cuda.memcpy_htod(max_device, np.int32(0))

        block, grid = cuda_utils.calc_cuda_kernel_layout_1D(array.size)
        self.kernel(array.gpudata, max_device, np.int32(array.size), block=block, grid=grid)

        max_host = np.empty(1, dtype=np.int32)
        cuda.memcpy_dtoh(max_host, max_device)
        return max_host[0]
