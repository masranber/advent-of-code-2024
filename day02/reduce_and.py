import pycuda.driver as cuda
import numpy as np
from common import cuda_utils
from common.gpu_array import GPUArray


class ReduceBitwiseAndKernel(object):

    def __init__(self):
        src = cuda_utils.compile_cuda_source_file('reduce_and.cu')
        self.kernel = src.get_function("reduce_and")

    def __call__(self, array: GPUArray):
        result_device = cuda.mem_alloc(np.int32().itemsize)
        cuda.memcpy_htod(result_device, np.int32(0))

        block, grid = cuda_utils.calc_cuda_kernel_layout_1D(array.size)
        self.kernel(array.gpudata, result_device, np.int32(array.size), block=block, grid=grid)

        result_host = np.empty(1, dtype=np.int32)
        cuda.memcpy_dtoh(result_host, result_device)
        return result_host[0]
