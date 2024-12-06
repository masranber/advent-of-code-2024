import numpy as np
from common import cuda_utils
from common.gpu_array import GPUArray


class HistogramKernel(object):

    def __init__(self):
        src = cuda_utils.compile_cuda_source_file('histogram.cu')
        self.kernel = src.get_function('histogram')

    def __call__(self, array: GPUArray, bins: int) -> GPUArray:
        hist = GPUArray.zeros(bins, dtype=np.int32)

        block, grid = cuda_utils.calc_cuda_kernel_layout_1D(array.size)
        self.kernel(array.gpudata, hist.gpudata, np.int32(array.size), block=block, grid=grid)

        return hist
