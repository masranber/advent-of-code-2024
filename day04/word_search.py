import pycuda.driver as cuda
import numpy as np
from common import cuda_utils
from common.gpu_array import GPUArray


class WordSearchKernel(object):

    def __init__(self):
        src = cuda_utils.compile_cuda_source_file('word_search.cu')
        self.kernel = src.get_function("word_search")

    def __call__(self, word_search: GPUArray, rows: int, cols: int):
        result_device = cuda.mem_alloc(np.int32().itemsize)
        cuda.memcpy_htod(result_device, np.int32(0))

        self.kernel(word_search.gpudata, np.int32(rows), np.int32(cols), result_device, block=(1024, 1, 1), grid=(1, 1, 1))

        result_host = np.empty(1, dtype=np.int32)
        cuda.memcpy_dtoh(result_host, result_device)
        return result_host[0]
