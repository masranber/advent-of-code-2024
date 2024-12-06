import pycuda.driver as cuda
import numpy as np
from common import cuda_utils
from common.gpu_array import GPUArray


class SafeReportKernel(object):

    def __init__(self):
        src = cuda_utils.compile_cuda_source_file('safe_report.cu')
        self.kernel = src.get_function("safe_report")

    def __call__(self, reports: GPUArray, cols: int, results: GPUArray = None, skip_col: int = -1):
        rows = reports.size // cols
        if results is None:
            results = GPUArray.empty(rows, np.int32)
        self.kernel(reports.gpudata, results.gpudata, np.int32(cols), np.int32(skip_col), block=(cols, 1, 1), grid=(rows, 1, 1))
        return results
