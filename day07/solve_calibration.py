from math import ceil

import pycuda.driver as cuda
import numpy as np
from common import cuda_utils
from common.gpu_array import GPUArray


class SolveCalibrationPart1Kernel(object):

    def __init__(self):
        src = cuda_utils.compile_cuda_source_file('solve_calibration.cu')
        self.kernel = src.get_function("solve_calibration1")

    def __call__(self, calibrations: GPUArray, rows: int, cols: int):
        # output needs to be int64 or it will overflow on full puzzle input
        result_device = cuda.mem_alloc(np.int64().itemsize)
        cuda.memcpy_htod(result_device, np.int64(0))

        # perfect use case for parallel algorithm on a GPU
        # solution space grows 2^N with # of operands
        # each solution can be evaluated independently concurrently
        test_combinations = 2 ** (cols - 2)  # exclude test value, N operands will use N-1 operators, and there are two possible operators for each
        threads_per_block = min(test_combinations, 512)
        blocks_per_grid_y = ceil(test_combinations / threads_per_block)
        blocks_per_grid_x = rows
        self.kernel(calibrations.gpudata, np.int32(rows), np.int32(cols), np.int32(test_combinations), result_device, block=(1, threads_per_block, 1), grid=(blocks_per_grid_x, blocks_per_grid_y, 1))

        result_host = np.empty(1, dtype=np.int64)
        cuda.memcpy_dtoh(result_host, result_device)
        return result_host[0]


class SolveCalibrationPart2Kernel(object):

    def __init__(self):
        src = cuda_utils.compile_cuda_source_file('solve_calibration.cu')
        self.kernel = src.get_function("solve_calibration2")

    def __call__(self, calibrations: GPUArray, rows: int, cols: int):
        # output needs to be int64 or it will overflow on full puzzle input
        result_device = cuda.mem_alloc(np.int64().itemsize)
        cuda.memcpy_htod(result_device, np.int64(0))

        # having 3 operators now makes the solution space significantly larger
        # for 12 operands and 850 rows, there are 3.6 billion total combinations to test
        # even better use case for a GPU now
        # encoding the operators the easy way (2 bits per) results in test cases being evaluated multiple times but oh well
        test_combinations = 2 ** ((cols - 2) * 2) # add *2 to account for each operator representing two bits now
        threads_per_block = min(test_combinations, 512)
        blocks_per_grid_y = ceil(test_combinations / threads_per_block)
        blocks_per_grid_x = rows
        self.kernel(calibrations.gpudata, np.int32(rows), np.int32(cols), np.int32(test_combinations), result_device, block=(1, threads_per_block, 1), grid=(blocks_per_grid_x, blocks_per_grid_y, 1))

        result_host = np.empty(1, dtype=np.int64)
        cuda.memcpy_dtoh(result_host, result_device)
        return result_host[0]
