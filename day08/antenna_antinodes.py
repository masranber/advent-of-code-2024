from math import ceil

import pycuda.driver as cuda
import numpy as np
from common import cuda_utils
from common.gpu_array import GPUArray


class AntennaAntinodesKernel(object):

    def __init__(self):
        src = cuda_utils.compile_cuda_source_file('antenna_antinodes.cu')
        self.kernel = src.get_function("antenna_antinodes")

    def __call__(self, antenna_map: GPUArray, rows, cols) -> GPUArray:
        antinodes_map = GPUArray.zeros(antenna_map.size, dtype=np.int32)

        threads = rows * cols
        threads_per_block = min(threads, 512)
        blocks_per_grid = ceil(threads / threads_per_block)

        self.kernel(antenna_map.gpudata, antinodes_map.gpudata, np.int32(rows), np.int32(cols), block=(threads_per_block, 1, 1), grid=(blocks_per_grid, 1, 1))
        return antinodes_map


class AntennaAntinodesHarmonicKernel(object):

    def __init__(self):
        src = cuda_utils.compile_cuda_source_file('antenna_antinodes.cu')
        self.kernel = src.get_function("antenna_antinodes_harmonic")

    def __call__(self, antenna_map: GPUArray, rows, cols) -> GPUArray:
        antinodes_map = GPUArray.zeros(antenna_map.size, dtype=np.int32)

        threads = rows * cols
        threads_per_block = min(threads, 512)
        blocks_per_grid = ceil(threads / threads_per_block)

        self.kernel(antenna_map.gpudata, antinodes_map.gpudata, np.int32(rows), np.int32(cols), block=(threads_per_block, 1, 1), grid=(blocks_per_grid, 1, 1))
        return antinodes_map

