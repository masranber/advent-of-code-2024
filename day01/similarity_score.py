import numpy as np
from common import cuda_utils
from common.gpu_array import GPUArray


class SimilarityScoreKernel(object):

    def __init__(self):
        src = cuda_utils.compile_cuda_source_file('similarity_score.cu')
        self.kernel = src.get_function('similarity_score')

    def __call__(self, array: GPUArray, hist: GPUArray) -> GPUArray:
        scores = GPUArray.empty(array.size, dtype=np.int32)

        block, grid = cuda_utils.calc_cuda_kernel_layout_1D(array.size)
        self.kernel(array.gpudata, hist.gpudata, scores.gpudata, np.int32(array.size), block=block, grid=grid)

        return scores
