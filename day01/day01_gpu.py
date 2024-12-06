import pycuda.autoinit

import numpy as np

from common.gpu_array import GPUArray
from common.solution import Solution
from bitonic_merge_sort import BitonicMergeSortKernel
from diff import ElementwiseAbsDiffKernel
from reduce_sum import ReduceSumKernel
from reduce_max import ReduceMaxKernel
from histogram import HistogramKernel
from similarity_score import SimilarityScoreKernel

class Day01Part01GpuSolution(Solution):

    def __init__(self):
        self.array1 = None
        self.array2 = None
        self.bitonicMergeSortKernel = BitonicMergeSortKernel()
        self.elementwiseAbsDiffKernel = ElementwiseAbsDiffKernel()
        self.reduceSumKernel = ReduceSumKernel()

    def name(self):
        return "Day 1 - Part 1 (GPU)"

    def parse_input(self, input_str: str):
        array1 = []
        array2 = []
        for line in input_str.splitlines():
            strs = [x for x in line.strip().split(' ') if x]
            array1.append(int(strs[0]))
            array2.append(int(strs[1]))

        self.array1 = GPUArray.from_cpu(np.array(array1, dtype=np.int32))
        self.array2 = GPUArray.from_cpu(np.array(array2, dtype=np.int32))

    def solve(self) -> str:
        sorted_array1 = self.bitonicMergeSortKernel(self.array1)
        sorted_array2 = self.bitonicMergeSortKernel(self.array2)
        diff_array = self.elementwiseAbsDiffKernel(sorted_array1, sorted_array2)
        result = self.reduceSumKernel(diff_array)
        return str(result)

class Day01Part02GpuSolution(Solution):

    def __init__(self):
        self.array1 = None
        self.array2 = None
        self.reduceSumKernel = ReduceSumKernel()
        self.reduceMaxKernel = ReduceMaxKernel()
        self.histogramKernel = HistogramKernel()
        self.similarityScoreKernel = SimilarityScoreKernel()

    def name(self):
        return "Day 1 - Part 2 (GPU)"

    def parse_input(self, input_str: str):
        array1 = []
        array2 = []
        for line in input_str.splitlines():
            strs = [x for x in line.strip().split(' ') if x]
            array1.append(int(strs[0]))
            array2.append(int(strs[1]))

        self.array1 = GPUArray.from_cpu(np.array(array1, dtype=np.int32))
        self.array2 = GPUArray.from_cpu(np.array(array2, dtype=np.int32))

    def solve(self) -> str:
        bins = self.reduceMaxKernel(self.array2)
        hist = self.histogramKernel(self.array2, bins)
        scores = self.similarityScoreKernel(self.array1, hist)
        result = self.reduceSumKernel(scores)
        return str(result)