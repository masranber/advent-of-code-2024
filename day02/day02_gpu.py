import pycuda.autoinit
import numpy as np

from common.solution import Solution
from common.gpu_array import GPUArray
from safe_report import SafeReportKernel
from day01.reduce_sum import ReduceSumKernel

class Day02Part01GpuSolution(Solution):
    def __init__(self):
        self.cols = 0
        self.reports = None
        self.safeReportKernel = SafeReportKernel()
        self.reduceSumKernel = ReduceSumKernel()

    def name(self):
        return "Day 2 - Part 1 (GPU)"

    def parse_input(self, input_str: str):
        self.cols = 9 # ideally find a way to compute the longest row but for now this works
        rows = len(input_str.splitlines())
        reports_matrix = np.zeros((rows, self.cols), dtype=np.int32)
        for i, line in enumerate(input_str.splitlines()):
            levels = line.split()
            reports_matrix[i][:len(levels)] = levels # shorter rows will be padded with 0s, which the kernel skips

        reports_flat = reports_matrix.flatten()
        self.reports = GPUArray.from_cpu(reports_flat)

    def solve(self):
        safe_results = self.safeReportKernel(self.reports, self.cols)
        result = self.reduceSumKernel(safe_results)
        return str(result)



class Day02Part02GpuSolution(Solution):
    def __init__(self):
        self.cols = 0
        self.reports = None
        self.safeReportKernel = SafeReportKernel()
        self.reduceSumKernel = ReduceSumKernel()

    def name(self):
        return "Day 2 - Part 2 (GPU)"

    def parse_input(self, input_str: str):
        self.cols = 8 # ideally find a way to compute the longest row but for now this works
        rows = len(input_str.splitlines())
        reports_matrix = np.zeros((rows, self.cols), dtype=np.int32)
        self.reports_2d = reports_matrix
        for i, line in enumerate(input_str.splitlines()):
            levels = line.split()
            reports_matrix[i][:len(levels)] = levels # shorter rows will be padded with 0s, which the kernel skips

        reports_flat = reports_matrix.flatten()
        self.reports = GPUArray.from_cpu(reports_flat)

    def solve(self):
        # skip each column, check which reports are safe, and accumulate the results in the existing buffer
        # you can skip the case where skip_col=-1 (don't skip any columns like in part 1)
        # because the results for each pass are accumulated (you basically get it for free)
        safe_results = self.safeReportKernel(self.reports, self.cols, skip_col=0)
        for skip_col in range(1, self.cols - 1):
            self.safeReportKernel(self.reports, self.cols, safe_results, skip_col)
        result = self.reduceSumKernel(safe_results)
        return str(result)