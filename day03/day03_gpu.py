import pycuda.autoinit
import numpy as np
import re

from common.solution import Solution
from common.gpu_array import GPUArray

from multiply import ElementwiseMultiplyKernel
from day01.reduce_sum import ReduceSumKernel

class Day03Part01GpuSolution(Solution):
    def __init__(self):
        self.array1 = None
        self.array2 = None
        self.mul_pattern = re.compile(r'mul\(([0-9]*),([0-9]*)\)') # two match groups for the operands
        self.multiplyKernel = ElementwiseMultiplyKernel()
        self.reduceSumKernel = ReduceSumKernel()

    def name(self):
        return "Day 3 - Part 1 (GPU)"

    def parse_input(self, input_str: str):
        # technically you could do the pattern matching on the GPU
        # but it's not really a parallel-friendly problem
        # so you'd just end up brute forcing it sequentially
        mul_matches = self.mul_pattern.findall(input_str)

        array1_np = np.empty(len(mul_matches), dtype=np.int32)
        array2_np = np.empty(len(mul_matches), dtype=np.int32)
        for idx, mul_match in enumerate(mul_matches):
            array1_np[idx] = int(mul_match[0])
            array2_np[idx] = int(mul_match[1])

        self.array1 = GPUArray.from_cpu(array1_np)
        self.array2 = GPUArray.from_cpu(array2_np)

    def solve(self):
        mult_array = self.multiplyKernel(self.array1, self.array2)
        result = self.reduceSumKernel(mult_array)
        return str(result)



class Day03Part02GpuSolution(Solution):
    def __init__(self):
        self.array1 = None
        self.array2 = None
        self.mul_pattern = re.compile(r'mul\(([0-9]*),([0-9]*)\)') # two match groups for the operands
        self.dont_pattern = re.compile(r'don\'t\(\).*?(?:do\(\)|$)') # .* needs ? to be non-greedy and avoid taking precedence over the optional group
        self.multiplyKernel = ElementwiseMultiplyKernel()
        self.reduceSumKernel = ReduceSumKernel()

    def name(self):
        return "Day 3 - Part 2 (GPU)"

    def parse_input(self, input_str: str):
        # technically you could do the pattern matching on the GPU
        # but it's not really a parallel-friendly problem
        # so you'd just end up brute forcing it sequentially
        input_stripped = input_str.replace('\n', '') # newlines cause regex to fail when do -> don't crosses into new line
        enabled_matches = re.sub(self.dont_pattern, '', input_stripped) # removes all don't -> do sections
        mul_matches = self.mul_pattern.findall(enabled_matches)

        array1_np = np.empty(len(mul_matches), dtype=np.int32)
        array2_np = np.empty(len(mul_matches), dtype=np.int32)
        for idx, mul_match in enumerate(mul_matches):
            array1_np[idx] = int(mul_match[0])
            array2_np[idx] = int(mul_match[1])

        self.array1 = GPUArray.from_cpu(array1_np)
        self.array2 = GPUArray.from_cpu(array2_np)

    def solve(self):
        mult_array = self.multiplyKernel(self.array1, self.array2)
        result = self.reduceSumKernel(mult_array)
        return str(result)