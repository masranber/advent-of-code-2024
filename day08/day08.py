import pycuda.autoinit
import numpy as np

from common.solution import Solution
from common.solution_runner import SolutionRunner
from common.gpu_array import GPUArray

from antenna_antinodes import *
from day01.reduce_sum import *

class Day08Part01GpuSolution(Solution):
    def __init__(self):
        self.antenna_map = None
        self.rows = None
        self.cols = None
        self.antenna_antinodes_kernel = AntennaAntinodesKernel()
        self.sum_kernel = ReduceSumKernel()

    def name(self):
        return "Day 8 - Part 1 (GPU)"

    def parse_input(self, input_str: str):
        lines = input_str.splitlines()
        antenna_map = np.empty((len(lines), len(lines[0])), dtype='S1')
        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                antenna_map[i][j] = char
        self.antenna_map = GPUArray.from_cpu(antenna_map.flatten())
        self.rows, self.cols = antenna_map.shape
        #print(antenna_map)

    def solve(self):
        antinodes_map = self.antenna_antinodes_kernel(self.antenna_map, self.rows, self.cols)
        sum = self.sum_kernel(antinodes_map)
        #arr = antinodes_map.to_cpu().reshape(self.rows, self.cols)
        #for row in arr:
            #print(''.join(map(str, row)))
        return str(sum)


class Day08Part02GpuSolution(Solution):
    def __init__(self):
        self.antenna_map = None
        self.rows = None
        self.cols = None
        self.antenna_antinodes_kernel = AntennaAntinodesHarmonicKernel()
        self.sum_kernel = ReduceSumKernel()

    def name(self):
        return "Day 8 - Part 2 (GPU)"

    def parse_input(self, input_str: str):
        lines = input_str.splitlines()
        antenna_map = np.empty((len(lines), len(lines[0])), dtype='S1')
        for i, line in enumerate(lines):
            for j, char in enumerate(line):
                antenna_map[i][j] = char
        self.antenna_map = GPUArray.from_cpu(antenna_map.flatten())
        self.rows, self.cols = antenna_map.shape
        #print(antenna_map)

    def solve(self):
        antinodes_map = self.antenna_antinodes_kernel(self.antenna_map, self.rows, self.cols)
        sum = self.sum_kernel(antinodes_map)
        #arr = antinodes_map.to_cpu().reshape(self.rows, self.cols)
        #for row in arr:
            #print(''.join(map(str, row)))
        return str(sum)



if __name__ == "__main__":

    with open('input.txt', 'r') as f:
        input_str = f.read()

    runner = SolutionRunner(solutions=[
        Day08Part01GpuSolution(),
        Day08Part02GpuSolution(),
    ])

    runner(input_str)
