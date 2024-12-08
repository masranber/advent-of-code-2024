import pycuda.autoinit
import re

from common.solution import Solution
from common.solution_runner import SolutionRunner
from common.gpu_array import GPUArray

from solve_calibration import *


class Day07Part01GpuSolution(Solution):
    def __init__(self):
        self.word_search = None
        self.calibration_matrix = None
        self.rows = 0
        self.cols = 0
        self.calibration_pattern = re.compile(r": | ")
        self.solve_calibration_kernel = SolveCalibrationPart1Kernel()

    def name(self):
        return "Day 7 - Part 1 (GPU)"

    def parse_input(self, input_str: str):
        calibrations = []
        max_calibration_length = 0
        for line in input_str.splitlines():
            operands = [int(x) for x in re.split(self.calibration_pattern, line)]
            max_calibration_length = max(max_calibration_length, len(operands))
            calibrations.append(operands)
        calibrations_matrix = np.zeros((len(calibrations), max_calibration_length), dtype=np.int64)
        for i, calibration in enumerate(calibrations):
            calibrations_matrix[i][:len(calibration)] = calibration

        self.calibration_matrix = GPUArray.from_cpu(calibrations_matrix.flatten())
        self.rows = calibrations_matrix.shape[0]
        self.cols = calibrations_matrix.shape[1]

    def solve(self):
        result = self.solve_calibration_kernel(self.calibration_matrix, self.rows, self.cols)
        return str(result)


class Day07Part02GpuSolution(Solution):
    def __init__(self):
        self.word_search = None
        self.calibration_matrix = None
        self.rows = 0
        self.cols = 0
        self.calibration_pattern = re.compile(r": | ") # matches ": " or " "
        self.solve_calibration_kernel = SolveCalibrationPart2Kernel()

    def name(self):
        return "Day 7 - Part 2 (GPU)"

    def parse_input(self, input_str: str):
        calibrations = []
        max_calibration_length = 0
        for line in input_str.splitlines():
            operands = [int(x) for x in re.split(self.calibration_pattern, line)]
            max_calibration_length = max(max_calibration_length, len(operands))
            calibrations.append(operands)
        calibrations_matrix = np.zeros((len(calibrations), max_calibration_length), dtype=np.int64)
        for i, calibration in enumerate(calibrations):
            calibrations_matrix[i][:len(calibration)] = calibration

        self.calibration_matrix = GPUArray.from_cpu(calibrations_matrix.flatten())
        self.rows = calibrations_matrix.shape[0]
        self.cols = calibrations_matrix.shape[1]

    def solve(self):
        result = self.solve_calibration_kernel(self.calibration_matrix, self.rows, self.cols)
        return str(result)



if __name__ == "__main__":

    with open('input.txt', 'r') as f:
        input_str = f.read()

    runner = SolutionRunner(solutions=[
        Day07Part01GpuSolution(),
        Day07Part02GpuSolution(),
    ])

    runner(input_str)
