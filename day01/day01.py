from day01_gpu import *
from day01_cpu import *
from common.solution_runner import SolutionRunner

if __name__ == "__main__":

    with open('input.txt', 'r') as f:
        input_str = f.read()

    runner = SolutionRunner(solutions=[
        Day01Part01GpuSolution(),
        Day01Part02GpuSolution(),
        Day01Part01CpuSolution()
    ])

    runner(input_str)
