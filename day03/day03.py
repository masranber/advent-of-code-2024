from day03_gpu import *
from common.solution_runner import SolutionRunner

if __name__ == "__main__":

    with open('input.txt', 'r') as f:
        input_str = f.read()

    runner = SolutionRunner(solutions=[
        Day03Part01GpuSolution(),
        Day03Part02GpuSolution(),
    ])

    runner(input_str)
