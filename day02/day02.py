from day02_gpu import *
from common.solution_runner import SolutionRunner

if __name__ == "__main__":

    with open('input.txt', 'r') as f:
        input_str = f.read()

    runner = SolutionRunner(solutions=[
        Day02Part01GpuSolution(),
        Day02Part02GpuSolution(),
    ])

    runner(input_str)
