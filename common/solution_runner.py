import time


class SolutionRunner(object):

    def __init__(self, solutions):
        self.solutions = solutions

    def __call__(self, puzzle_input):
        for solution in self.solutions:
            print(f'---- {solution.name()} ----')

            parse_start = time.perf_counter()
            solution.parse_input(puzzle_input)

            solve_start = time.perf_counter()

            result = solution.solve()
            solve_end = time.perf_counter()

            parse_time = solve_start - parse_start
            solve_time = solve_end - solve_start

            print('Result: ', result)
            print('Benchmark:')
            print(f'\tParse Input: {parse_time * 1000:.2f} ms')
            print(f'\tSolve:       {solve_time * 1000:.2f} ms')
            print()
