from common.solution import Solution


class Day01Part01CpuSolution(Solution):
    def __init__(self):
        self.array1 = None
        self.array2 = None

    def name(self):
        return "Day 1 - Part 1 (CPU)"

    def parse_input(self, input_str: str):
        array1 = []
        array2 = []
        for line in input_str.splitlines():
            strs = [x for x in line.strip().split(' ') if x]
            array1.append(int(strs[0]))
            array2.append(int(strs[1]))

        self.array1 = array1
        self.array2 = array2

    def solve(self) -> str:
        self.array1.sort()
        self.array2.sort()

        sum = 0

        for int1, int2 in zip(self.array1, self.array2):
            sum += abs(int1 - int2)

        return str(sum)
