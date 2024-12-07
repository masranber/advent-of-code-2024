import pycuda.autoinit
import numpy as np

from common.solution import Solution
from common.gpu_array import GPUArray

from word_search import WordSearchKernel

class Day04Part01GpuSolution(Solution):
    def __init__(self):
        self.word_search = None
        self.wordSearchKernel = WordSearchKernel()

    def name(self):
        return "Day 4 - Part 1 (GPU)"

    def parse_input(self, input_str: str):
        lines = input_str.splitlines()
        rows = len(lines)
        cols = len(lines[0])

        # Build the word search from the input
        word_search = np.empty((rows, cols), dtype='S1') # S1 = 1 byte / char
        for i, line in enumerate(lines):
            word_search[i] = list(line)

        # Rotating the word search allows you to iterate vertically in sequence
        word_search_rotated = np.rot90(word_search)

        # Extract the first set of diagonals and unfold them into an iterable 2d array
        # The missing values for the shorter diagonals must be filled with something other than 0x00 or extra matches will appear
        # This is probably a bug in my CUDA kernel but too lazy to find it
        word_search_diag = np.full(((rows + cols), min(rows, cols)), fill_value='.', dtype='S1')
        for i in range(-word_search.shape[0] + 1, word_search.shape[1]):  # Loop through diagonals
            diag = np.diag(word_search, k=i)  # Extract diagonal with offset k
            word_search_diag[i][:len(diag)] = diag

        # Extract the second set of diagonals
        word_search_rotated_diag = np.full(((rows + cols), min(rows, cols)), fill_value='.', dtype='S1')
        for i in range(-word_search_rotated.shape[0] + 1, word_search_rotated.shape[1]):  # Loop through diagonals
            diag = np.diag(word_search_rotated, k=i)  # Extract diagonal with offset k
            word_search_rotated_diag[i][:len(diag)] = diag

        self.rows = rows
        self.cols = cols
        self.rows_diag = (rows + cols)
        self.cols_diag = min(rows, cols)
        self.word_search = GPUArray.from_cpu(word_search.flatten())
        self.word_search_rotated = GPUArray.from_cpu(word_search_rotated.flatten())
        self.word_search_diag = GPUArray.from_cpu(word_search_diag.flatten())
        self.word_search_rotated_diag = GPUArray.from_cpu(word_search_rotated_diag.flatten())


    def solve(self):
        result1 = self.wordSearchKernel(self.word_search, self.rows, self.cols)
        result2 = self.wordSearchKernel(self.word_search_rotated, self.cols, self.rows)
        result3 = self.wordSearchKernel(self.word_search_diag, self.rows_diag, self.cols_diag)
        result4 = self.wordSearchKernel(self.word_search_rotated_diag, self.cols_diag, self.rows_diag)
        return str(result1 + result2 + result3 + result4)


class Day04Part02GpuSolution(Solution):
    def __init__(self):
        self.word_search = None

    def name(self):
        return "Day 4 - Part 2 (GPU)"

    def parse_input(self, input_str: str):
        pass

    def solve(self):
        return ''