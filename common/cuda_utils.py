from pycuda.compiler import SourceModule
import math

def compile_cuda_source_file(cu_file_path):
    with open(cu_file_path, 'r') as f:
        cuda_source = f.read()

    cuda_module = SourceModule(cuda_source)
    return cuda_module


def next_pow2(n):
    if n <= 0: return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


def calc_cuda_kernel_layout_1D(n, threads_per_block=256):
    blocks_per_grid = math.ceil(n / threads_per_block)
    return (threads_per_block, 1, 1), (blocks_per_grid, 1, 1)