import pycuda.driver as cuda
import numpy as np


class GPUArray(object):

    def __init__(self, gpudata, size, nbytes, dtype):
        self.gpudata = gpudata
        self.size = size
        self.nbytes = nbytes
        self.dtype = dtype


    @staticmethod
    def empty(size: int, dtype: np.dtype) -> 'GPUArray':
        nbytes = int(size * np.dtype(dtype).itemsize)
        gpudata = cuda.mem_alloc(nbytes)
        return GPUArray(gpudata, size, nbytes, dtype)

    @staticmethod
    def zeros(size: int, dtype: np.dtype) -> 'GPUArray':
        nbytes = int(size * np.dtype(dtype).itemsize)
        gpudata = cuda.mem_alloc(nbytes)
        cuda.memset_d8(gpudata, 0, nbytes)
        return GPUArray(gpudata, size, nbytes, dtype)

    @staticmethod
    def from_cpu(array: np.ndarray) -> 'GPUArray':
        gpudata = cuda.mem_alloc(array.nbytes)
        cuda.memcpy_htod(gpudata, array)
        return GPUArray(gpudata, array.size, array.nbytes, array.dtype)

    def to_cpu(self) -> np.ndarray:
        array = np.empty(self.size, dtype=self.dtype)
        cuda.memcpy_dtoh(array, self.gpudata)
        return array