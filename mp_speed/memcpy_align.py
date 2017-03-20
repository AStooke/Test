
import numpy as np
from timeit import default_timer as timer


SHAPE = (80000, 100, 100)
DTYPE = 'float64'
OFFSET = 0


def byte_aligned(shape, dtype='float64', alignment=64, offset=0):
    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape)) * dtype.itemsize
    buf = np.empty(nbytes + alignment, dtype=np.uint8)
    s_idx = -buf.ctypes.data % alignment + offset
    return buf[s_idx:s_idx + nbytes].view(dtype).reshape(shape)

def aln(arr, alignment=64):
    return arr.ctypes.data % alignment


dtype = np.dtype(DTYPE)
size = int(np.prod(SHAPE))
print("Size of arrays: {} MB".format(dtype.itemsize * size // 1024 // 1024))

# x = np.ones(SHAPE, dtype=DTYPE)
# y = np.empty(SHAPE, dtype=DTYPE)
# print("x alignment % 64: {}, y alignment: {}".format(aln(x), aln(y)))

w = byte_aligned(SHAPE, dtype=DTYPE, offset=OFFSET)
w[:] = 1
z = byte_aligned(SHAPE, dtype=DTYPE, offset=16)
print("w alignment % 64: {}, z alignment: {}".format(aln(w), aln(z)))


# t_start = timer()
# y[:] = x
# t_end = timer()
# print("copy x into y: {} seconds".format(t_end - t_start))

t_start = timer()
z[:] = w
t_end = timer()
print("copy z into w: {} seconds".format(t_end - t_start))
