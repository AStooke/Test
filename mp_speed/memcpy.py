import psutil

import numpy as np
import multiprocessing as mp
from timeit import default_timer as timer

N_WORKER = 16
SHARED_DEST = True
SHAPE = (320000, 105, 80)
def copy_worker(source, dest, copy_slice, barriers):
    p = psutil.Process()
    p.cpu_affinity(list(range(20, 40)))
    # t_start = timer()
    # source[0] = 1
    # t_first = timer()
    # source[1] = 1
    # t_second = timer()
    # print("worker time first source touch: {}, second source touch: {}".format(t_first-t_start, t_second-t_first))
    barriers[0].wait()
    barriers[1].wait()
    dest[copy_slice] = source[copy_slice]
    x = dest[copy_slice.start]
    print(x[0, 0])
    barriers[2].wait()
    
p = psutil.Process()
p.cpu_affinity([0])
barriers = [mp.Barrier(N_WORKER + 1) for _ in range(3)]
print("Allocating source and destination arrays")
t_alloc = timer()
dtype = np.dtype(np.float32)
nbytes = int(np.prod(SHAPE)) * dtype.itemsize
buf = np.empty(nbytes + 64, dtype=np.uint8)
start_index = -buf.ctypes.data % 64
source = buf[start_index:start_index + nbytes].view(dtype).reshape(SHAPE)

# alloc_shape = list(SHAPE)
# alloc_shape[0] += 4
# source = np.ones(alloc_shape, dtype='float32')
# source = source[1:]
t_source = timer()
print("done allocating source: {} seconds".format(t_source - t_alloc))
if SHARED_DEST:
    dest_mp = mp.RawArray('f', int(np.prod(SHAPE)))
    dest = np.ctypeslib.as_array(dest_mp).reshape(SHAPE)
else:
    buf = np.empty(nbytes + 64, dtype=np.uint8)
    start_index = -buf.ctypes.data % 64
    dest = buf[start_index:start_index + nbytes].view(dtype).reshape(SHAPE)
    # dest = np.empty(SHAPE, dtype='float32')
print("Done allocating dest: {} seconds.".format(timer() - t_source))
print("Source -- Dest Alignment % 64: {}, {}".format(source.ctypes.data % 64, dest.ctypes.data % 64))
size_per_worker = SHAPE[0] // N_WORKER
idx = 0
slice_list = list()
for worker in range(N_WORKER):
    slice_list.append(slice(idx, idx + size_per_worker))
    idx += size_per_worker

workers = [mp.Process(target=copy_worker,
                      args=(source, dest, copy_slice, barriers))
                for copy_slice in slice_list]

print("Starting {} workers.".format(N_WORKER))
for w in workers: w.start()

t_start = timer()
barriers[0].wait()
print("Starting copy in workers.")
t_start=timer()
barriers[1].wait()
barriers[2].wait()
t_end = timer()
print("Copy time: {}".format(t_end - t_start))
for w in workers: w.join()
# assert np.allclose(source, dest)



