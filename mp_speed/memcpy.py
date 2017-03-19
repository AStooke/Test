

import numpy as np
import multiprocessing as mp
from timeit import default_timer as timer
import time

N_WORKER = 8
SHAPE = (64000, 105, 80)
def copy_worker(source, dest, copy_slice, barriers):
    barriers[0].wait()
    dest[copy_slice] = source[copy_slice]
    barriers[1].wait()


barriers = [mp.Barrier(N_WORKER + 1) for _ in range(2)]
source = np.ones(SHAPE, dtype='float64')

dest_mp = mp.RawArray('d', int(np.prod(SHAPE)))
dest = np.ctypeslib.as_array(dest_mp).reshape(SHAPE)
dest_local = np.empty(SHAPE, dtype='float64')
size_per_worker = SHAPE[0] // N_WORKER
idx = 0
slice_list = list()
for worker in range(N_WORKER):
    slice_list.append(slice(idx, idx + size_per_worker))
    idx += size_per_worker

workers = [mp.Process(target=copy_worker,
                      args=(source, dest, copy_slice, barriers))
                for copy_slice in slice_list]

for w in workers: w.start()

time.sleep(1)
t_start = timer()
barriers[0].wait()
barriers[1].wait()
t_end = timer()
print("Copy time: {}".format(t_end - t_start))
assert np.allclose(source, dest)



