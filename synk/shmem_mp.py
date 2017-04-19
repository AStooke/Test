
import synkhronos
from synkhronos import shmemarray
from synkhronos import data_builder
synkhronos.fork()
import multiprocessing as mp
import time


def worker(shmem_array):
    print("worker arr start: \n", shmem_array)
    shmem_array[:] = 0.

def worker_synk(synk_data):
    print("worker synk_data start: \n", synk_data.data)
    synk_data[:] = 0.



arr = shmemarray.NpShmemArray(dtype='float32', shape=(10, 10), tag=str(1), create=True)
arr[:] = 1.
s_dat = data_builder.data(dtype='float32', shape=(10, 10))
s_dat[:] = 1.




p = mp.Process(target=worker, args=(arr,))
p.start()
p.join()

p2 = mp.Process(target=worker_synk, args=(s_dat,))
p2.start()
p2.join()
time.sleep(1)
print("master arr end: \n", arr)
print("master synk_data at end: \n", s_dat.data)

# YES IT WORKS!
