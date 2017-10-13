
import multiprocessing as mp
import numpy as np
import time

NUM_TEST = 100
DELTA_T = 1.5


def worker(barrier, semaphore, worker_times):
    barrier.wait()
    for i in range(NUM_TEST):
        semaphore.acquire()
        worker_times[i] = time.time()


def master(barrier, semaphore, master_times):
    barrier.wait()
    for i in range(NUM_TEST):
        time.sleep(DELTA_T)
        master_times[i] = time.time()
        semaphore.release()


worker_times = np.ctypeslib.as_array(mp.RawArray('d', NUM_TEST))
master_times = np.ctypeslib.as_array(mp.RawArray('d', NUM_TEST))
semaphore = mp.Semaphore(0)
barrier = mp.Barrier(2)

w = mp.Process(target=worker, args=(barrier, semaphore, worker_times))
m = mp.Process(target=master, args=(barrier, semaphore, master_times))

w.start()
m.start()

w.join()
m.join()

delta_times = worker_times - master_times

print("mean delta: ", np.mean(delta_times))
print("max delta: ", np.max(delta_times))
print("min delta: ", np.min(delta_times))

