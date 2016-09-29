
"""
CONCLUSION:
In Pthon3, the built-in barrier is much faster than the custom barrier (4x)
"""


import multiprocessing as mp
from multiprocessing import Semaphore
from multiprocessing.managers import BaseManager
import time

import gtimer as gt



def target(barrier):
    x = 0
    loop = gt.timed_loop('bar')
    for _ in range(100):
        next(loop)
        x += 1
        time.sleep(0.1)
        gt.stamp('before_bar')
        barrier.wait()
        gt.stamp('after_bar')
    loop.exit()
    gt.stop()
    print(gt.report(include_itrs=False, include_stats=False))


class Barrier:

    def __init__(self, n):
        self.n = n  # number of threads subject to barrier
        self.count = 0  # number of threads at the barrier
        self.mutexIn = Semaphore(1)  # initialize as 'available'
        self.mutexOut = Semaphore(1)
        self.barrier = Semaphore(0)  # initialize as 'blocked'

    def wait(self):
        self.mutexIn.acquire()  # one at a time
        self.count += 1  # check in
        if self.count < self.n:  # if not the last one
            self.mutexIn.release()  # let others check in
        else:  # if the last one
            self.barrier.release()  # begin barrier release chain
                                    # mutexIn stays acquired, no re-entry yet

        self.barrier.acquire()  # wait until all have checked in
        self.barrier.release()  # all threads pass simultaneously

        self.mutexOut.acquire()  # one at a time
        self.count -= 1  # checkout
        if self.count == 0:  # if the last one
            self.barrier.acquire()  # block the barrier
            self.mutexIn.release()  # allow re-entry
        self.mutexOut.release()  # allow next checkout


class BarrierManager(BaseManager):
    pass

BarrierManager.register('Barrier',Barrier)
barrier_mgr = BarrierManager()
barrier_mgr.start()


n_par = 4

# barrier = barrier_mgr.Barrier(n_par)
barrier = mp.Barrier(4)

processes = [mp.Process(target=target, args=(barrier,)) for _ in range(n_par)]

for p in processes:
    p.start()
gt.stamp('spawn')

for p in processes:
    p.join()
gt.stop('join')

print(gt.report())

