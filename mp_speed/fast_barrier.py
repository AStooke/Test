"""
Fast barrier test

Result: list comprehension is fastest was to release semaphore multiple times.
BarrierLite class is maybe 1.5x slower than the bare semaphores.
"""

import multiprocessing as mp
from timeit import default_timer as timer
import time
import psutil
from ctypes import c_bool


class FastBarrier(object):
    """
    WARNING: Not safe to use one of these in a loop, must use an alternating pair.
    """

    def __init__(self, n):
        self.range_release = range(n - 1)
        self.n = n
        self.enter = mp.Semaphore(n - 1)
        self.hold = mp.Semaphore(0)

    def wait(self):
        if self.enter.acquire(block=False):
            self.hold.acquire()
        else:
            [self.enter.release() for _ in range(self.n - 1)]
            [self.hold.release() for _ in range(self.n - 1)]


def master(n_itr, stop_signal, barrier_1, barrier_2):
    p = psutil.Process()
    p.cpu_affinity([0])
    for i in range(n_itr):
        barrier_1.wait()
        barrier_2.wait()

    barrier_1.wait()
    stop_signal.value = False
    barrier_2.wait()

    print("Master exited at itr count: ", i)


def worker(rank, stop_signal, barrier_1, barrier_2):
    p = psutil.Process()
    p.cpu_affinity([rank + 1])
    i = -1
    while stop_signal.value:
        i += 1
        barrier_1.wait()
        barrier_2.wait()
        if rank == 0:
            time.sleep(0.01)

    print("Rank: ", rank, "exited at itr count: ", i)


n_itr = 100
n_worker = 7

barrier_1 = FastBarrier(n_worker + 1)
barrier_2 = FastBarrier(n_worker + 1)
stop_signal = mp.RawValue(c_bool, True)

worker_procs = [mp.Process(target=worker, args=(rank, stop_signal, barrier_1, barrier_2)) for rank in range(n_worker)]


for p in worker_procs:
    p.start()
master(n_itr, stop_signal, barrier_1, barrier_2)
for p in worker_procs:
    p.join()




































# OLD 2


# It's not safe to use a single pair of these together in a fast loop!
# Must use two pairs, and alternate them!


# class ManyBlocksOneBarrier(object):
#     """ Special one waits on many blockers """

#     def __init__(self, n_blockers):
#         self.range_release = range(n_blockers - 1)
#         self.waiter = mp.Semaphore(0)
#         self.blocker = mp.Semaphore(n_blockers - 1)

#     def wait(self):
#         self.waiter.acquire()

#     def checkin(self):
#         if not self.blocker.acquire(block=False):
#             self.waiter.release()
#             [self.blocker.release() for _ in self.range_release]


# class OneBlocksManyBarrier(object):
#     """ All others wait for one blocker """

#     def __init__(self, n_waiters):
#         self.range_release = range(n_waiters)
#         self.waiter = mp.Semaphore(0)

#     def wait(self):
#         self.waiter.acquire()

#     def release(self):
#         [self.waiter.release() for _ in self.range_release]


# def master(n_itr, mbo_bar_1, obm_bar_1, mbo_bar_2, obm_bar_2, stop_signal):
#     p = psutil.Process()
#     p.cpu_affinity([0])
#     for i in range(n_itr):
#         mbo_bar_1.wait()
#         obm_bar_1.release()
#         mbo_bar_2.wait()
#         if i == n_itr - 1:
#             stop_signal.value = False
#         obm_bar_2.release()

#     # stop_signal.value = False
#     # obm_bar_1.release()
#     # obm_bar_2.release()
#     print("Master exited at itr count: ", i)


# def worker(rank, mbo_bar_1, obm_bar_1, mbo_bar_2, obm_bar_2, stop_signal):
#     p = psutil.Process()
#     p.cpu_affinity([rank + 1])
#     i = -1
#     while stop_signal.value:
#         i += 1
#         mbo_bar_1.checkin()
#         obm_bar_1.wait()
#         mbo_bar_2.checkin()
#         obm_bar_2.wait()

#     print("Rank: ", rank, "exited at itr count: ", i)


# n_itr = 10000
# n_worker = 15

# mbo_bar_1 = ManyBlocksOneBarrier(n_worker)
# mbo_bar_2 = ManyBlocksOneBarrier(n_worker)
# obm_bar_1 = OneBlocksManyBarrier(n_worker)
# obm_bar_2 = OneBlocksManyBarrier(n_worker)
# stop_signal = mp.RawValue(c_bool, True)


# worker_procs = [mp.Process(target=worker, args=(rank, mbo_bar_1, obm_bar_1, mbo_bar_2, obm_bar_2, stop_signal)) for rank in range(n_worker)]


# for p in worker_procs:
#     p.start()
# master(n_itr, mbo_bar_1, obm_bar_1, mbo_bar_2, obm_bar_2, stop_signal)
# for p in worker_procs:
#     p.join()


















## OLD 1

# class BarrierLite(object):

#     def __init__(self, n):
#         self.n = n
#         self.range_n_1 = range(n - 1)
#         # self.range_n_2 = range(n - 2)
#         self.semaphore_1 = [mp.Semaphore(n - 1) for _ in range(2)]
#         self.semaphore_2 = [mp.Semaphore(0) for _ in range(2)]

#     def wait(self, ind):
#         if self.semaphore_1[ind].acquire(block=False):
#             self.semaphore_2[ind].acquire()
#         else:
#             [self.semaphore_1[ind].release() for _ in self.range_n_1]
#             [self.semaphore_2[ind].release() for _ in self.range_n_1]


# def target0(rank, n, sema1a, sema1b, sema2a, sema2b):

#     p = psutil.Process()
#     p.cpu_affinity([rank])

#     # r_n_2 = range(n - 2)
#     r_n_1 = range(n - 1)

#     n_itr = 10
#     t0 = timer()
#     for i in range(n_itr):
#         if rank == 0:
#             time.sleep(.01)
#         print("rank: ", rank, "itr: ", i)
#         if sema1a.acquire(block=False):
#             sema2a.acquire()
#         else:
#             # [sema1.release() for _ in range(n_2)]  # Charge up the outer one first, but leave one less than n.
#             # [sema2.release() for _ in range(n_1)]  # Let everyone ELSE go by but leave at zero.
#             # [sema1a.release() for _ in r_n_1]  # Of the list comprehensions, this is fastest.
#             # [sema2a.release() for _ in r_n_1]
#             # map(lambda x: sema1a.release(), r_n_1)
#             # map(lambda x: sema2a.release(), r_n_1)
#             # sema1a.release(n - 1)  # Doesn't work.
#             # sema2a.release(n - 1)
#         if sema1b.acquire(block=False):
#             sema2b.acquire()
#         else:
#             [sema1b.release() for _ in r_n_1]
#             [sema2b.release() for _ in r_n_1]
#             # map(lambda x: sema1b.release(), r_n_1)
#             # map(lambda x: sema2b.release(), r_n_1)




#             # [sema1.release() for _ in l_n_2]
#             # [sema2.release() for _ in l_n_1]
#             # for _ in r_n_2:
#             #     sema1.release()
#             # for _ in r_n_1:
#             #     sema2.release()
#             # [rel1() for _ in range(n_2)]
#             # [rel2() for _ in range(n_1)]
#             # [r1() for r1 in rel1_all]
#             # [r2() for r2 in rel2_all]
#             # next(m1)
#             # next(m2)
#             # next(mt)
#             # y = map(y1, l_n_2)
#             # y = map(y2, l_n_1)

#     t1 = timer()
#     if rank == 0:
#         print("Time raw sema: ", t1 - t0)

# def target1(rank, n, barrier):
#     p = psutil.Process()
#     p.cpu_affinity([rank])

#     n_itr = 4000
#     t0 = timer()
#     for i in range(n_itr):
#         barrier.wait()
#     t1 = timer()
#     if rank == 0:
#         print("Time barrier: ", t1 - t0)

# def target2(rank, n, barrier):
#     p = psutil.Process()
#     p.cpu_affinity([rank])

#     n_itr = 2000
#     t0 = timer()
#     for i in range(n_itr):
#         barrier.wait(0)
#         barrier.wait(1)
#     t1 = timer()
#     if rank == 0:
#         print("Time barrierlite: ", t1 - t0)


# n = 8

# sema1a = mp.Semaphore(n - 1)
# sema1b = mp.Semaphore(n - 1)
# sema2a = mp.Semaphore(0)
# sema2b = mp.Semaphore(0)
# barrier = mp.Barrier(n)
# barrierlite = BarrierLite(n)

# processes = [mp.Process(target=target2, args=(rank, n, barrierlite)) for rank in range(n)]
# for p in processes:
#     p.start()
# for p in processes:
#     p.join()

# processes = [mp.Process(target=target1, args=(rank, n, barrier)) for rank in range(n)]
# for p in processes:
#     p.start()
# for p in processes:
#     p.join()

# processes = [mp.Process(target=target0, args=(rank, n, sema1a, sema1b, sema2a, sema2b)) for rank in range(n)]
# for p in processes:
#     p.start()
# for p in processes:
#     p.join()


