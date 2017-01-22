
import multiprocessing as mp
from ctypes import c_bool
import time

#################################################################
# It's not safe to use a single pair of these gates in a loop.  #
# Must use two pairs and alternate them.                        #
#################################################################


class ManyBlocksOneGate(object):
    """ Special one waits on many blockers """

    def __init__(self, n_blockers):
        self.range_release = range(n_blockers - 1)
        self.waiter = mp.Semaphore(0)
        self.blocker = mp.Semaphore(n_blockers - 1)

    def wait(self):
        self.waiter.acquire()

    def checkin(self):
        if not self.blocker.acquire(block=False):
            self.waiter.release()
            [self.blocker.release() for _ in self.range_release]

    # def block_wait(self):
    #     if self.block_wait1.acquire(block=False):
    #         self.block_wait2.acquire()
    #     else:
    #         [self.block_wait1.release() for _ in self.range_release]
    #         [self.block_wait2.release() for _ in self.range_release]
    #         self.waiter.release()


class OneBlocksManyGate(object):
    """ All others wait for one blocker """

    def __init__(self, n_waiters):
        self.range_release = range(n_waiters)
        self.waiters = [mp.Semaphore(0) for _ in range(n_waiters)]

    def wait(self, rank):
        self.waiters[rank].acquire()

    def release(self):
        [waiter.release() for waiter in self.waiters]

####################################################################
####################################################################


def master(n_itr, stopper, step_A, act_A, step_B, act_B, step_C, act_C):

    step_A.wait()
    act_A.release()
    step_A.wait()
    act_A.release()
    i = 0
    for _ in range(n_itr):
        i += 1
        step_A.wait()
        # time.sleep(0.01)
        act_A.release()
        # step_B.wait()
        # act_B.release()

    print("Master, i: ", i)

    step_A.wait()
    # step_B.wait()
    stopper.value = False  # Signal to break.
    act_A.release()
    # step_A.wait()
    # act_B.release()
    # step_B.wait()


def worker(rank, stopper, step_A, act_A, step_B, act_B, step_C, act_C):

    step_A.checkin()
    act_A.wait(rank)
    step_A.checkin()

    i = 0
    while stopper.value:
        i += 1
        # act_A.wait(rank)
        if rank == 0:
            time.sleep(0.011)
        # print(rank, i)
        # if rank == 0:
            # time.sleep(0.05)
        step_A.checkin()
        act_A.wait(rank)
        # act_B.wait(rank)
        # step_B.checkin()

    print("Rank: ", rank, "itrs: ", i)


n_itr = 100
n_workers = 8
step_A = ManyBlocksOneGate(n_workers)
step_B = ManyBlocksOneGate(n_workers)
step_C = ManyBlocksOneGate(n_workers)
act_A = OneBlocksManyGate(n_workers)
act_B = OneBlocksManyGate(n_workers)
act_C = OneBlocksManyGate(n_workers)

stopper = mp.RawValue(c_bool, True)

procs = [mp.Process(target=worker, args=(rank, stopper, step_A, act_A, step_B, act_B, step_C, act_C)) for rank in range(n_workers)]

for p in procs:
    p.start()
master(n_itr, stopper, step_A, act_A, step_B, act_B, step_C, act_C)

for p in procs:
    p.join()



# ## OLD: This doesn't work.

# #################################################################
# # It's not safe to use a single pair of these gates in a loop.  #
# # Must use two pairs and alternate them.                        #
# #################################################################

# class ManyBlocksOneGate(object):
#     """ Special one waits on many blockers """

#     def __init__(self, n_blockers):
#         self.range_release = range(n_blockers - 1)
#         self.waiter = mp.Semaphore(0)
#         self.blocker = mp.Semaphore(n_blockers - 1)
#         self.block_wait1 = mp.Semaphore(n_blockers - 1)
#         self.block_wait2 = mp.Semaphore(0)

#     def wait(self):
#         self.waiter.acquire()

#     def checkin(self):
#         if not self.blocker.acquire(block=False):
#             self.waiter.release()
#             [self.blocker.release() for _ in self.range_release]

#     # def block_wait(self):
#     #     if self.block_wait1.acquire(block=False):
#     #         self.block_wait2.acquire()
#     #     else:
#     #         [self.block_wait1.release() for _ in self.range_release]
#     #         [self.block_wait2.release() for _ in self.range_release]
#     #         self.waiter.release()


# class OneBlocksManyGate(object):
#     """ All others wait for one blocker """

#     def __init__(self, n_waiters):
#         self.range_release = range(n_waiters)
#         self.waiter = mp.Semaphore(0)

#     def wait(self):
#         self.waiter.acquire()

#     def release(self):
#         [self.waiter.release() for _ in self.range_release]

# ####################################################################
# ####################################################################
