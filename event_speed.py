
import multiprocessing as mp
import psutil

from timeit import default_timer as timer
import time


def main(n_proc=8, itr=100):

    barrier = mp.Barrier(n_proc)
    event = mp.Event()
    semaphore = mp.Semaphore(n_proc)
    semaphore_2 = [mp.Semaphore(n_proc - 1) for _ in range(itr)]
    semaphore_3 = [mp.Semaphore(value=0) for _ in range(itr)]
    lock = mp.Lock()

    processes = [mp.Process(target=worker, args=(rank, itr, barrier, event, semaphore, semaphore_2, semaphore_3, lock)) for rank in range(n_proc)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()


def worker(rank, itr, barrier, event, semaphore, semaphore_2, semaphore_3, lock):

    p = psutil.Process()
    p.cpu_affinity([rank])

    # if rank == 0:
    #     event.set()

    t_bar = 0.
    t_sem = 0.
    t_sev = 0.

    t_0 = timer()
    for i in range(itr):
        t_start = timer()
        barrier.wait()
        t_bar += timer() - t_start
    t_1 = timer()
    # for i in range(itr):
    #     if rank == 0:
    #         event.clear()
    #         time.sleep(0.001)
    #         event.set()
    #     else:
    #         t_0 = timer()
    #         event.wait()
    t_2 = timer()
    # for i in range(itr):
    #     semaphore.acquire(block=False)
    #     semaphore.release()
    t_3 = timer()
    for i in range(itr):
        # semaphore_3.acquire(block=False)  # Should make it so no one can pass in a moment, including self.
        # time.sleep(0.001)
        t_start = timer()
        if semaphore_2[i].acquire(block=False):
            semaphore_3[i].acquire()  # wait here until the last one.
            semaphore_3[i].release()  # let the next one pass.
            semaphore_2[i].release()  # recharge this one for next iter.
            # time.sleep(0.001)
        else:
            # print("rank: {}, didn't get sema_2 in itr: {}".format(rank, i))
            semaphore_3[i].release()  # let the first one pass.
            # time.sleep(0.001)
        t_sem += timer() - t_start
    t_4 = timer()
    for i in range(itr):
        time.sleep(0.01)
        if rank == 0:
            event.clear()
        time.sleep(0.01)
        t_start = timer()
        if semaphore_2[i].acquire(block=False):
            # print("rank: {} acquired semaphore in itr: {}".format(rank, i))
            event.wait()
        else:
            # print("rank: {} set the event in itr: {}".format(rank, i))
            event.set()
        t_sev += timer() - t_start

    # t_bar = t_1 - t_0
    t_eve = t_2 - t_1
    # t_sem = t_3 - t_2
    t_se2 = t_4 - t_3

    with lock:
        print("\nRank: {}".format(rank))
        print("Barrier: {:.4f}, Semaphore: {:.4f}, Semaphore_Event: {:.4f}".format(t_bar, t_sem, t_sev))


if __name__ == "__main__":
    main()
