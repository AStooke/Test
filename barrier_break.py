
import multiprocessing as mp
from threading import BrokenBarrierError
import numpy as np
import time


def main(n_proc=10, n_sit=2, itrs=10):

    barriers = [mp.Barrier(n_proc - n_sit * 2) for _ in range(itrs)]
    barrier_in = mp.Barrier(n_proc - n_sit)
    barrier_outer = mp.Barrier(n_proc)
    lock = mp.Lock()

    processes = [mp.Process(target=worker, args=(rank, n_sit, itrs, barriers, barrier_in, barrier_outer, lock)) for rank in range(n_proc)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()


def worker(rank, n_sit, itrs, barriers, barrier_in, barrier_outer, lock):

    for o in range(2):
        if rank < n_sit:
            sit_out_next = True
        else:
            sit_out_next = False
        barrier_outer.wait()
        if rank == 0:
            [barrier.reset() for barrier in barriers]
            print("\n\nOuter loop iteration: {}".format(o))
        barrier_outer.wait()

        for i in range(itrs):
            if not sit_out_next:
                # Oh, but now rank==0 might be one sitting out. dagrummit!  OK deal with this in a sec.
                barrier_in.wait()  # master, no, SOMEONE needs to set something before everyone proceeds.
                x = 0.001 * np.random.rand()
                time.sleep(x)
                try:
                    j = barriers[i].wait()
                    print("itr: {}, worker released: {}".format(i, rank))
                    if j == 0:  # selects on arbitrary thread that was released
                        time.sleep(0.001)  # slightly annoying, need this because release order not sequential, otherwise will abort some about to release anyway
                        barriers[i].abort()
                except BrokenBarrierError:
                    print("itr: {}, worker passed broken: {}".format(i, rank))
                    sit_out_next = True
            else:
                print("itr: {}, worker sat out: {}".format(i, rank))
                sit_out_next = False

    time.sleep(0.2)
    print("worker finished: {}".format(rank))


if __name__ == "__main__":
    main()
