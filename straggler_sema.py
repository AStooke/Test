
import multiprocessing as mp
# from threading import BrokenBarrierError
import numpy as np
import time


def main(n_proc=10, n_sit=2, itrs=10):

    n_pass = n_proc - n_sit * 2

    # barriers = [mp.Barrier(n_proc - n_sit * 2) for _ in range(itrs)]
    barrier_in = [mp.Barrier(n_proc - n_sit) for _ in range(itrs)]
    barrier_outer = mp.Barrier(n_proc)
    barrier_first = mp.Barrier(n_proc - n_sit)
    barrier_rest = mp.Barrier(n_pass)
    sema_first = mp.Semaphore(value=n_proc - n_sit)
    sema_rest = [mp.Semaphore(value=n_pass) for _ in range(itrs)]

    lock = mp.Lock()

    processes = [mp.Process(target=worker, args=(rank, n_sit, n_pass, itrs, barrier_in, barrier_outer, barrier_first, barrier_rest, sema_first, sema_rest, lock)) for rank in range(n_proc)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()


def worker(rank, n_sit, n_pass, itrs, barrier_in, barrier_outer, barrier_first, barrier_rest, sema_first, sema_rest, lock):

    for o in range(2):
        if rank < n_sit:
            sit_out_next = True
        else:
            sit_out_next = False
        barrier_outer.wait()
        # barrier_outer.wait()

        for i in range(itrs):
            if not sit_out_next:
                # Oh, but now rank==0 might be one sitting out. dagrummit!  OK deal with this in a sec.
                barrier_in[i].wait()  # master, no, SOMEONE needs to set something before everyone proceeds.
                x = 0.01 * np.random.rand()
                time.sleep(x)
                sema_value = sema_rest[i].acquire(block=False)
                if sema_value:
                    print("itr: {}, worker passed True semaphore: {}".format(i, rank))
                    barrier_rest.wait()
                    print("itr: {}, worker released: {}".format(i, rank))
                else:
                    print("itr: {}, worker passed False semaphore: {}".format(i, rank))
                    sit_out_next = True
            else:
                print("itr: {}, worker sat out: {}".format(i, rank))
                sit_out_next = False

        barrier_outer.wait()
        if rank == 0:
            # [barrier.reset() for barrier in barriers]
            [sema.release() for _ in range(n_pass) for sema in sema_rest]
            print("\n\nOuter loop iteration: {}".format(o))
    time.sleep(0.2)
    print("worker finished: {}".format(rank))


if __name__ == "__main__":
    main()
