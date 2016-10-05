"""
Python script for recreating irregularities in time required to write to
multiprocessing shared variable array.
"""

import multiprocessing as mp
import numpy as np
import psutil
from optparse import OptionParser
import sys
from timeit import default_timer as timer


def execute(n_proc, vec_dim, itrs, use_lock, prec, size_X, verbose):

    shared_array = np.frombuffer(mp.RawArray('d', n_proc * vec_dim)
                                 ).reshape((vec_dim, n_proc))
    barriers = [mp.Barrier(n_proc) for _ in range(3)]
    lock = mp.Lock()

    np.set_printoptions(formatter={'float': '{{: 0.{}f}}'.format(prec).format})

    processes = [mp.Process(target=run_worker,
                            args=(rank, vec_dim, shared_array, barriers, lock,
                                  itrs, use_lock, size_X, verbose)
                            )
                    for rank in range(n_proc)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()


def run_worker(rank, vec_dim, shared_array, barriers, lock, itrs, use_lock,
               size_X, verbose):
    p = psutil.Process()
    p.cpu_affinity([rank % psutil.cpu_count()])

    if size_X > 0:
        X = np.random.randn(size_X, size_X)
    Y = np.random.randn(vec_dim)

    # private_array = np.zeros_like(shared_array)  # compare times for private

    t_itrs = []
    t_bar = []
    for _ in range(itrs):
        if size_X > 0:
            Z = X + X  # Optional, to cycle cache.
            Z[0] += 1
        # Y = np.random.randn(vec_dim)  # doesn't seem to affect anything.
        barriers[0].wait()
        t_start = timer()
        if use_lock:
            with lock:
                shared_array[:, rank] = Y
                # private_array[:, rank] = Y
        else:
            shared_array[:, rank] = Y
            # private_array[:, rank] = Y
        t_itrs.append(timer() - t_start)
        barriers[1].wait()
        t_bar.append(timer() - t_start)

    t_itrs = np.asarray(t_itrs)  # for printing nicely.
    t_bar = np.asarray(t_bar)
    if rank == 0:
        print("\nOverall Itr Times: {}\n".format(t_bar))
        print("Rank: {:2}, Itr Times: {}".format(rank, t_itrs))
        barriers[2].wait()
    else:
        barriers[2].wait()
        if verbose:
            with lock:
                print("Rank: {:2}, Itr Times: {}".format(rank, t_itrs))


parser = OptionParser(
    usage='%prog <options>\nTry to recreate irregular write times to '
    'multiprocessing shared variable array.')

parser.add_option('-n', '--n_proc', action='store', dest='n', default=40,
                  type='int', help='Number of parallel processes to run.')
parser.add_option('-d', '--vec_dim', action='store', dest='d', default=300000,
                  type='int',
                  help='Length of shared vector (array size = n * v)')
parser.add_option('-i', '--itrs', action='store', dest='i', default=100,
                  type='int', help='Number of iterations of writing to shared.')
parser.add_option('-l', '--lock', action='store_true', dest='lock',
                  default=False,
                  help='If True, acquire lock when writing to shared.')
parser.add_option('-p', '--prec', action='store', dest='p', default=2,
                  type='int', help='Precision of times printed.')
parser.add_option('-X', '--size_X', action='store', dest='X', default=2000,
                  type='int',
                  help='Size of matrix (linear dimension) used to cycle cache.')
parser.add_option('-v', '--verbose', action='store_true', dest='v',
                  default=False, help='If True, print all worker times.')


if __name__ == "__main__":
    options, arguments = parser.parse_args(sys.argv)

    execute(n_proc=options.n,
            vec_dim=options.d,
            itrs=options.i,
            use_lock=options.lock,
            prec=options.p,
            size_X=options.X,
            verbose=options.v,
            )
