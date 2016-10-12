"""
Python script for recreating irregularities in time required to write to
multiprocessing shared variable array.
"""

import multiprocessing as mp
import numpy as np
import psutil
from optparse import OptionParser
from timeit import default_timer as timer

# import ipdb


def execute(options):

    n_proc = options.n_proc
    vec_dim = options.vec_dim
    # itrs = options.itrs
    # use_lock = options.lock
    prec = options.prec
    # size_X = options.size_X
    # verbose = options.verbose

    shared_array = np.ctypeslib.as_array(mp.RawArray('d', n_proc * vec_dim)).reshape(n_proc, vec_dim)

    barriers = [mp.Barrier(n_proc) for _ in range(3)]
    lock = mp.Lock()
    # ipdb.set_trace()
    np.set_printoptions(formatter={'float': '{{: 0.{}f}}'.format(prec).format})

    processes = [mp.Process(target=run_worker,
                            args=(rank, shared_array, barriers, lock, options))
                    for rank in range(n_proc)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()


def run_worker(rank, shared_array, barriers, lock, options):

    # n_proc = options.n_proc
    vec_dim = options.vec_dim
    itrs = options.itrs
    # prec = options.prec
    size_X = options.size_X
    verbose = options.verbose
    # misalign = options.misalign
    # typecode = options.typecode

    p = psutil.Process()
    p.cpu_affinity([rank % psutil.cpu_count()])


    if size_X > 0:
        X = np.random.randn(size_X, size_X)
    Y = np.random.randn(vec_dim)
    W = np.random.randn(vec_dim)

    # private_array = np.zeros_like(shared_array)  # compare times for private

    shared_view = shared_array[rank, :]

    t_itrs = []
    t_bar = []
    for i in range(itrs):
        U = Y if (i % 2 == 0) else W
        if size_X > 0:
            Z = X + X  # Optional, to cycle cache.
            Z[0] += 1
        # U = np.random.randn(vec_dim)  # doesn't seem to affect anything.
        barriers[0].wait()
        t_start = timer()
        # A lot of different cases here....hope they are all right.
        shared_view[:] = U
        t_itrs.append(timer() - t_start)
        barriers[1].wait()
        t_bar.append(timer() - t_start)

    t_itrs = np.asarray(t_itrs)  # for printing nicely.
    t_bar = np.asarray(t_bar)
    if rank == 0:
        print("\nOverall Sum: {}, Itr Times: \n{}\n".format(t_bar.sum(), t_bar))
        print("Rank: {}, CPU: , Sum: {}, Itr Times: \n{}\n".format(
            rank, t_itrs.sum(), t_itrs))
        barriers[2].wait()
    else:
        barriers[2].wait()
        if verbose:
            with lock:
                print("Rank: {}, CPU: , Sum: {}, Itr Times: \n{}\n".format(
                    rank, t_itrs.sum(), t_itrs))





parser = OptionParser(
    usage='%prog <options>\nTry to recreate irregular write times to '
    'multiprocessing shared variable array.')

parser.add_option('-n', '--n_proc', action='store', dest='n_proc', default=8,
                  type='int', help='Number of parallel processes to run.')
parser.add_option('-d', '--vec_dim', action='store', dest='vec_dim', default=10000,
                  type='int',
                  help='Length of shared vector (array size = n * v)')
parser.add_option('-i', '--itrs', action='store', dest='itrs', default=100,
                  type='int', help='Number of iterations of writing to shared.')
parser.add_option('-l', '--lock', action='store_true', dest='lock',
                  default=False,
                  help='If True, acquire lock when writing to shared.')
parser.add_option('-p', '--prec', action='store', dest='prec', default=3,
                  type='int', help='Precision of times printed.')
parser.add_option('-X', '--size_X', action='store', dest='size_X', default=2000,
                  type='int',
                  help='Size of matrix (linear dimension) used to cycle cache.')
parser.add_option('-v', '--verbose', action='store_true', dest='verbose',
                  default=False, help='If True, print all worker times.')
parser.add_option('-r', '--rev_idx', action='store_true', dest='rev_idx',
                  default=False, help='If True, reverse indeces of array.')
parser.add_option('-m', '--misalgin', action='store', dest='misalign',
                  default=0, type="int",
                  help='Misalign rows vs cache line boundary by this many elements')
parser.add_option('-t', '--typecode', action='store', dest='typecode',
                  default='d',
                  help='Typecode used in multiprocessing.RawArray()')
parser.add_option('-c', '--chunks', action='store', dest='chunks', default=1,
                  type="int", help='Allocate shared array in multiple chunks')
parser.add_option('--cols_whole', action='store_true', dest='cols_whole',
                  default=False,
                  help='If True, chunk along rows instead of columns')
parser.add_option('-s', '--scramble_cpus', action='store_true',
                  dest='scramble_cpus', default=False,
                  help='If True, randomly permute CPU affinity of ranks')


if __name__ == "__main__":
    options, arguments = parser.parse_args()

    execute(options)
