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
    rev_idx = options.rev_idx
    misalign = options.misalign
    typecode = options.typecode

    m = n_proc if not rev_idx else vec_dim
    n = vec_dim if not rev_idx else n_proc
    shared_array = row_aligned_array(m, n, typecode=typecode, misalign=misalign)
    barriers = [mp.Barrier(n_proc) for _ in range(3)]
    lock = mp.Lock()

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
    use_lock = options.lock
    # prec = options.prec
    size_X = options.size_X
    verbose = options.verbose
    rev_idx = options.rev_idx
    # misalign = options.misalign
    # typecode = options.typecode

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
                if rev_idx:
                    shared_array[:vec_dim, rank] = Y
                else:
                    shared_array[rank, :vec_dim] = Y
                # private_array[:, rank] = Y
        else:
            if rev_idx:
                shared_array[:vec_dim, rank] = Y
            else:
                shared_array[rank, :vec_dim] = Y
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


def row_aligned_array(m, n, typecode='d', alignment=64, misalign=False):
    """
    m: number of rows of data
    n: number of columns of data (will be padded)
    typecode: first argument sent to mp.RawArray() (could also be a c_type)
    alignment: cache coherency line size (bytes)
    misalign: if True, deliberately misalign rows with cache line boundaries.
    """
    elem_size = np.ctypeslib.as_array(mp.RawArray(typecode, 1)).itemsize
    assert alignment % elem_size == 0, "alignment must be multiple of elem_size"
    cache_line = alignment // elem_size  # units: elements
    n_pad = n + cache_line - (n % cache_line)
    num_elements = n_pad * m
    x = np.ctypeslib.as_array(mp.RawArray(typecode, num_elements + cache_line))
    assert x.ctypes.data % elem_size == 0, "multiprocessing.RawArray() did  \
        not provide element-aligned memory (re-write this fcn to be byte-based)"
    start_idx = -x.ctypes.data % alignment
    if misalign:
        start_idx += 1
    z = x[start_idx:start_idx + num_elements].reshape(m, n_pad)
    assert misalign != (z.ctypes.data % alignment == 0), "array alignment did not work"
    if m > 1:
        assert misalign != (z[1, :].ctypes.data % alignment == 0), "row alignment did not work"
    return z


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
parser.add_option('-m', '--misalgin', action='store_true', dest='misalign',
                  default=False, help='If True, misalign rows vs cache line '
                  'boundary (default is row aligned) ')
parser.add_option('-t', '--typecode', action='store', dest='typecode',
                  default='d',
                  help='Typecode used in multiprocessing.RawArray()')


if __name__ == "__main__":
    options, arguments = parser.parse_args()

    execute(options)
