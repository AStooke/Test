"""
Python script for recreating irregularities in time required to write to
multiprocessing shared variable array.
"""

import multiprocessing as mp
import numpy as np
import psutil
from optparse import OptionParser
from timeit import default_timer as timer

import ipdb


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
    chunk = options.chunk

    if not chunk:
        m = n_proc if not rev_idx else vec_dim
        n = vec_dim if not rev_idx else n_proc
        shared_array = row_aligned_array(m, n, typecode=typecode, misalign=misalign)
        vb_idx = None
    else:
        # ignore rev_idx for the moment.
        # n_elm_worker = -(-vec_dim // n_proc)  # ceiling div
        # vec_boundaries = [n_elm_worker * i for i in range(n_proc + 1)]
        # vec_boundaries[-1] = vec_dim
        # vb_idx = [(vec_boundaries[i], vec_boundaries[i + 1]) for i in range(n_proc)]
        # shared_array = [row_aligned_array(n_proc, n_elm_worker, typecode=typecode, misalign=misalign) for _ in range(n_proc)]
        shared_array, vb_idx = chunked_array(n_proc, vec_dim, typecode)
        # ipdb.set_trace()
    barriers = [mp.Barrier(n_proc) for _ in range(3)]
    lock = mp.Lock()
    # ipdb.set_trace()
    np.set_printoptions(formatter={'float': '{{: 0.{}f}}'.format(prec).format})

    processes = [mp.Process(target=run_worker,
                            args=(rank, shared_array, vb_idx, barriers, lock, options))
                    for rank in range(n_proc)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()


def run_worker(rank, shared_array, vb_idx, barriers, lock, options):

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
    chunk = options.chunk

    p = psutil.Process()
    p.cpu_affinity([rank % psutil.cpu_count()])

    if size_X > 0:
        X = np.random.randn(size_X, size_X)
    Y = np.random.randn(vec_dim)
    W = np.random.randn(vec_dim)

    # private_array = np.zeros_like(shared_array)  # compare times for private

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
        if use_lock:
            lock.acquire()
            t_start_worker = timer()
        else:
            t_start_worker = t_start
        if chunk:
            for array, vb in zip(shared_array, vb_idx):
                array[rank, :] = U[vb[0]:vb[1]]  # :(vb[1] - vb[0])
        elif rev_idx:
            shared_array[:vec_dim, rank] = U
        else:
            shared_array[rank, :vec_dim] = U
        # private_array[:, rank] = Y
        if use_lock:
            lock.release()
        t_itrs.append(timer() - t_start_worker)
        barriers[1].wait()
        t_bar.append(timer() - t_start)

    t_itrs = np.asarray(t_itrs)  # for printing nicely.
    t_bar = np.asarray(t_bar)
    if rank == 0:
        print("\nOverall Sum: {}".format(t_bar.sum()))
        print("\nRank: {}, Sum: {}".format(rank, t_itrs.sum()))
        print("\nOverall Itr Times: \n{}\n".format(t_bar))
        print("Rank: {}, Itr Times: \n{}\n".format(rank, t_itrs))
        barriers[2].wait()
    else:
        barriers[2].wait()
        if verbose:
            with lock:
                print("Rank: {}, Sum: {}".format(rank, t_itrs.sum()))
                print("Rank: {}, Itr Times: \n{}\n".format(rank, t_itrs))


def chunked_array(m, n, typecode='d'):
    """
    m: number of rows of data (will also be the number of chunks)
    n: number of columns of data
    typecode: used in multiprocessing.RawArray() (could also be a c_type)

    returns: a list of m numpy arrays referencing multiprocessing raw arrays,
    and a list of tuples containing the chunk boundaries
    """
    chunk_size = -(-n // m)  # (ceiling)
    bounds = [chunk_size * i for i in range(m + 1)]
    bounds[-1] = n
    boundary_pairs = [(bounds[i], bounds[i + 1]) for i in range(m)]
    chunked_array = [np.ctypeslib.as_array(mp.RawArray(typecode, m * chunk_size)).reshape(m, chunk_size) for i in range(m)]
    return chunked_array, boundary_pairs


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
parser.add_option('-c', '--chunk', action='store_true', dest='chunk',
                  default=False,
                  help='If True, allocate shared array in multiple chunks')


if __name__ == "__main__":
    options, arguments = parser.parse_args()

    execute(options)
