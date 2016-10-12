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
    # misalign = options.misalign
    typecode = options.typecode
    chunks = options.chunks
    rows_whole = not options.cols_whole
    scramble_cpus = options.scramble_cpus

    m = n_proc if not rev_idx else vec_dim
    n = vec_dim if not rev_idx else n_proc
    shared_array, vb_pairs = chunked_2d_shared_array(m, n, chunks, rows_whole, typecode)

    barriers = [mp.Barrier(n_proc) for _ in range(3)]
    lock = mp.Lock()
    # ipdb.set_trace()
    np.set_printoptions(formatter={'float': '{{: 0.{}f}}'.format(prec).format})

    # Scramble the affinities, so that adjacent data overlap is not in same L2 cache.
    ranks = list(range(n_proc))
    cpus = ranks if not scramble_cpus else np.random.permutation(ranks)

    processes = [mp.Process(target=run_worker,
                            args=(rank, cpu, shared_array, vb_pairs, barriers, lock, options))
                    for rank, cpu in zip(ranks, cpus)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()


def run_worker(rank, cpu, shared_array, vb_pairs, barriers, lock, options):

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
    chunks = options.chunks
    rows_whole = not options.cols_whole

    p = psutil.Process()
    # p.cpu_affinity([rank % psutil.cpu_count()])
    p.cpu_affinity([cpu])

    if rows_whole and not rev_idx:
        rows_per_chunk = shared_array[0].shape[0]
        a_ind_0 = rank // rows_per_chunk
        a_ind_1 = rank % rows_per_chunk
    if not rows_whole and rev_idx:
        cols_per_chunk = shared_array[0].shape[1]
        a_ind_0 = rank // cols_per_chunk
        a_ind_2 = rank % cols_per_chunk

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
        # A lot of different cases here....hope they are all right.
        if chunks > 1:
            if not rows_whole:
                if not rev_idx:
                    for array, vb in zip(shared_array, vb_pairs):
                        array[rank, :(vb[1] - vb[0])] = U[vb[0]:vb[1]]
                else:
                    shared_array[a_ind_0][:vec_dim, a_ind_2] = U
            else:
                if not rev_idx:
                    shared_array[a_ind_0][a_ind_1, :vec_dim] = U
                else:
                    for array, vb in zip(shared_array, vb_pairs):
                        arra[:(vb[1] - vb[0]), rank] = U[vb[0]:vb[1]]
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
        print("\nOverall Sum: {}, Itr Times: \n{}\n".format(t_bar.sum(), t_bar))
        print("Rank: {}, CPU: {}, Sum: {}, Itr Times: \n{}\n".format(
            rank, cpu, t_itrs.sum(), t_itrs))
        barriers[2].wait()
    else:
        barriers[2].wait()
        if verbose:
            with lock:
                print("Rank: {}, CPU: {}, Sum: {}, Itr Times: \n{}\n".format(
                    rank, cpu, t_itrs.sum(), t_itrs))


def chunked_2d_shared_array(m, n, chunks=2, rows_whole=True, typecode='d', misalign=0, size_tight=False):
    """
    m: number of rows of data
    n: number of columns of data
    chunks: number of chunks to use
    rows_whole: if True, chunk along dimension 0, else chunk along dimension 1
    typecode: used in multiprocessing.RawArray() (could also be a c_type)
    misalign: if True, deliberately misalign rows with cache line boundaries
    size_tight: if True, last chunk might not be same shape as other chunks

    returns: a list of m numpy arrays referencing multiprocessing raw arrays,
    and a list of tuples containing the chunk boundaries
    """
    if not isinstance(chunks, int):
        raise TypeError("param chunks must be an integer")
    chunk_dim_size = m if rows_whole else n
    chunks = min(chunks, chunk_dim_size)
    chunk_size = -(-chunk_dim_size // chunks)  # (ceiling)
    if rows_whole:
        chunk_shape = (chunk_size, n)
    else:
        chunk_shape = (m, chunk_size)
    chunk_flat_size = chunk_shape[0] * chunk_shape[1]

    # chunked_array = [row_aligned_array(chunk_shape[0], chunk_shape[1], typecode, misalign=misalign)
    #     for _ in range(chunks)]

    chunked_array = [np.ctypeslib.as_array(
        mp.RawArray(typecode, chunk_flat_size)).reshape(chunk_shape)
        for _ in range(chunks)]

    bounds = [chunk_size * i for i in range(chunks + 1)]
    bounds[-1] = chunk_dim_size
    boundary_pairs = [(bounds[i], bounds[i + 1]) for i in range(chunks)]

    if chunks == 1:
        chunked_array = chunked_array[0]  # don't return list if length is 1.
    elif size_tight:
        last_size = bounds[-1] - bounds[-2]
        if rows_whole:
            last_shape = (last_size, n)
        else:
            last_shape = (m, last_size)
        last_flat_size = last_shape[0] * last_shape[1]

        # Make a reduced (but contiguous) view into the last chunk.
        chunked_array[-1] = chunked_array[-1].reshape(
            chunk_flat_size)[:last_flat_size].reshape(last_shape)

        # Alternatively, make a new chunk and ignore the previous allocation.
        # chunked_array[-1] = np.ctypeslib.as_array(
        #     mp.RawArray(typecode, last_flat_size)).reshape(last_shape)

        # chunked_array[-1] = row_aligned_array(last_shape[0], last_shape[1], typecode, misalign=misalign)

    return chunked_array, boundary_pairs


# def row_aligned_array(m, n, typecode='d', alignment=64, misalign=0):
#     """
#     m: number of rows of data
#     n: number of columns of data (will be padded)
#     typecode: first argument sent to mp.RawArray() (could also be a c_type)
#     alignment: [bytes] cache coherency line size
#     misalign: [elements] deliberately misalign rows by this many elements.
#     """
#     elem_size = np.ctypeslib.as_array(mp.RawArray(typecode, 1)).itemsize
#     assert alignment % elem_size == 0, "alignment must be multiple of elem_size"
#     cache_line = alignment // elem_size  # units: elements
#     misalign = int(misalign) % cache_line
#     misalign_bytes = misalign * elem_size
#     n_pad = n + cache_line - (n % cache_line)
#     num_elements = n_pad * m
#     x = np.ctypeslib.as_array(mp.RawArray(typecode, num_elements + cache_line))
#     assert x.ctypes.data % elem_size == 0, "multiprocessing.RawArray() did  \
#         not provide element-aligned memory (re-write this fcn to be byte-based)"
#     start_idx = -x.ctypes.data % alignment + misalign
#     # if misalign:
#     #     start_idx += cache_line // 2  # try to make maximum overlap
#     z = x[start_idx:start_idx + num_elements].reshape(m, n_pad)
#     assert misalign_bytes == (z.ctypes.data % alignment), "array (mis)alignment did not work"
#     # assert misalign != (z.ctypes.data % alignment == 0), "array alignment did not work"
#     for i in range(m):
#         assert misalign_bytes == (z[i, :].ctypes.data % alignment), "row (mis)alignment did not work"
#     #     assert misalign != (z[1, :].ctypes.data % alignment == 0), "row alignment did not work"

#     return z


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
# parser.add_option('-m', '--misalgin', action='store', dest='misalign',
#                   default=0, type="int",
#                   help='Misalign rows vs cache line boundary by this many elements')
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
