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

    m = options.m
    n = options.n
    chunks = options.chunks
    rows_whole = not options.cols_whole
    dtype = options.dtype
    itrs = options.itrs
    prec = options.prec
    sum_opt = options.sum_opt

    if dtype == 'float64':
        dtype = np.float64
    elif dtype == 'float32':
        dtype = np.float32
    else:
        raise ValueError('Unrecognized dtype. Available: [float64, float32]')

    np.set_printoptions(formatter={'float': '{{: 0.{}f}}'.format(prec).format})

    chunked_array, boundary_pairs = chunked_2d_array(m, n, chunks, rows_whole, dtype)
    out = np.zeros(n, dtype)


    t_itrs = list()
    t_start = timer()
    if chunks == 1:
        for i in range(itrs):
            t_start = timer()
            out[:] = chunked_array.sum(axis=0)
            t_itrs.append(timer() - t_start)
    else:
        if rows_whole:
            if sum_opt == 1:
                for i in range(itrs):
                    t_start = timer()
                    out[:] = np.sum([array.sum(axis=0) for array in chunked_array], axis=0)
                    t_itrs.append(timer() - t_start)
            elif sum_opt == 2:
                # Can be 2x faster than sum_opt 1, at least on the i7.
                out_intermediate = np.zeros((chunks, n), dtype)  # maybe a lot of memory
                for i in range(itrs):
                    t_start = timer()
                    for j, array in enumerate(chunked_array):
                        out_intermediate[j, :] = array.sum(axis=0)
                    out[:] = out_intermediate.sum(axis=0)
                    t_itrs.append(timer() - t_start)
            elif sum_opt == 3:
                # Same speed as sum_opt 1.
                for i in range(itrs):
                    t_start = timer()
                    out[:] = np.array([array.sum(axis=0) for array in chunked_array]).sum(axis=0)
                    t_itrs.append(timer() - t_start)
            else:
                raise ValueError("sum_opt not recognized. available: [1, 2, 3]")
        else:
            # Definitely the fastest, insensitive to number of chunks.
            for i in range(itrs):
                t_start = timer()
                for array, vb in zip(chunked_array, boundary_pairs):
                    out[vb[0]:vb[1]] = array.sum(axis=0)
                t_itrs.append(timer() - t_start)


    t_itrs = np.asarray(t_itrs)

    print("\nOverall time: {}, Itr Times: \n{}\n".format(t_itrs.sum(), t_itrs))


def chunked_2d_array(m, n, chunks=2, rows_whole=True, dtype=np.float64):
    """
    m: number of rows of data
    n: number of columns of data
    chunks: number of chunks to use
    rows_whole: if True, chunk along dimension 0, else chunk along dimension 1
    dtype: used in numpy.zeros()

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

    chunked_array = [np.zeros(chunk_shape, dtype) for _ in range(chunks)]

    bounds = [chunk_size * i for i in range(chunks + 1)]
    bounds[-1] = chunk_dim_size
    boundary_pairs = [(bounds[i], bounds[i + 1]) for i in range(chunks)]

    if chunks == 1:
        chunked_array = chunked_array[0]  # don't return list if length is 1.
    else:
        last_size = bounds[-1] - bounds[-2]
        if rows_whole:
            last_shape = (last_size, n)
        else:
            last_shape = (m, last_size)
        chunked_array[-1] = np.zeros(last_shape, dtype)

    return chunked_array, boundary_pairs


parser = OptionParser(
    usage='%prog <options>\nTry to recreate irregular write times to '
    'multiprocessing shared variable array.')

parser.add_option('-m', action='store', dest='m', default=8,
                  type='int', help='Number of rows')
parser.add_option('-n', action='store', dest='n', default=100000,
                  type='int',
                  help='Number of columns')
parser.add_option('-i', '--itrs', action='store', dest='itrs', default=100,
                  type='int', help='Number of iterations of summing.')
parser.add_option('-p', '--prec', action='store', dest='prec', default=3,
                  type='int', help='Precision of times printed.')
parser.add_option('-X', '--size_X', action='store', dest='size_X', default=2000,
                  type='int',
                  help='Size of matrix (linear dimension) used to cycle cache.')
parser.add_option('-v', '--verbose', action='store_true', dest='verbose',
                  default=False, help='If True, print all worker times.')
parser.add_option('-t', '--dtype', action='store', dest='dtype',
                  default='float64', type="str",
                  help='dtype used in numpy.array()')
parser.add_option('-c', '--chunks', action='store', dest='chunks', default=1,
                  type="int", help='Allocate array in multiple chunks')
parser.add_option('--cols_whole', action='store_true', dest='cols_whole',
                  default=False,
                  help='If True, chunk along rows instead of columns')
parser.add_option('-s', '--sum_opt', action='store', dest='sum_opt', type="int",
                  default=1, help='Option for how to perform the sum')
# parser.add_option('-s', '--scramble_cpus', action='store_true',
#                   dest='scramble_cpus', default=False,
#                   help='If True, randomly permute CPU affinity of ranks')


if __name__ == "__main__":
    options, arguments = parser.parse_args()

    execute(options)
