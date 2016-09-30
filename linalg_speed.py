"""
Linear algebra speed test.
Uses numpy.linalg.lstqr
"""

from __future__ import absolute_import, print_function, division

import os
import sys
# import time
from timeit import default_timer as timer
from optparse import OptionParser
# import subprocess

import numpy as np
# import theano
# import theano.tensor as T


def execute(N=1000, iters=10, verbose=True):

    if verbose:
        print('Some OS information:')
        print('    sys.platform=', sys.platform)
        print('    sys.version=', sys.version)
        print('    sys.prefix=', sys.prefix)
        print('Some environment variables:')
        print('    MKL_NUM_THREADS=', os.getenv('MKL_NUM_THREADS'))
        print('    OMP_NUM_THREADS=', os.getenv('OMP_NUM_THREADS'))
        print('    GOTO_NUM_THREADS=', os.getenv('GOTO_NUM_THREADS'))
        print()
        print('Numpy config: (used when the Theano flag'
              ' "blas.ldflags" is empty)')
        np.show_config()
        print('Numpy dot module:', np.dot.__module__)
        print('Numpy location:', np.__file__)
        print('Numpy version:', np.__version__)

    # Might want to make a separate one of these for each iteration, to also
    # check time to load data.
    b = np.random.randn(N)
    x = np.random.randn(N, N)
    X = x.T.dot(x) + 1e-2 * np.ones(N)

    # Do the first call without timing.
    t_prep = timer()
    y = np.linalg.lstsq(X, b)
    t_first = timer()
    print("\ntime for prep run: {}".format(t_first - t_prep))
    for _ in range(iters):
        y = np.linalg.lstsq(X, b)
        y[0][0] += 1  # Just to make sure it's actually giving a value each time.
    t_last = timer()

    return t_last - t_first


parser = OptionParser(
    usage='%prog <options>\nCompute time needed to perform linear algebra least'
    ' squares fit of dimension N.')

parser.add_option('-q', '--quiet', action='store_true', dest='quiet',
                  default=False,
                  help="If true, do not print the config options")

parser.add_option('-N', '--N', action='store', dest='N',
                  default=1000, type="int",
                  help="The N size to gemm")

parser.add_option('--iter', action='store', dest='iter',
                  default=10, type="int",
                  help="The number of calls to gemm")


if __name__ == "__main__":
    options, arguments = parser.parse_args(sys.argv)

    if hasattr(options, "help"):
        print(options.help)
        sys.exit(0)

    verbose = not options.quiet

    t = execute(N=options.N, iters=options.iter, verbose=verbose)

    print("\nWe executed", options.iter, end=' ')
    print("\ncalls to numpy.linalg.lstsq on problem size %d" % options.N)
    print('\nTotal execution time: %.2fs.\n' % t)
