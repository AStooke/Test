"""
Linear algebra speed test.
Uses numpy.linalg.lstqr
"""

from __future__ import absolute_import, print_function, division

import os
import sys
from timeit import default_timer as timer
from optparse import OptionParser

import numpy as np


def execute(N=1000, iters=10, verbose=True, solver='numpy'):

    if verbose:
        print('Some OS information:')
        print('    sys.platform=', sys.platform)
        print('    sys.version=', sys.version)
        print('    sys.prefix=', sys.prefix)
        print('Some environment variables:')
        print('    MKL_NUM_THREADS=', os.getenv('MKL_NUM_THREADS'))
        # print('    OMP_NUM_THREADS=', os.getenv('OMP_NUM_THREADS'))
        # print('    GOTO_NUM_THREADS=', os.getenv('GOTO_NUM_THREADS'))
        print()
        print('Numpy config: ')
        np.show_config()
        print('Numpy linalg path:', np.linalg.__path__)
        print('Numpy location:', np.__file__)
        print('Numpy version:', np.__version__)

    b = np.random.randn(N)
    x = np.random.randn(N, N)
    X = x.T.dot(x) + 1e-2 * np.ones(N)  # real problem has positive definite matrix

    # Do the first call without timing.
    t_prep = timer()
    y = np.linalg.lstsq(X, b)
    t_first = timer()
    print("\nTime for prep run: {}".format(t_first - t_prep))
    for _ in range(iters):
        y = np.linalg.lstsq(X, b)
        y[0][0] += 1  # Use the result.
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
                  help="The N size to dgesld")
parser.add_option('--iter', action='store', dest='iter',
                  default=10, type="int",
                  help="The number of calls to dgesld")
parser.add_option('-t', '--threads', action='store', dest='threads', default=0,
                  type="int",
                  help="Number assigned to MKL_NUM_THREADS (0 for unset)")


if __name__ == "__main__":
    options, arguments = parser.parse_args(sys.argv)

    if hasattr(options, "help"):
        print(options.help)
        sys.exit(0)

    th = options.threads
    if th != 0:
        os.environ['MKL_NUM_THREADS'] = str(th)
    else:
        try:
            del os.environ['MKL_NUM_THREADS']
        except KeyError:
            pass

    verbose = not options.quiet

    t = execute(N=options.N, iters=options.iter, verbose=verbose)

    print("\nWe executed", options.iter)
    print("calls to numpy.linalg.lstsq on problem size {}".format(options.N))
    print("using MKL_NUM_THREADS = {}".format("unset" if th == 0 else th))
    print('\nTotal execution time: %.2fs.\n' % t)
