"""
Speed test for spawning processes and importing.
"""

# from __future__ import absolute_import, print_function, division

import sys
from timeit import default_timer as timer
from optparse import OptionParser

# import numpy
import multiprocessing as mp
import psutil


def execute(n_proc=10, iters=10, verbose=True, set_affinity=False, import_numpy=False):

    if verbose:
        print('Some OS information:')
        print('    sys.platform=', sys.platform)
        print('    sys.version=', sys.version)
        print('    sys.prefix=', sys.prefix)

    s = set_affinity
    imp = import_numpy

    t_start = timer()
    for i in range(iters):
        procs = [mp.Process(target=worker, args=(r, s, imp)) for r in range(n_proc)]
        if verbose:
            print("\nStarting procs in iteration {}".format(i))
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        if verbose:
            print("\nJoined procs in iteration {}".format(i))
    t_end = timer()

    return t_end - t_start


def worker(rank, set_affinity, import_numpy):
    if set_affinity:
        # import psutil  # Much slower if this is here instead of header!
        p = psutil.Process()
        num_cpu = psutil.cpu_count()
        p.cpu_affinity([rank % num_cpu])
    if import_numpy:
        import numpy
        x = numpy.ndarray([1, 2])
    x = 7.
    y = 8.
    z = x * y
    z += 1


parser = OptionParser(
    usage='%prog <options>\nCompute time needed to perform linear algebra least'
    ' squares fit of dimension N.')

parser.add_option('-q', '--quiet', action='store_true', dest='quiet',
                  default=False,
                  help="If true, do not print the config options")

parser.add_option('-n', '--n_proc', action='store', dest='n_proc',
                  default=10, type="int",
                  help="The number of processes to spawn")

parser.add_option('--iter', action='store', dest='iter',
                  default=10, type="int",
                  help="The number of times spawning and joining processes")

parser.add_option('-s', '--set_affinity', action='store_true',
                  dest='set_affinity', default=False,
                  help="Set CPU affinity inside subprocesses or not")

parser.add_option('-i', '--import_numpy', action='store_true',
                  dest='import_numpy', default=False,
                  help="import Numpy inside subprocesses or not")


if __name__ == "__main__":
    options, arguments = parser.parse_args(sys.argv)

    if hasattr(options, "help"):
        print(options.help)
        sys.exit(0)

    n = options.n_proc
    verbose = not options.quiet
    iters = options.iter
    s = options.set_affinity
    i = options.import_numpy
    print(i)
    t = execute(n_proc=n, iters=iters, verbose=verbose, set_affinity=s, import_numpy=i)

    r = ''
    r += '\nWe executed {} loops ({} setting cpu affinity),'.format(iters,
        'with' if s else 'without')
    r += '\nof multiprocessing start/join with {} processes,'.format(n)
    r += '\n{} importing Numpy in each subprocess.'.format('with' if i else 'without')
    r += '\nTotal execution time: {:.2f}s.\n'.format(t)
    print(r)

