
import multiprocessing as mp 
import numpy as np 
import psutil
from optparse import OptionParser
import sys
from timeit import default_timer as timer


def execute(n_proc, vec_dim, itrs, use_lock, prec):

    shared_array = np.frombuffer(mp.RawArray('d', n_proc * vec_dim)
                                 ).reshape((vec_dim, n_proc))
    barriers = [mp.Barrier(n_proc) for _ in range(2)]
    lock = mp.Lock()

    np.set_printoptions(formatter={'float': '{{: 0.{}f}}'.format(prec).format})

    processes = [mp.Process(target=run_worker, 
                            args=(rank, vec_dim, shared_array, barriers, lock, 
                                  itrs, use_lock)
                            ) 
                    for rank in range(n_proc)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()


def run_worker(rank, vec_dim, shared_array, barriers, lock, itrs, use_lock):
    p = psutil.Process()
    p.cpu_affinity([rank % psutil.cpu_count()])
    
    t_itrs = []
    t_next = []
    t_next.append(timer())
    for _ in range(itrs):
        local_vec = np.random.randn(vec_dim)
        if use_lock:
            with lock:
                shared_array[:, rank] = local_vec
        else:
            shared_array[:, rank] = local_vec
        barriers[0].wait()
        t_next.append(timer())
        t_itrs.append(t_next[-1] - t_next[-2])
    t_itrs = np.asarray(t_itrs)  # Just for printing nicely.
    with lock:
        print("Rank: {:2}, Itr Times: {}".format(rank, t_itrs))


    




parser = OptionParser(
    usage='%prog <options>\nTry to recreate irregular write times to '
    'multiprocessing shared variable.')

parser.add_option('-n', '--n_proc', action='store', dest='n', default=1, 
                  type='int', help='Number of parallel processes to run.')
parser.add_option('-d', '--vec_dim', action='store', dest='d', default=1000,
                  type='int', 
                  help='Length of shared vector (array size = n * v)')
parser.add_option('-i', '--itrs', action='store', dest='i', default=10,
                  type='int', help='Number of iterations of writing to shared.')
parser.add_option('-l', '--lock', action='store_true', dest='lock',
                  default=False, 
                  help='If True, acquire lock when writing to shared.')
parser.add_option('-p', '--prec', action='store', dest='p', default=3,
                  type='int', help='Precision of times printed.')


if __name__ == "__main__":
    options, arguments = parser.parse_args(sys.argv)

    execute(n_proc=options.n,
            vec_dim=options.d,
            itrs=options.i,
            use_lock=options.lock,
            prec=options.p,
            )

