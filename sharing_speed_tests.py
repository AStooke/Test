import multiprocessing as mp
import numpy as np
import psutil
import cProfile
from line_profiler import LineProfiler

from multiprocessing import Semaphore
from multiprocessing.managers import BaseManager

from timeit import default_timer as timer

class Barrier:
    ''' Reusable barrier for a specified number of threads, 
        implemented with semaphores. '''
    def __init__(self, n):
        self.n = n # make equal to the number of threads that need to
                   # adhere to the barrier
        self.count = 0 # number of threads having reached the barrier
        self.gate_in = Semaphore(1) # this one is initialized as 'available'
        self.gate_out = Semaphore(1)
        self.barrier = Semaphore(0) # this one is initialized as 'blocked'

    def wait(self):
        
        self.gate_in.acquire() # prevent race condition on inward increment
        self.count += 1 # another thread has reached the barrier 
        
        if self.count == self.n: # i.e. if last thread in
            self.barrier.release() # release barrier so that one other 
                                   # thread can pass the barrier.acqurire()
            # Note: keep gate_in acquired to prevent any thread from re-entering
            # before everyone is released and exits.
        else:
            self.gate_in.release() # let other threads pass the inward gate

        self.barrier.acquire() # threads wait here until the barrier.release() above

        self.gate_out.acquire() # prevent race condition on outward decrement
        self.count -= 1 # one less thread is still at the barrier     

        if self.count == 0: # i.e. if last thread out
            # Note: leave the barrier acquired (it will be in that state now)

            self.gate_in.release() # reset the inward gate other threads can enter
                                   # on the next time through

        else:
            self.barrier.release() # whenever getting past the acquire, re-release
                                   # so another thread can get by
        
        self.gate_out.release() # let the next thread exit OR reset the outward gate
                                # for next time through
            
class BarrierManager(BaseManager):
    pass


def set_affinity(rank):
    cpu_order = [0,1,2,3,4,5,6,7,8,9,10,11]
    n_cpu = len(cpu_order)
    proc = psutil.Process()
    assigned_affinity = [cpu_order[rank % n_cpu]]
    proc.cpu_affinity(assigned_affinity)
    # proc.cpu_affinity([0])

    # affin = '\nRank: {},  Affinity: {}'.format(rank, proc.cpu_affinity())
    # print affin


def loop_prof_1(rank, n_proc, shared_array, shared_array_np, shared_arrays, locks, barrier, array_size, n_loops, vb_bounds, vb_idx_lists):
    # profile = LineProfiler(loop_func_1)
    # profile.runctx('loop_func_1(rank, n_proc, shared_array, shared_array_np, shared_arrays, locks, barrier, array_size, n_loops, vb_bounds, vb_idx_lists)', 
    #                 globals(), locals(), 'prof%d.prof' %rank)
    profile = LineProfiler()
    # profile.timer_unit = 1. # doesn't work
    prof_func = profile.__call__(loop_func_1)
    profile.runcall(loop_func_1,
        *(rank, n_proc, shared_array, shared_array_np, shared_arrays, locks, barrier, array_size, n_loops, vb_bounds, vb_idx_lists)
        )
    if rank==0:
        # lstats = profile.get_stats()
        # print 'lstats.timing: ', lstats.timings
        profile.print_stats()


def loop_prof_2(rank, n_proc, shared_array, shared_arrays, barrier, array_size, n_loops):
    # profile = LineProfiler(loop_func_1)
    # profile.runctx('loop_func_1(rank, n_proc, shared_array, shared_array_np, shared_arrays, locks, barrier, array_size, n_loops, vb_bounds, vb_idx_lists)', 
    #                 globals(), locals(), 'prof%d.prof' %rank)
    profile = LineProfiler()
    # profile.timer_unit = 1. # doesn't work
    prof_func = profile.__call__(loop_func_2)
    profile.runcall(loop_func_2,
        *(rank, n_proc, shared_array, shared_arrays, barrier, array_size, n_loops)
        )
    if rank==0:
        # lstats = profile.get_stats()
        # print 'lstats.timing: ', lstats.timings
        profile.print_stats()


def loop_prof_3(rank, n_proc, shared_array, shared_array_2d, gb, barrier, array_size, n_loops, vb_bounds):
    # profile = LineProfiler(loop_func_1)
    # profile.runctx('loop_func_1(rank, n_proc, shared_array, shared_array_np, shared_arrays, locks, barrier, array_size, n_loops, vb_bounds, vb_idx_lists)', 
    #                 globals(), locals(), 'prof%d.prof' %rank)
    profile = LineProfiler()
    # profile.timer_unit = 1. # doesn't work
    prof_func = profile.__call__(loop_func_3)
    profile.runcall(loop_func_3,
        *(rank, n_proc, shared_array, shared_array_2d, gb, barrier, array_size, n_loops, vb_bounds)
        )
    if rank==0:
        # lstats = profile.get_stats()
        # print 'lstats.timing: ', lstats.timings
        profile.print_stats()


def loop_func_1(rank, n_proc, shared_array, shared_array_np, shared_arrays, locks, barrier, array_size, n_loops, vb_bounds, vb_idx_lists):
    set_affinity(rank)

    vb_idx_list = vb_idx_lists[rank]


    time_total = 0.
    time_rand = 0.
    time_fill = 0.
    time_share = 0.

    start_time = timer()
    for i in xrange(n_loops):
        time_1 = timer()
        my_array = np.random.rand(array_size)
        time_2 = timer()
        if rank == 0:
            shared_array_np.fill(0.)
        barrier.wait()
        time_3 = timer()
        # with locks[0]:
            # shared_array[:] += my_array
        for i in vb_idx_list:
            with locks[i]:
                shared_array[vb_bounds[i][0]:vb_bounds[i][1]] += my_array[vb_bounds[i][0]:vb_bounds[i][1]]
        barrier.wait()
        time_4 = timer()
        time_rand += time_2 - time_1
        time_fill += time_3 - time_2
        time_share += time_4 - time_3

    time_total = time_3 - start_time

    if rank == 0:
        report = '\n\nn_proc: {}'.format(n_proc)
        report += '\nfunc_1 \ntime_total: {}'.format(time_total)
        report += '\ntime_rand: {}'.format(time_rand)
        report += '\ntime_fill: {}'.format(time_fill)
        report += '\ntime_share: {}'.format(time_share)
        print report


def loop_func_2(rank, n_proc, shared_array, shared_arrays, barrier, array_size, n_loops):
    set_affinity(rank)


    time_total = 0.
    time_rand = 0.
    time_share = 0.  
    time_write = 0.
    time_reduce = 0.

    start_time = timer()
    for i in xrange(n_loops):
        barrier.wait()
        time_1 = timer()
        my_array = np.random.rand(array_size)
        time_2 = timer()
        # with locks[0:
        shared_arrays[rank][:] = my_array
        time_3 = timer()
        barrier.wait()

        if rank==0:
            shared_array[:] = reduce(np.add, shared_arrays)

        barrier.wait()
        time_4 = timer()

        time_rand += time_2 - time_1
        time_write += time_3 - time_2
        time_reduce += time_4 - time_3
        time_share = time_write + time_reduce
    time_total = time_4 - start_time

    if rank == 0:
        report = '\n\nn_proc: {}'.format(n_proc)
        report += '\nfunc_2 \ntime_total: {}'.format(time_total)
        report += '\ntime_rand: {}'.format(time_rand)
        report += '\ntime_write: {}'.format(time_write)
        report += '\ntime_reduce: {}'.format(time_reduce)
        report += '\ntime_share: {}'.format(time_share)
        print report


def loop_func_3(rank, n_proc, shared_array, shared_array_2d, gb, barrier, array_size, n_loops, vb):
    set_affinity(rank)


    time_total = 0.
    time_rand = 0.
    time_share = 0.  
    time_write = 0.
    time_reduce = 0.
    check_array = np.zeros(array_size)

    start_time = timer()
    for i in xrange(n_loops):
        barrier.wait()
        time_1 = timer()
        my_array = np.random.rand(array_size)
        # shared_arrays[rank][:]= np.random.rand(array_size)
        time_2 = timer()
        # with locks[0:
        shared_array_2d[:,rank] = my_array[:]
        time_3 = timer()
        barrier.wait()

        # shared_array[vb[0]:vb[1]] = reduce(np.add, [shared_array_list[r*array_size + vb[0]:r*array_size +vb[1]] for r in range(n_proc)])
        shared_array[vb[0]:vb[1]] = np.sum(shared_array_2d[vb[0]:vb[1],:], axis=1) # sum along row-major order
        # if rank == 0:
        #     shared_array[:] = np.sum(shared_array_2d, axis=1)
        barrier.wait()
        time_4 = timer()

        # if rank == 0:
        #     check_array[:] = np.sum(shared_array_2d,axis=1)
        #     print 'iteration: ', i
        #     print shared_array[:5]
        #     print check_array[:5]
        #     assert all(check_array[:] == shared_array[:])

        time_rand += time_2 - time_1
        time_write += time_3 - time_2
        time_reduce += time_4 - time_3
        time_share = time_write + time_reduce
    time_total = time_4 - start_time

    if rank == 0:
        report = '\n\nn_proc: {}'.format(n_proc)
        report += '\nfunc_3 \ntime_total: {}'.format(time_total)
        report += '\ntime_rand: {}'.format(time_rand)
        report += '\ntime_write: {}'.format(time_write)
        report += '\ntime_reduce: {}'.format(time_reduce)
        report += '\ntime_share: {}'.format(time_share)
        print report


def main():
    
    array_size = 1000000
    n_loops = 100

    n_proc = 6

    shared_array = mp.RawArray('d', array_size)
    shared_array_np = np.frombuffer(shared_array)

    shared_arrays = [mp.RawArray('d', array_size) for _ in range(n_proc)]
    shared_arrays_np = [np.frombuffer(sa) for sa in shared_arrays]


    

    BarrierManager.register('Barrier',Barrier)
    barrier_mgr = BarrierManager()
    barrier_mgr.start()
    barrier = barrier_mgr.Barrier(n_proc)

    # For loop_func_1
    vec_block_factor = 3. # (blocks per process)
    n_vec_blocks = int(vec_block_factor * n_proc)
    n_per_block = -(-array_size // n_vec_blocks) # (ceiling division operator)
    vb_indeces = [n_per_block * i for i in range(n_vec_blocks + 1)]
    vb_indeces[-1] = array_size
    vb_bounds = [ (vb_indeces[i], vb_indeces[i+1]) for i in range(n_vec_blocks) ]
    vb_starts = [int(vec_block_factor * proc) for proc in range(n_proc)]
    vb_idx_lists = [ range(vb_starts[proc], n_vec_blocks) + range(vb_starts[proc]) for proc in range(n_proc)]
    locks = [mp.Lock() for _ in range(n_vec_blocks) ]




    start_1 = timer()
    processes = [ mp.Process(target=loop_func_1, 
                            args=(rank, n_proc, shared_array, shared_array_np, shared_arrays, locks, barrier, array_size, n_loops, vb_bounds, vb_idx_lists)) 
                    for rank in range(n_proc) ]
    for p in processes: p.start()
    for p in processes: p.join()
    end_1 = timer()


    # start_2 = timer()
    # processes = [ mp.Process(target=loop_prof_2, 
    #                         args=(rank, n_proc, shared_array, shared_arrays, barrier, array_size, n_loops)) 
    #                 for rank in range(n_proc) ]
    # for p in processes: p.start()
    # for p in processes: p.join()
    # end_2 = timer()


    # For loop_func_3
    shared_array_list = mp.RawArray('d', array_size * n_proc)
    shared_array_2d = np.frombuffer(shared_array_list).reshape((array_size, n_proc))
    grad_idx = [array_size * i for i in range(n_proc + 1)]
    gb = [ (grad_idx[i], grad_idx[i+1]) for i in range(n_proc)]
    n_per_proc = -(-array_size // n_proc)
    vb_indeces = [n_per_proc * i for i in range(n_proc + 1)]
    vb_indeces[-1] = array_size
    vb_bounds = [ (vb_indeces[i], vb_indeces[i+1]) for i in range(n_proc) ]
    vb = vb_bounds
    # shared_arrays_views = [ [ shared_arrays[r1][vb[r2][0]:vb[r2][1]] for r1 in range(n_proc) ] for r2 in range(n_proc)]
    # print vb


    start_3 = timer()
    processes = [ mp.Process(target=loop_func_3, 
                            args=(rank, n_proc, shared_array, shared_array_2d, gb[rank], barrier, array_size, n_loops, vb[rank])) 
                    for rank in range(n_proc) ]
    for p in processes: p.start()
    for p in processes: p.join()
    end_3 = timer()

    print '\ntime_1: ', end_1 - start_1

    # print 'time_2: ', end_2 - start_2

    print 'time_3: ', end_3 - start_3

    # print 'interim:', start_2 - end_1

if __name__ == '__main__':
    main()

