
import theano
import multiprocessing as mp
import numpy as np
import theano.gpuarray
from pygpu import collectives as gpu_coll
import time

N_GPU = 2
MASTER = 0
OP = "sum"


def worker(mgr_dict, barrier, rank):
    theano.gpuarray.use("cuda" + str(rank))
    gpu_ctx = theano.gpuarray.get_context(None)
    clique_id = gpu_coll.GpuCommCliqueId(gpu_ctx)
    barrier.wait()
    clique_id.comm_id = mgr_dict["master_id"]
    gpu_comm = gpu_coll.GpuComm(clique_id, N_GPU, rank)
    print("worker rank in gpu_comm: {}".format(rank))
    time.sleep(2)
    test_sequence(rank, gpu_comm, barrier)


def master():
    barrier = mp.Barrier(N_GPU)
    mgr = mp.Manager()
    mgr_dict = mgr.dict()
    procs = [mp.Process(target=worker, args=(mgr_dict, barrier, rank))
        for rank in range(0, N_GPU) if rank != MASTER]
    for p in procs: p.start()
    theano.gpuarray.use("cuda" + str(MASTER))
    gpu_ctx = theano.gpuarray.get_context(None)
    clique_id = gpu_coll.GpuCommCliqueId(gpu_ctx)
    mgr_dict["master_id"] = clique_id.comm_id
    barrier.wait()
    gpu_comm = gpu_coll.GpuComm(clique_id, N_GPU, MASTER)
    print("master rank in gpu_comm: {}".format(MASTER))
    time.sleep(2)
    test_sequence(MASTER, gpu_comm, barrier)
    for p in procs: p.join()


def test_sequence(rank, gpu_comm, barrier):
    my_arr = rank * np.ones([2, 2], dtype='float32')
    s = theano.shared(my_arr)
    d = theano.shared(my_arr)
    if rank == MASTER:
        print("\nTesting all_gather")
        gpu_comm.all_gather(src=s.container.data, nd_up=0, dest=s.container.data)
        print(s.container.data)
        s.set_value(my_arr)
        barrier.wait()
        print("all_gather test complete")
        time.sleep(1)
    else:
        gpu_comm.all_gather(s.container.data)
        s.set_value(my_arr)
        barrier.wait()
        time.sleep(1)


if __name__ == "__main__":
    master()
