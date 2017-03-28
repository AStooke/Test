
import theano
import multiprocessing as mp
import numpy as np
import theano.gpuarray
from pygpu import collectives as gpu_coll

N_GPU = 2
MASTER = 0


def worker(mgr_dict, barrier, rank):
    theano.gpuarray.use("cuda" + str(rank))
    gpu_ctx = theano.gpuarray.get_context(None)
    clique_id = gpu_coll.GpuCommCliqueId(gpu_ctx)
    barrier.wait()
    clique_id.comm_id = mgr_dict["master_id"]
    gpu_comm = gpu_coll.GpuComm(clique_id, N_GPU, rank)
    print("worker rank in gpu_comm: {}".format(rank))

    my_arr = rank * np.ones([2, 2], dtype='float32')
    s = theano.shared(my_arr)
    gpu_comm.all_gather(s.container.data)


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

    my_arr = MASTER * np.ones([2, 2], dtype='float32')
    s = theano.shared(my_arr)
    r = gpu_comm.all_gather(s.container.data)
    print(r)

    for p in procs: p.join()


if __name__ == "__main__":
    master()
