"""
Quick demo to show using NCCL GPU collectives on Theano variables via pygpu.
"""

import os
import numpy as np
import multiprocessing as mp
import time
import theano

import ipdb

N_GPU = 2
MASTER_RANK = 0


def target(rank, d, b):
    import theano.gpuarray
    theano.gpuarray.use('cuda1')

    os.environ["THEANO_FLAGS"] = "device=cuda" + str(rank)
    from pygpu import collectives as gpucoll

    gpuctx = theano.gpuarray.get_context(None)
    # Worker calls GpuCommCliqueId only to get a clique ID object; will
    # overwrite the actual ID with that from the master (object not pickleable).
    local_id = gpucoll.GpuCommCliqueId(gpuctx)
    b.wait()
    local_id.comm_id = d["comm_id"]
    local_comm = gpucoll.GpuComm(local_id, N_GPU, rank)

    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = x + y
    f = theano.function([x, y], z.transfer(None))
    x_dat = np.ones([3, 3], dtype='float32')
    r = f(x_dat, x_dat)
    gathered_r = local_comm.all_gather(r, nd_up=0)
    local_comm.reduce(r, 'sum', root=MASTER_RANK)
    local_comm.all_gather(r, nd_up=0)
    time.sleep(1)
    print("\nworker gathered_r: \n", gathered_r)

    # time.sleep(2)  # (just for sequential printing)

    # s = theano.shared(np.zeros([2, 2], dtype='float32'))
    # print("\nworker shared before broadcast: \n", s.get_value())
    # local_comm.broadcast(s.container.data, root=MASTER_RANK)
    # print("\nworker shared after broadcast: \n", s.get_value())


def main():

    mgr = mp.Manager()
    d = mgr.dict()
    b = mp.Barrier(2)
    p = mp.Process(target=target, args=(1, d, b))
    p.start()

    import theano.gpuarray
    theano.gpuarray.use('cuda0')
    from pygpu import collectives as gpucoll

    gpuctx = theano.gpuarray.get_context(None)
    local_id = gpucoll.GpuCommCliqueId(gpuctx)
    d["comm_id"] = local_id.comm_id  # Give the ID to the worker.
    b.wait()  # signal the worker that the ID is ready.
    local_comm = gpucoll.GpuComm(local_id, N_GPU, MASTER_RANK)

    x = theano.tensor.matrix('x')
    y = theano.tensor.matrix('y')
    z = x + y
    f = theano.function([x, y], z.transfer(None))
    x_dat = np.ones([3, 3], dtype='float32')
    r = f(x_dat, x_dat)

    gathered_r = local_comm.all_gather(r, nd_up=0)
    print("\nmaster gathered_r: \n", gathered_r)

    print("\nresult in master before reduce: \n", r)
    local_comm.reduce(r, 'sum', r, root=0)
    print("\nresult in master after reduce: \n", r)

    gather_ret2 = local_comm.all_gather(r, dest=gathered_r, nd_up=1)

    ipdb.set_trace()

    # time.sleep(2)

    # s = theano.shared(np.ones([2, 2], dtype='float32'))
    # local_comm.broadcast(s.container.data)

    p.join()


if __name__ == "__main__":
    main()


