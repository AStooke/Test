"""
Run theano functions in parallel on multiple GPUs (data parallelism).

This file has everything the workers do.
"""


import pickle

from Function import WorkerFunction
import handling as h
import util
from util import (PKL_FILE, FUNCTION, GPU_COMM, BROADCAST, REDUCE, ALL_REDUCE,
                  ALL_GATHER, REDUCE_SCATTER)


def worker_exec(rank, n_gpu, master_rank, sync, inputs):
    """
    This is the function the subprocess is set to run.

    It needs to:
    1. Wait for the signal to initialize, then do initialization:
       a. Unpickle Theano functions.
       b. Maybe some record-keeping around theano shared variables.
       c. Set up / receive key of different signals.
    2. Go into infinite loop of waiting to see a signal then acting.

    Open questions...what to house in classes and what to just run as functions.

    """
    # Initialize distinct GPU.
    util.use_gpu(rank)

    # Receive functions.
    sync.barriers.distribute.wait()
    gpu_comm = util.init_gpu_comm(n_gpu, rank, sync.dict["comm_id"])
    WorkerFunction.master_rank = master_rank
    WorkerFunction._sync = sync  # endow all functions
    WorkerFunction._gpu_comm = gpu_comm
    with open(PKL_FILE, "rb") as f:
        theano_functions = pickle.load(f)  # should be all in one list
    # Might have the last worker delete the pkl file.
    functions, shareds, named_shareds = h.functions_handling(
        theano_functions, inputs, rank)

    # Infinite execution loop.
    while True:
        sync.barriers.exec_in.wait()
        if sync.quit.value:
            break
        if sync.exec_type.value == FUNCTION:
            functions[sync.func_code.value]()
        elif sync.exec_type.value == GPU_COMM:
            # TODO: figure out how to organize these better.
            # And how to reduce using other ops.
            # And how to make AVERAGING happen.
            if sync.comm_code.value == BROADCAST:
                for idx in sync.shared_codes[:sync.n_shareds.value]:
                    gpu_comm.broadcast(shareds[idx].data, root=master_rank)
            elif sync.comm_code.value == REDUCE:
                for idx in sync.shared_codes[:sync.n_shareds.value]:
                    gpu_comm.reduce(shareds[idx].data, 'sum', root=master_rank)
            elif sync.comm_code.value == ALL_REDUCE:
                for idx in sync.shared_codes[:sync.n_shareds.value]:
                    gpu_comm.all_reduce(shareds[idx].data, 'sum', shareds[idx].data)
            elif sync.comm_code.value == ALL_GATHER:
                for idx in sync.shared_codes[:sync.n_shareds.value]:
                    gpu_comm.all_gather(shareds[idx].data)  # FIXME: call signature
            elif sync.comm_code.value == REDUCE_SCATTER:
                for idx in sync.shared_codes[:sync.n_shareds.value]:
                    gpu_comm.reduce_scatter(shareds[idx].data, 'sum')

        sync.barriers.exec_out.wait()


