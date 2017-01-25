"""
Run theano functions in parallel on multiple GPUs (data parallelism).

This file has everything the workers do.
"""


import pickle
import numpy as np

from Function import SynkFunction
import handling
import util
from util import (PKL_FILE, FUNCTION, GPU_COMM, BROADCAST, REDUCE, ALL_REDUCE,
                  ALL_GATHER, WORKER_OPS, SH_ARRAY_TAG, SHMEM_TAG_PRE)
from shmemarray import ShmemRawArray


def get_op(sync, op_code):
    if op_code not in WORKER_OPS:
        raise ValueError("Unrecognized reduce operation in worker.")
    op = WORKER_OPS[op_code]
    if op == "avg":
        raise NotImplementedError


class Function(SynkFunction):

    rank = None
    master_rank = None

    def __call__(self, sync, all_inputs, gpu_comm):
        """
        This needs to:
        1. Gather the right inputs from mp shared values.
        2. Execute local theano function on those inputs.
        3. Send results back to master.

        NOTE: Barriers happen OUTSIDE worker function call.
        """
        my_inputs = self.receive_inputs(sync, all_inputs)
        my_results = self._call_theano_function(my_inputs)  # (always returns a tuple)
        self._collect_results(my_results, gpu_comm)

    def receive_inputs(self, sync, all_inputs):
        my_inputs = list()
        assign_idx = self.sync.assign_idx[self.code]
        my_idx = (assign_idx[self.rank], assign_idx[self.rank + 1])
        for inpt_code in self.input_codes:
            if sync.input_tags[inpt_code] != all_inputs.tags[inpt_code]:
                # Then a new shmem has been allocated, need to get it.
                all_inputs.shmems[inpt_code] = np.ctypeslib.as_array(
                    ShmemRawArray(sync.input_typecodes[inpt_code],  # make this get ctype from dict
                                  sync.input_sizes[inpt_code],
                                  SHMEM_TAG_PRE + str(sync.input_tags[inpt_code]),
                                  False)
                    ).reshape(sync.input_shapes[inpt_code])  # this is a list of separate shmems
                all_inputs.tags[inpt_code] = sync.input_tags[inpt_code]
            my_inputs.append(all_inputs.shmems[inpt_code][my_idx[0]:my_idx[1]])
        return tuple(my_inputs)

    def _reduce_results(self, my_results, gpu_comm):
        for r in my_results:
            gpu_comm.reduce(r, op=self.reduce_op, root=self.master_rank)

    def _gather_results(self, my_results, gpu_comm):
        """ use as gather (i.e. ignore "all") """
        for r in my_results:
            gpu_comm.all_gather(r)


def worker_exec(rank, n_gpu, master_rank, sync):
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
    if not sync.distributed.value:
        return  # (master closed before distributing functions--an error)
    gpu_comm = util.init_gpu_comm(n_gpu, rank, sync.dict["comm_id"])
    with open(PKL_FILE, "rb") as f:
        theano_functions = pickle.load(f)  # should be all in one list
    # Might have the last worker delete the pkl file.
    functions, inputs, shareds = handling.unpack_functions(theano_functions)
    sync.shared_codes = ShmemRawArray('i', shareds.num, SH_ARRAY_TAG, False)
    Function.rank = rank  # endow all functions
    Function.master_rank = master_rank

    # Infinite execution loop.
    while True:
        sync.barriers.exec_in.wait()
        if sync.quit.value:
            break
        if sync.exec_type.value == FUNCTION:
            functions[sync.func_code.value](sync, inputs, gpu_comm)
        elif sync.exec_type.value == GPU_COMM:
            shared_codes = sync.shared_codes[:sync.n_shareds.value]
            comm_code = sync.comm_code.value
            op = get_op(sync, sync.comm_op.value)  # (might not be used)
            for idx in shared_codes:
                if comm_code == BROADCAST:
                    gpu_comm.broadcast(shareds.gpuarrays[idx], root=master_rank)
                elif comm_code == REDUCE:
                    gpu_comm.reduce(shareds.gpuarrays[idx], op=op, root=master_rank)
                elif comm_code == ALL_REDUCE:
                    gpu_comm.all_reduce(shareds.gpuarrays[idx], op=op,
                            dest=shareds.gpuarrays[idx])
                elif comm_code == ALL_GATHER:
                    # NOTE: Results on the workers are ignored (behaves like
                    # 'gather' to the master).  I think this will put results in
                    # new memory on the worker without affecting the shared
                    # variables, and we don't keep the reference to that memory.
                    gpu_comm.all_gather(shareds.gpuarrays[idx])
                else:
                    raise ValueError("Unrecognized communication type in worker.")

        sync.barriers.exec_out.wait()  # TODO: decide if this barrier is helpful--yes


