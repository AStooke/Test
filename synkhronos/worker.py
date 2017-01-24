"""
Run theano functions in parallel on multiple GPUs (data parallelism).

This file has everything the workers do.
"""


import pickle

from Function import WorkerFunction
import handling
import util
from util import (PKL_FILE, FUNCTION, GPU_COMM, BROADCAST, REDUCE, ALL_REDUCE,
                  ALL_GATHER, WORKER_OPS, SH_ARRAY_TAG)
from shmemarray import ShmemRawArray


def get_op(sync, op_code):
    if op_code not in WORKER_OPS:
        raise ValueError("Unrecognized reduce operation in worker.")
    op = WORKER_OPS[op_code]
    if op == "avg":
        raise NotImplementedError


class Function(SynkFunction):

    sync = None
    gpu_comm = None
    master_rank = None

    def __call__(self):
        """
        This needs to:
        1. Gather the right inputs from mp shared values.
        2. Execute local theano function on those inputs.
        3. Send results back to master.

        NOTE: Barriers happen OUTSIDE worker function call.
        """
        inputs = self._receive_inputs()
        results = self._call_theano_function(inputs)
        self._send_results(results)

    def _receive_inputs(self):
        my_inputs = list()
        my_idx = (self.sync.assign_idx[self.rank],
            self.sync.assign_idx[self.rank + 1])
        for inpt in self.input_shmems




        for inpt in self.mp_inputs:
            my_inputs.append(inpt[self.s_ind:self.e_ind])  # a view
        return my_inputs

    def _send_results(self, results):
        if isinstance(results, (list, tuple)):
            for r in results:
                self._gpu_comm.reduce(r, 'sum', root=self.master_rank)
        else:
            self._gpu_comm.reduce(r, 'sum', root=self.master_rank)





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
    if not sync.distributed.value:
        return  # (master closed before distributing functions--an error)
    gpu_comm = util.init_gpu_comm(n_gpu, rank, sync.dict["comm_id"])
    Function.master_rank = master_rank
    Function._sync = sync  # endow all functions
    Function._gpu_comm = gpu_comm
    with open(PKL_FILE, "rb") as f:
        theano_functions = pickle.load(f)  # should be all in one list
    # Might have the last worker delete the pkl file.
    functions, shareds, named_shareds = handling.unpack_functions(
        theano_functions, inputs, rank)
    sync.shared_codes = ShmemRawArray('i', len(shareds), SH_ARRAY_TAG, False)

    # Infinite execution loop.
    while True:
        sync.barriers.exec_in.wait()
        if sync.quit.value:
            break
        if sync.exec_type.value == FUNCTION:
            functions[sync.func_code.value]()
        elif sync.exec_type.value == GPU_COMM:
            shared_codes = sync.shared_codes[:sync.n_shareds.value]
            if sync.comm_code.value == BROADCAST:
                for idx in shared_codes:
                    gpu_comm.broadcast(array=shareds[idx].data, root=master_rank)
            elif sync.comm_code.value == REDUCE:
                op = get_op(sync, sync.comm_op.value)
                for idx in shared_codes:
                    gpu_comm.reduce(src=shareds[idx].data, op=op, root=master_rank)
            elif sync.comm_code.value == ALL_REDUCE:
                op = get_op(sync, sync.comm_op.value)
                for idx in shared_codes:
                    gpu_comm.all_reduce(src=shareds[idx].data, op=op, dest=shareds[idx].data)
            elif sync.comm_code.value == ALL_GATHER:
                for idx in shared_codes:
                    # NOTE: Results on the workers are ignored (behaves like
                    # 'gather' to the master).  I think this will put results in
                    # new memory on the worker without affecting the shared
                    # variables, and we don't keep the reference to that memory.
                    gpu_comm.all_gather(shareds[idx].data)
            else:
                raise ValueError("Unrecognized communication type in worker.")

        sync.barriers.exec_out.wait()  # TODO: decide if this barrier is helpful


