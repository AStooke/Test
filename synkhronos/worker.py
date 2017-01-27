"""
Run theano functions in parallel on multiple GPUs (data parallelism).

This file has everything the workers do.
"""


import pickle

from common import Inputs, Shareds, SynkFunction
from common import (use_gpu, init_gpu_comm, build_vars_sync, register_inputs,
                    allocate_shmem)
from common import (PKL_FILE, FUNCTION, GPU_COMM, BROADCAST, REDUCE, ALL_REDUCE,
                  ALL_GATHER, WORKER_OPS)


def get_op(sync, op_ID):
    if op_ID not in WORKER_OPS:
        raise ValueError("Unrecognized reduce operation in worker.")
    op = WORKER_OPS[op_ID]
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
        assign_idx = self.sync.vars.assign_idx[self._ID]
        my_idx = (assign_idx[self.rank], assign_idx[self.rank + 1])
        for inpt_ID, shmem in \
                [(i, all_inputs.shmems[i]) for i in self.input_IDs]:
            if sync.vars.input_tags[inpt_ID] != all_inputs.tags[inpt_ID]:
                # Then a new shmem has been allocated, need to get it.
                shmem = allocate_shmem(input_ID=inpt_ID,
                                       inputs_global=all_inputs,
                                       shape=sync.vars.shapes[inpt_ID],
                                       tag_ID=sync.vars.input_tags[inpt_ID],
                                       create=False,
                                       )
                all_inputs.tags[inpt_ID] = sync.vars.input_tags[inpt_ID]
            my_inputs.append(shmem[my_idx[0]:my_idx[1]])
        return tuple(my_inputs)

    def _collect_results(self, my_results, gpu_comm):
        for r, mode, op in zip(my_results, self.collect_modes, self.reduce_ops):
            if mode == "reduce":
                gpu_comm.reduce(r, op=op, root=self.master_rank)
            elif mode == "gather":
                gpu_comm.all_gather(r)
            elif mode is not None:
                raise RuntimeError("Unrecognized collect mode in worker function.")

        # TODO worker needs to know the reduce ops and collect modes...can pass
        # these through the dictionary.


def unpack_functions(theano_functions, collect_modes_all, reduce_ops_all, n_fcn):
    """
    Worker will recover shared variables in the same order as the master
    committed them, so they will have the same ID (index).
    """
    from worker import Function

    synk_functions = list()
    inputs = Inputs()
    shareds = Shareds()
    for idx, fcn in enumerate(theano_functions[:n_fcn]):
        input_IDs, shared_IDs, _ = register_inputs(fcn, inputs, shareds)
        synk_functions.append(Function(name=fcn.name,
                                       ID=idx,
                                       theano_function=fcn,
                                       input_IDs=input_IDs,
                                       shared_IDs=shared_IDs,
                                       collect_modes=collect_modes_all[idx],
                                       reduce_ops=reduce_ops_all[idx],
                                       )
                              )
    shareds.avg_functions = theano_functions[n_fcn:]
    return synk_functions, inputs, shareds


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
    use_gpu(rank)

    # Receive functions.
    sync.barriers.distribute.wait()
    if not sync.distributed.value:
        return  # (master closed before distributing functions--an error)
    sync_dict = sync.dict.copy()  # (retrieve it as a normal dict)
    gpu_comm = init_gpu_comm(n_gpu, rank, sync_dict["comm_id"])
    with open(PKL_FILE, "rb") as f:
        theano_functions = pickle.load(f)  # should be all in one list
    # Might have the last worker delete the pkl file.
    synk_functions, inputs, shareds = \
        unpack_functions(theano_functions,
                         sync_dict["collect_modes"],
                         sync_dict["reduce_ops"],
                         sync.n_user_fcns.value,
                         )
    sync.vars = build_vars_sync(inputs, shareds, len(synk_functions), n_gpu,
                                False)
    Function.rank = rank  # endow all functions
    Function.master_rank = master_rank

    # Infinite execution loop.
    while True:
        sync.barriers.exec_in.wait()
        if sync.quit.value:
            break
        if sync.exec_type.value == FUNCTION:
            synk_functions[sync.func_ID.value](sync, inputs, gpu_comm)
        elif sync.exec_type.value == GPU_COMM:
            shared_IDs = sync.shared_IDs[:sync.n_shareds.value]
            comm_ID = sync.comm_ID.value
            op = get_op(sync, sync.comm_op.value)  # (might not be used)
            for idx in shared_IDs:
                if comm_ID == BROADCAST:
                    gpu_comm.broadcast(shareds.gpuarrays[idx], root=master_rank)
                elif comm_ID == REDUCE:
                    gpu_comm.reduce(shareds.gpuarrays[idx], op=op,
                                    root=master_rank)
                elif comm_ID == ALL_REDUCE:
                    gpu_comm.all_reduce(shareds.gpuarrays[idx], op=op,
                            dest=shareds.gpuarrays[idx])
                elif comm_ID == ALL_GATHER:
                    # NOTE: Results on the workers are ignored (behaves like
                    # 'gather' to the master).  I think this will put results in
                    # new memory on the worker without affecting the shared
                    # variables, and we don't keep the reference to that memory.
                    gpu_comm.all_gather(shareds.gpuarrays[idx])
                else:
                    raise ValueError("Unrecognized communication type in \
                        worker.")

        sync.barriers.exec_out.wait()  # TODO: decide if this barrier is helpful--yes


