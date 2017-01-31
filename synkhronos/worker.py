"""
Run theano functions in parallel on multiple GPUs (data parallelism).

This file has everything unique to the workers.
"""


import pickle

from variables import Inputs, Shareds
from common import SynkFunction
from common import use_gpu, init_gpu_comm
from common import (PKL_FILE, FUNCTION, GPU_COMM, BROADCAST, REDUCE, ALL_REDUCE,
                  ALL_GATHER, WORKER_OPS, AVG_ALIASES, CPU_COMM, SCATTER)


class Function(SynkFunction):

    rank = None
    master_rank = None

    def __call__(self, sync, g_inputs, gpu_comm):
        """
        This needs to:
        1. Gather the right inputs from mp shared values.
        2. Execute local theano function on those inputs.
        3. Send results back to master.

        NOTE: Barriers happen OUTSIDE worker function call.
        """
        my_inputs = self.receive_inputs(g_inputs)
        my_results = self._call_theano_function(my_inputs)  # (always returns a tuple)
        self._collect_results(my_results, gpu_comm)

    def receive_inputs(self, g_inputs):
        my_inputs = list()
        assign_idx = g_inputs.sync.assign_idx[self._ID]
        my_idx = (assign_idx[self.rank], assign_idx[self.rank + 1])
        for inpt_ID, shmem, scatter in \
                [(i, g_inputs.shmems[i], s) for i, s in zip(self._input_IDs, self._inputs_scatter)]:
            if g_inputs.sync.input_tags[inpt_ID] != g_inputs.tags[inpt_ID]:
                # Then a new shmem has been allocated, need to get it.
                shape = g_inputs.sync.shapes[inpt_ID]
                tag_ID = g_inputs.sync.input_tags[inpt_ID]
                shmem = g_inputs.alloc_shmem(inpt_ID, shape, tag_ID, False)
                if scatter:
                    my_inputs.append(shmem[my_idx[0]:my_idx[1]])
                else:
                    my_inputs.append(shmem[:g_inputs.sync.max_idx[inpt_ID]])

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


def unpack_functions(theano_functions, sync_dict, n_fcn):
    """
    Worker will recover shared variables in the same order as the master
    committed them, so they will have the same ID (index).
    """
    collect_modes_all = sync_dict["collect_modes"]
    reduce_ops_all = sync_dict["reduce_ops_all"]
    inputs_scatter_all = sync_dict["inputs_scatter"]
    synk_functions = list()
    g_inputs = Inputs()
    g_shareds = Shareds()
    for idx, fcn in enumerate(theano_functions[:n_fcn]):
        input_IDs, _, _ = g_inputs.register_func(fcn)
        shared_IDs = g_shareds.register_func(fcn, build_avg_func=False)
        synk_functions.append(Function(name=fcn.name,
                                       ID=idx,
                                       theano_function=fcn,
                                       input_IDs=input_IDs,
                                       inputs_scatter=inputs_scatter_all[idx],
                                       shared_IDs=shared_IDs,
                                       collect_modes=collect_modes_all[idx],
                                       reduce_ops=reduce_ops_all[idx],
                                       )
                              )
    g_shareds.avg_functions = theano_functions[n_fcn:]
    return synk_functions, g_inputs, g_shareds


def receive_distribution(rank, n_gpu, sync):
    if not sync.distributed.value:
        return  # (master closed before distributing functions--an error)
    sync_dict = sync.dict.copy()  # (retrieve it as a normal dict)
    gpu_comm = init_gpu_comm(n_gpu, rank, sync_dict["comm_id"])
    with open(PKL_FILE, "rb") as f:
        theano_functions = pickle.load(f)  # should be all in one list
    # Might have the last worker delete the pkl file.
    synk_functions, g_inputs, g_shareds = \
        unpack_functions(theano_functions, sync_dict, sync.n_user_fcns.value)
    # g_shareds.unpack_avg_facs()  # (only needed to change avg_fac later)
    g_inputs.build_sync(len(synk_functions), n_gpu, False)
    g_shareds.build_sync(False)
    return synk_functions, g_inputs, g_shareds, gpu_comm


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
    synk_functions, g_inputs, g_shareds, gpu_comm = \
        receive_distribution(rank, n_gpu, sync)
    Function.rank = rank  # endow all functions
    Function.master_rank = master_rank

    # Infinite execution loop.
    while True:
        sync.barriers.exec_in.wait()
        if sync.quit.value:
            break
        if sync.exec_type.value == FUNCTION:
            synk_functions[sync.func_ID.value](sync, g_inputs, gpu_comm)
        elif sync.exec_type.value == GPU_COMM:
            shared_IDs = sync.shared_IDs[:sync.n_shareds.value]
            comm_ID = sync.comm_ID.value
            op = WORKER_OPS.get(sync.comm_op.value, None)  # only for (all)reduce
            if op is None:
                raise RuntimeError("incorrect op code in worker.")
            for idx in shared_IDs:
                if comm_ID == BROADCAST:
                    gpu_comm.broadcast(g_shareds.gpuarrays[idx], root=master_rank)
                elif comm_ID == REDUCE:
                    op = "sum" if op in AVG_ALIASES else op
                    gpu_comm.reduce(g_shareds.gpuarrays[idx], op=op,
                                    root=master_rank)
                elif comm_ID == ALL_REDUCE:
                    avg = op in AVG_ALIASES
                    op = "sum" if avg else op
                    gpu_comm.all_reduce(g_shareds.gpuarrays[idx], op=op,
                            dest=g_shareds.gpuarrays[idx])
                    # TODO: possibly move this to its own loop over idx
                    if avg:
                        g_shareds.avg_functions[idx]()
                elif comm_ID == ALL_GATHER:
                    # NOTE: Results on the workers are ignored (behaves like
                    # 'gather' to the master).  I think this will put results in
                    # new memory on the worker without affecting the shared
                    # variables, and we don't keep the reference to that memory.
                    gpu_comm.all_gather(g_shareds.gpuarrays[idx])
                else:
                    raise RuntimeError("Unrecognized GPU communication type in \
                        worker.")
        elif sync.exec_type.value == CPU_COMM:
            shrd_ID = sync.shared_IDs[0]
            comm_ID = sync.comm_ID.value
            if comm_ID == SCATTER:
                if g_shareds.shmems[shrd_ID] is None:
                    g_shareds.alloc_shmem(shrd_ID, rank, False)
                g_shareds.vars[shrd_ID].set_value(g_shareds.shmems[shrd_ID])
            else:
                raise RuntimeError("Unrecognized CPU comm type in worker.")
        else:
            raise RuntimeError("Unrecognized execution type in worker.")

        sync.barriers.exec_out.wait()  # TODO: decide if this barrier is helpful--yes


