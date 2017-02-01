"""
Run theano functions in parallel on multiple GPUs (data parallelism).

This file has everything exposed to the user.
"""

import pickle
import numpy as np
import multiprocessing as mp
import theano
import functools

from variables import Inputs, Shareds, Outputs, SynkFunction
from common import use_gpu, init_gpu_comm
from common import (PKL_FILE, FUNCTION, GPU_COMM, BROADCAST, REDUCE, ALL_REDUCE,
                    ALL_GATHER, GATHER, CPU_COMM, AVG_ALIASES, SCATTER)
from util import (struct, get_n_gpu, build_sync, check_collect, check_op,
                  check_func_scatter, get_worker_reduce_ops,
                  check_shared_var, check_scatter_sources, get_shared_IDs)


# Globals  (only functions exposed to user will use via global access)
g = struct(
    # State
    forked=False,
    distributed=False,
    closed=False,
    # Multiprocessing
    sync=None,
    processes=list(),
    # Theano
    inputs=Inputs(),
    shareds=Shareds(),
    outputs=Outputs(),
    # GPU
    synk_functions=list(),
    n_gpu=None,
    gpu_comm=None,
    master_rank=None,
)


###############################################################################
#                                                                             #
#                           Building Functions.                               #
#                                                                             #
###############################################################################


class Function(SynkFunction):

    _n_gpu = None
    _master_rank = None

    def __init__(self, shared_IDs, output_IDs, g_inputs, g_outputs,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shared_IDs = shared_IDs
        self._output_IDs = output_IDs
        self._name = self._theano_function.name
        self._n_outputs = len(output_IDs)

        # For quick/easy reference during execution.
        self._input_names = [g_inputs.names[i] for i in self._input_IDs]
        self._input_vars = [g_inputs.vars[i] for i in self._input_IDs]
        self._output_to_cpu = [g_outputs.to_cpu[i] for i in self._output_IDs]
        self._output_avg_funcs = [g_outputs.avg_funcs[i] for i in self._output_IDs]

        self._previous_batch_size = None
        self._my_idx = [0, 0]
        self._previous_output_subset = None
        self._n_inputs = len(self._input_IDs)
        self._build_output_subset_shmem()

    @property
    def output_to_cpu(self):
        return self._output_to_cpu

    ###########################################################################
    #                       User callables (use globals g directly)           #

    def __call__(self, *args, **kwargs):
        """
        This needs to:
        1. Share input data.
        2. Signal to workers to start and what to do.
        3. Call the local theano function on data.
        4. Collect result from workers and return it.

        NOTE: Barriers happen INSIDE master function call.
        """
        if not g.distributed:
            raise RuntimeError("Synkhronos functions have not been distributed \
                to workers, can only call Theano function.")
        if g.closed:
            raise RuntimeError("Synkhronos already closed, can only call \
                Theano function.")
        return_shmems = kwargs.pop("return_shmems", False)
        output_subset = kwargs.pop("output_subset", None)
        input_datas = self._order_inputs(g.inputs, args, kwargs)
        input_shmems = self._update_shmems(g.inputs, input_datas)
        output_set = self._share_output_subset(output_subset)
        self._set_worker_signal(g.sync)
        my_inputs = self._get_my_inputs(g.inputs)
        my_results = self._call_theano_function(my_inputs, output_subset)  # always a list
        results = self._collect_results(g.gpu_comm, my_results, output_set)  # returns a list
        self._sync.barriers.exec_out.wait()  # NOTE: Not sure if keeping this--yes
        if return_shmems:
            results += input_shmems  # append list of results with tuple of shmems
        if len(results) == 1:
            results = results[0]
        return results

    def get_input_shmems(self, *args, **kwargs):
        if not g.distributed or g.closed:
            raise RuntimeError("Cannot call this method on inactive synkhronos \
                function.")
        if not args and not kwargs:  # (simply gather existing)
            input_shmems = list()
            for input_ID in self._input_IDs:
                input_shmems.append(g.inputs.shmems[input_ID])
        else:  # (make new ones according to input datas)
            input_datas = self._order_inputs(g.inputs, args, kwargs)
            input_shmems = self._update_shmems(g.inputs, input_datas)
        return input_shmems

    ###########################################################################
    #                     Helpers (not for user)                              #

    def _order_inputs(self, g_inputs, args, kwargs):
        """ Includes basic datatype and ndims checking. """
        if len(args) + len(kwargs) != self._n_inputs:
            raise TypeError("Incorrect number of inputs to synkhronos function.")
        ordered_inputs = list(args)
        if kwargs:
            ordered_inputs += [None] * len(kwargs)
            for key, input_data in kwargs.iteritems():
                if key in self._input_names:
                    idx = self._input_names.index(key)
                elif key in self._input_vars:
                    idx = self._input_vars.index(key)
                else:
                    raise ValueError("Input passed as keyword arg not found  \
                        in inputs (vars or names) of function: ", key)
                if ordered_inputs[idx] is None:
                    ordered_inputs[idx] = input_data
                else:
                    raise ValueError("Received duplicate input args/kwargs: ",
                        key)
        input_datas = g_inputs.check_inputs(self._input_IDs, ordered_inputs)
        return input_datas

    def _share_output_subset(self, output_subset):
        if output_subset != self._previous_output_subset:
            if output_subset is None:
                self._output_subset_shmem[:] = True
            else:
                if not isinstance(output_subset, list):
                    raise TypeError("Optional param output_subset must be a \
                        list of ints.")
                for idx in output_subset:
                    if not isinstance(idx, int):
                        raise TypeError("Optional param output_subset must a \
                            list of ints.")
                    if idx < 0 or idx > self._n_outputs - 1:
                        raise ValueError("Output_subset entry out of range.")
                self._output_subset_shmem[:] = False
                for idx in output_subset:
                    self._output_subset_shmem[idx] = True
            self._previous_output_subset = output_subset
        output_set = [i for i, x in enumerate(self._output_subset_shmem) if x]
        return output_set

    def _update_shmems(self, g_inputs, input_datas):
        self._update_batch_size(g_inputs, input_datas)
        shmems = list()
        for input_data, input_ID in zip(input_datas, self._input_IDs):
            shmems.append(g_inputs.update_shmem(input_ID, input_data))
        return shmems

    def _update_batch_size(self, g_inputs, input_datas):
        if not any(self._inputs_scatter):
            return  # (all inputs broadcast, no data parallel)
        b_size = None
        for input_data, scatter in zip(input_datas, self._inputs_scatter):
            if scatter:
                b_size = input_data.shape[0] if b_size is None else b_size
                if input_data.shape[0] != b_size:
                    raise ValueError("Scatter Inputs of different batch sizes \
                        (using 0-th index).")
        if b_size != self._previous_batch_size:
            assign_idx = int(np.ceil(np.linspace(0, b_size, self._n_gpu + 1)))
            g_inputs.sync.assign_idx[self._ID][:] = assign_idx
            self._my_idx = (assign_idx[self._master_rank],
                            assign_idx[self._master_rank + 1])
            self._previous_batch_size = b_size

    def _get_my_inputs(self, g_inputs):
        s_idx = self._my_idx[0]
        e_idx = self._my_idx[1]
        my_inputs = list()
        for input_ID, scatter in zip(self._input_IDs, self._inputs_scatter):
            if scatter:
                my_inputs.append(g_inputs.shmems[input_ID][s_idx:e_idx])
            else:
                max_idx = g_inputs.sync.max_idx[input_ID]
                my_inputs.apppend(g_inputs.shmem[input_ID][:max_idx])
        return my_inputs

    def _set_worker_signal(self, sync):
        sync.exec_type.value = FUNCTION
        sync.func_ID.value = self._ID
        sync.barriers.exec_in.wait()

    def _collect_results(self, gpu_comm, my_results, output_set):
        """ This one is now clean from global accesses. """
        results = list()
        for idx, r in zip(output_set, my_results):
            mode = self._collect_modes[idx]
            op = self._reduce_ops[idx]
            if mode == "reduce":
                if op in AVG_ALIASES:
                    gpu_comm.reduce(r, op="sum", dest=r)
                    # Then do the average (maybe do in separate loop)
                    r = self._output_avg_funcs[idx](r)
                else:
                    gpu_comm.reduce(r, op=op, dest=r)  # (in-place)
                results.append(r)
            elif mode == "gather":
                res = gpu_comm.all_gather(r)  # TODO: figure out exactly what this returns
                results.append(res)
            elif mode is None:
                results.append(r)
            else:
                raise RuntimeError("Unrecognized collect mode in master function.")
        for idx in output_set:
            if self._outputs_to_cpu[idx]:
                results[idx] = np.array(results[idx])
        return results


def function(inputs, outputs=None,
             collect_modes="reduce", reduce_ops="avg",
             broadcast_inputs=None, scatter_inputs=None,
             **kwargs):
    """
    Call this in the master process when normally creating a theano function.

    What does it need to do:
    1. Create & compile theano function.
       a. Register this function to be pickled later (or just do it now?).
    2. Register the inputs to be made into mp shared variables. (well, no, they
       already will be shared variables, but somehow associate them?)
       a. maybe have the user also input the shared variables here.
    """
    if not g.forked:
        raise RuntimeError("Must fork before making functions for GPU.")
    if g.distributed:
        raise RuntimeError("Cannot make new functions after distributing.")

    inputs_scatter = check_func_scatter(inputs, broadcast_inputs, scatter_inputs)
    collect_modes, reduce_ops = check_collect(outputs, collect_modes, reduce_ops)
    gpu_outputs, output_IDs = g.outputs.register(outputs)
    theano_function = theano.function(inputs, gpu_outputs, **kwargs)
    input_IDs = g.inputs.register_func(theano_function)
    shared_IDs = g.shareds.register_func(theano_function)
    synk_function = Function(ID=len(g.synk_functions),  # Fcn can ID itself
                             theano_function=theano_function,
                             input_IDs=input_IDs,
                             shared_IDs=shared_IDs,
                             output_IDs=output_IDs,
                             inputs_scatter=inputs_scatter,
                             collect_modes=collect_modes,
                             reduce_ops=reduce_ops,
                             g_inputs=g.inputs,
                             g_outputs=g.outputs,
                             )
    g.synk_functions.append(synk_function)
    return synk_function


###############################################################################
#                                                                             #
#                      GPU Collectives.                                       #
#                                                                             #
###############################################################################


###############################################################################
#                           Helpers (not for user)                            #


def gpu_comm_prep(comm_ID, has_op, functions=None, shared_vars=None, op=None):
    if not g.distributed:
        raise RuntimeError("Synk functions not yet distributed-- \
            cannot call comm functions.")
    if g.closed:
        raise RuntimeError("synk already closed--cannot call comm \
            functions.")
    g.sync.exec_type = GPU_COMM
    g.sync.comm_ID = comm_ID
    shared_IDs = get_shared_IDs(g.shareds, functions, shared_vars)
    n_shared = len(shared_IDs)
    g.sync.shared_IDs[:n_shared] = shared_IDs
    g.sync.n_shared.value = n_shared
    return shared_IDs


def reduce_func(g_shareds, gpu_comm, shared_ID, op, in_place=True, dest=None):
    avg = op in AVG_ALIASES
    op = "sum" if avg else op
    src = g_shareds.gpuarrays[shared_ID]
    if in_place:
        gpu_comm.reduce(src, op=op, dest=src)
        if avg:
            g_shareds.avg_functions[shared_ID]()
    else:
        if avg:
            raise ValueError("Cannot use 'average' reduce op if not in-place.")
        return gpu_comm.reduce(src, op=op, dest=dest)  # makes a new gpuarray


def all_reduce_func(g_shareds, gpu_comm, shared_ID, op):
    # workers can't get new arrays; everyone (including master) overwrites src
    avg = op in AVG_ALIASES
    op = "sum" if avg else op
    src = g_shareds.gpuarrays[shared_ID]
    gpu_comm.all_reduce(src, op=op, dest=src)
    if avg:
        g_shareds.avg_functions[shared_ID]()


def broadcast_func(g_shareds, gpu_comm, shared_ID):
    src = g_shareds.gpuarrays[shared_ID]
    gpu_comm.broadcast(src)


def gather_func(g_shareds, gpu_comm, shared_ID, dest=None, nd_up=1):
    src = g_shareds.gpuarrays[shared_ID]
    return gpu_comm.all_gather(src, dest=dest, nd_up=nd_up)


def gpu_comm_function(gpu_comm_func, comm_ID, has_op=False):
    def build_comm_procedure(f):
        @functools.wraps(f)  # (preserves signature and docstring of wrapped)
        def gpu_comm_procedure(functions=None, shared_vars=None, op=None,
                               **kwargs):
            shared_IDs = gpu_comm_prep(comm_ID, functions, shared_vars)
            if has_op:
                op_ID = check_op(op)
                kwargs["op"] = op
                g.sync.comm_op.value = op_ID
            g.sync.barriers.exec_in.wait()
            results = list()
            for shared_ID in shared_IDs:
                r = gpu_comm_func(g.shareds, g.gpu_comm, shared_ID, **kwargs)
                if r is not None:
                    results.append(r)
            results = None if len(results) == 0 else results
            g.sync.barriers.exec_out.wait()
            return results
        return gpu_comm_procedure
    return build_comm_procedure


###############################################################################
#                       User functions                                        #


@gpu_comm_function(broadcast_func, BROADCAST)
def broadcast(functions=None, shared_vars=None):
    """ broadcast docstring """
    pass


@gpu_comm_function(all_reduce_func, ALL_REDUCE, has_op=True)
def all_reduce(functions=None, shared_vars=None, op="avg"):
    """ all_reduce docstring """
    pass


@gpu_comm_function(reduce_func, REDUCE, has_op=True)
def reduce(functions=None, shared_vars=None, op="avg", in_place=True,
           dest=None):
    """ reduce docstring """
    pass


@gpu_comm_function(gather_func, GATHER)
def gather(functions=None, shared_vars=None, dest=None, nd_up=1):
    """ gather docstring """
    pass


def all_gather(source, dest):
    """ all_gather docstring """
    shared_IDs = gpu_comm_prep(ALL_GATHER, shared_vars=[source, dest])
    g.sync.barriers.exec_in.wait()
    src = g.shareds.gpuarrays[shared_IDs[0]]
    dest = g.shareds.gpuarrays[shared_IDs[1]]
    g.gpu_comm.all_gather(src, dest)
    g.sync.barriers.exec_out.wait()


###############################################################################
#                                                                             #
#                         CPU-based Communications                            #
#                                                                             #
###############################################################################


def scatter(shared_var, sources):
    shared_var, shared_ID = check_shared_var(g.shareds, shared_var)
    sources = check_scatter_sources(g.shareds, g.n_gpu, sources, shared_ID)
    if g.shareds.shmems[shared_ID] is None:
        g.shareds.build_shmems(shared_ID, g.n_gpu, g.master_rank)
    for rank, src in enumerate(sources):
        if rank == g.master_rank:
            shared_var.set_value(src)
        else:
            g.shareds.shmems[shared_ID][rank][:] = src
    g.sync.exec_type.value = CPU_COMM
    g.sync.comm_id.value = SCATTER
    g.sync.shared_IDs[0] = shared_ID  # (can only to one per call)
    g.sync.barriers.exec_in.wait()
    g.sync.barriers.exec_out.wait()


###############################################################################
#                                                                             #
#                       Initializing and Exiting.                             #
#                                                                             #
###############################################################################


def fork(n_gpu=None, master_rank=0):
    if g.forked:
        raise RuntimeError("Only fork once.")
    from worker import worker_exec

    n_gpu, master_rank = get_n_gpu(n_gpu, master_rank)
    sync = build_sync(n_gpu)

    for rank in [r for r in range(n_gpu) if r != master_rank]:
        args = (rank, n_gpu, master_rank, sync)
        g.processes.append(mp.Process(target=worker_exec, args=args))
    for p in g.processes:
        p.start()

    import atexit
    atexit.register(close)

    g.forked = True
    g.n_gpu = n_gpu
    g.master_rank = master_rank
    Function._n_gpu = n_gpu
    Function._master_rank = master_rank
    g.sync = sync

    use_gpu(master_rank)
    return n_gpu


def distribute_functions():
    if not g.forked:
        raise RuntimeError("Need to fork before distributing functions.")
    if g.distributed:
        raise RuntimeError("Can distribute only once.")

    # NOTE: pickle all functions together in one list to preserve
    # correspondences among shared variables in different functions.
    g.shareds.set_avg_facs(g.n_gpu)
    pkl_functions = [sf.theano_function for sf in g.synk_functions]
    pkl_functions += g.shareds.avg_funcs
    with open(PKL_FILE, "wb") as f:
        pickle.dump(pkl_functions, f, pickle.HIGHEST_PROTOCOL)

    gpu_comm, comm_id = init_gpu_comm(g.n_gpu, g.master_rank)
    g.sync.dict["comm_id"] = comm_id  # workers need it to join gpu comm clique
    g.sync.n_user_fcns.value = len(g.theano_functions)
    g.sync.dict["collect_modes"] = [fn._collect_modes for fn in g.synk_functions]
    g.sync.dict["reduce_ops"] = get_worker_reduce_ops(g.synk_functions)
    g.sync.dict["inputs_scatter"] = [fn._inputs_scatter for fn in g.synk_functions]
    g.inputs.build_sync(len(g.functions), g.n_gpu)
    g.shareds.build_sync()
    g.sync.distributed.value = True  # let the workers know this part succeeded
    g.sync.barriers.distribute.wait()  # signal workers to receive & join comm
    g.outputs.set_avg_facs(g.n_gpu)
    g.gpu_comm = gpu_comm
    g.distributed = True


def close():
    """ Called automatically on exit, but user can call before. """
    if not g.forked:
        raise Warning("Calling synkhronos.close() before forking does nothing.")
        return
    elif not g.sync.distributed.value:  # (likely closing due to error)
        g.sync.barriers.distribute.wait()  # (Workers will know to exit)
        for p in g.processes:
            p.join()
        g.closed = True
    elif not g.closed:
        g.sync.quit.value = True
        g.sync.barriers.exec_in.wait()
        for p in g.processes:
            p.join()
        g.closed = True
