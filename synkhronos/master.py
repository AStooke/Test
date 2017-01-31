"""
Run theano functions in parallel on multiple GPUs (data parallelism).

This file has everything exposed to the user.
"""

import pickle
import numpy as np
import multiprocessing as mp
import theano
import functools

from variables import Inputs, Shareds, Outputs
from common import struct, SynkFunction
from common import use_gpu, init_gpu_comm
from common import (PKL_FILE, FUNCTION, GPU_COMM, BROADCAST, REDUCE, ALL_REDUCE,
                    ALL_GATHER, CPU_COMM, AVG_ALIASES, SCATTER)
from util import (get_n_gpu, build_sync, check_collect, check_op,
                  check_inputs_scatter, get_worker_reduce_ops,
                  check_shared_var, check_scatter_sources, get_shared_IDs)


# globals
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
    theano_functions=list(),
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

    def __init__(self, input_names, input_vars, outputs_to_cpu, output_avg_funcs,
                 *args, **kwargs):
        super(Function).__init__(*args, **kwargs)
        self._call = self._pre_distributed_call
        self._input_names = input_names
        self._input_vars = input_vars
        self._outputs_to_cpu = outputs_to_cpu
        self._output_avg_funcs = output_avg_funcs
        self._previous_batch_size = None
        self._my_idx = [0, 0]
        self._n_inputs = len(self._input_IDs)

    @property
    def outputs_to_cpu(self):
        return self._outputs_to_cpu

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
        input_datas, output_subset = self._order_inputs(g_inputs, *args, **kwargs)
        input_shmems = self._update_shmems(g_inputs, input_datas)
        self._set_worker_signal(g.sync)
        my_inputs = self._get_my_inputs(input_shmems, input_datas)
        my_results = self._call_theano_function(my_inputs, output_subset)  # always a list
        results = self._collect_results(g.gpu_comm, my_results)  # returns a list
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
            input_datas, _ = self._order_inputs(g_inputs, *args, **kwargs)
            input_shmems = self._update_shmems(g.inputs, input_datas)
        return input_shmems

    ###########################################################################
    #                     Helpers (not for user)                              #

    def _order_inputs(self, g_inputs, *args, **kwargs):
        """ Includes basic datatype and ndims checking. """
        ordered_inputs = list(args)
        if not kwargs:
            output_subset = None
        else:
            output_subset = kwargs.pop("output_subset", None)
            if output_subset is not None:
                raise NotImplementedError
            ordered_inputs += [None] * len(kwargs)
            for key, input_data in kwargs.iteritems():
                if key in self._input_names:
                    ordered_inputs[self._input_names.index(key)] = input_data
                elif key in self._input_vars:
                    ordered_inputs[self._input_vars.index(key)] = input_data
                else:
                    raise ValueError("Input passed as keyword arg not found  \
                        in inputs (vars or names) of function: ", key)
        if len(ordered_inputs) != self._n_inputs:
            raise TypeError("Incorrect number of inputs to synkhronos function.")
        input_datas = g_inputs.check_inputs(self._input_IDs, ordered_inputs)
        return input_datas, output_subset

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

    def _get_my_inputs(self, input_shmems, input_datas):
        input_lengths = [input_data.shape[0] for input_data in input_datas]
        s_idx = self._my_idx[0]
        e_idx = self._my_idx[1]
        my_inputs = list()
        for shmem, scatter, length in \
                zip(input_shmems, self._inputs_scatter, input_lengths):
            if scatter:
                my_inputs.append(shmem[s_idx:e_idx])
            else:
                my_inputs.append(shmem[:length])  # no more references to input_data..?
        return my_inputs

    def _set_worker_signal(self, sync):
        sync.exec_type.value = FUNCTION
        sync.func_ID.value = self._ID
        sync.barriers.exec_in.wait()

    def _collect_results(self, gpu_comm, my_results):
        """ This one is now clean from global accesses. """
        results = list()
        for (idx, r), mode, op in zip(enumerate(my_results),
                                      self.collect_modes, self.reduce_ops):
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
        for idx, to_cpu in enumerate(self._outputs_to_cpu):
            if to_cpu:
                results[idx] = np.array(results[idx])
        return results


def function(inputs, outputs=None, updates=None, name=None,
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

    inputs_scatter = check_inputs_scatter(inputs, broadcast_inputs, scatter_inputs)
    collect_modes, reduce_ops = check_collect(outputs, collect_modes, reduce_ops)
    gpu_outputs, output_avg_funcs, outputs_to_cpu = g.outputs.register(outputs)
    theano_function = theano.function(inputs=inputs,
                                      outputs=gpu_outputs,
                                      updates=updates,
                                      name=name,
                                      **kwargs,
                                      )
    g.theano_functions.append(theano_function)
    input_IDs, input_names, input_vars = g.inputs.register_func(theano_function)
    shared_IDs = g.shareds.register_func(theano_function)
    synk_function = Function(name=name,
                             ID=len(g.synk_functions),  # Fcn can ID itself
                             theano_function=theano_function,
                             input_IDs=input_IDs,
                             input_names=input_names,
                             input_vars=input_vars,
                             inputs_scatter=inputs_scatter,
                             shared_IDs=shared_IDs,
                             output_IDs=output_IDs,
                             output_avg_funcs=output_avg_funcs,
                             outputs_to_cpu=outputs_to_cpu,
                             collect_modes=collect_modes,
                             reduce_ops=reduce_ops,
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


def reduce_func(g_shareds, gpu_comm, idx, op, in_place=True, dest=None):
    avg = op in AVG_ALIASES
    op = "sum" if avg else op
    src = g_shareds.gpuarrays[idx]
    if in_place:
        gpu_comm.reduce(src, op=op, dest=src)
        if avg:
            g_shareds.avg_functions[idx]()
        return g_shareds.vars[idx]
    else:
        if avg:
            raise NotImplementedError  # Because I don't know what comes out.
        return gpu_comm.reduce(src, op=op, dest=dest)  # makes a new gpuarray


def all_reduce_func(g_shareds, gpu_comm, idx, op):
    # workers can't get new arrays; everyone (including master) overwrites src
    avg = op in AVG_ALIASES
    op = "sum" if avg else op
    src = g_shareds.gpuarrays[idx]
    gpu_comm.all_reduce(src, op=op, dest=src)
    if avg:
        g_shareds.avg_functions[idx]()
    return g_shareds.vars[idx]


def broadcast_func(g_shareds, gpu_comm, idx):
    src = g_shareds.gpuarrays[idx]
    gpu_comm.broadcast(src)
    return g_shareds.vars[idx]


def all_gather_func(g_shareds, gpu_comm, idx, dest=None, nd_up=1):
    src = g_shareds.gpuarrays[idx]
    # Still not sure what this returns.
    return gpu_comm.all_gather(src, dest=dest, nd_up=nd_up)


def gpu_comm_function(gpu_comm_func, comm_ID, has_op=False):
    def build_comm_procedure(f):
        @functools.wraps(f)  # (preserves signature and docstring of wrapped)
        def gpu_comm_procedure(functions=None, shared_vars=None, op=None,
                               **kwargs):
            if g.closed:
                raise RuntimeError("synk already closed--cannot call comm \
                    function.")
            g.sync.exec_type = GPU_COMM
            g.sync.comm_ID = comm_ID
            shared_IDs = get_shared_IDs(g.shareds, functions, shared_vars)
            n_shared = len(shared_IDs)
            g.sync.shared_IDs[:n_shared] = shared_IDs
            g.sync.n_shared.value = n_shared
            if has_op:
                op_ID = check_op(op)
                kwargs["op"] = op
                g.sync.comm_op.value = op_ID
            g.sync.barriers.exec_in.wait()
            results = list()
            for idx in shared_IDs:
                r = gpu_comm_func(g.shareds, g.gpu_comm, idx, **kwargs)
                results.append(r)
            g.sync.barriers.exec_out.wait()
            return results  # (mostly just in case of non-in-place operation)
        return gpu_comm_procedure
    return build_comm_procedure


###############################################################################
#                       User functions                                        #


@gpu_comm_function(broadcast_func, BROADCAST)
def broadcast(functions=None, shared_vars=None):
    """broadcast docstring"""
    pass


@gpu_comm_function(all_reduce_func, ALL_REDUCE, has_op=True)
def all_reduce(functions=None, shared_vars=None, op="avg"):
    """all_reduce docstring"""
    pass


@gpu_comm_function(reduce_func, REDUCE, has_op=True)
def reduce(functions=None, shared_vars=None, op="avg", in_place=True,
           dest=None):
    """reduce docstring"""
    pass


@gpu_comm_function(all_gather_func, ALL_GATHER)
def all_gather(functions=None, shared_vars=None, dest=None, nd_up=1):
    """all_gather docstring"""
    pass


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


def close():
    if not g.forked:
        return
    elif not g.sync.distributed.value:  # (will be closing due to error)
        g.sync.barriers.distribute.wait()  # (Workers will know to exit)
        for p in g.processes:
            p.join()
    elif not g.closed:
        g.sync.quit.value = True
        g.sync.barriers.exec_in.wait()
        for p in g.processes:
            p.join()
        g.closed = True


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
    pkl_functions = g.theano_functions + g.shareds.avg_functions  # (combine lists)
    with open(PKL_FILE, "wb") as f:
        pickle.dump(pkl_functions, f, pickle.HIGHEST_PROTOCOL)

    gpu_comm, comm_id = init_gpu_comm(g.n_gpu, g.master_rank)
    g.sync.dict["comm_id"] = comm_id  # workers need it to join gpu comm clique
    g.sync.n_user_fcns.value = len(g.theano_functions)
    g.sync.dict["collect_modes"] = [fn._collect_modes for fn in g.synk_functions]
    g.sync.dict["reduce_ops"] = get_worker_reduce_ops(g.synk_functions)
    g.sync.dict["inputs_scatter"] = [fn._inputs_scatter for fn in g.synk_functions]
    g.sync.distributed.value = True  # let the workers know this part succeeded
    g.sync.barriers.distribute.wait()  # signal workers to receive & join comm
    g.inputs.build_sync(len(g.functions), g.n_gpu)
    g.outputs.set_avg_facs(g.n_gpu)
    g.shareds.build_sync()
    g.gpu_comm = gpu_comm
    g.distributed = True
