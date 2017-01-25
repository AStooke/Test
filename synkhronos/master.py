"""
Run theano functions in parallel on multiple GPUs (data parallelism).

This file has everything the master does.
"""

import pickle
import numpy as np
import multiprocessing as mp
import theano
import functools


from Function import SynkFunction
import util
import handling
from util import struct, Inputs, Shareds
from util import check_collect, check_op
from util import (FUNCTION, GPU_COMM, BROADCAST, REDUCE, ALL_REDUCE, ALL_GATHER,
                  PKL_FILE, REDUCE_OPS, SH_ARRAY_TAG, INPUT_TAG_CODES_TAG,
                  ASGN_IDX_TAG, SHMEM_TAG_PRE, COLLECT_MODES)
from shmemarray import ShmemRawArray


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
    theano_functions=list(),
    # GPU
    synk_functions=list(),
    n_gpu=None,
    gpu_comm=None,
    master_rank=None,
)


###############################################################################
#
#                           Building Functions.
#
###############################################################################


def alloc_write_shmem(input_arg, input_code):
    shape = list(input_arg.shape)
    shape[0] = int(np.ceil(shape[0] * 1.05))  # (a little extra)
    # FIXME: this is possibly a bad idea, might hit some max length for tag code?
    tag_code = np.max(g.sync.input_tag_codes) + 1
    shmem = np.ctypeslib.as_array(ShmemRawArray(g.inputs.ctypes[input_code],
                                                int(np.prod(shape)),
                                                SHMEM_TAG_PRE + str(tag_code),
                                                )
                                  ).reshape(shape)
    shmem[:input_arg.shape[0]] = input_arg  # (copy arg data into shared memory buffer)
    g.sync.input_tag_codes[input_code] = tag_code
    g.inputs.shmems[input_code] = shmem
    g.inputs.tags[input_code] = tag_code


class Function(SynkFunction):

    def __init__(self, outputs_to_cpu, *args, **kwargs):
        super(Function).__init__(*args, **kwargs)
        self._call = self._pre_distributed_call
        self._outputs_to_cpu = outputs_to_cpu
        self._previous_batch_size = None
        self._my_idx = None

    def __call__(self, *args, **kwargs):
        self._call(*args, **kwargs)  # What this refers to is set dynamically

    @property
    def outputs_to_cpu(self):
        return self._outputs_to_cpu

    def _set_normal_call(self):
        self._call = self._synk_call

    def _close(self):
        self._call = self._closed_call

    def _share_inputs(self, args):
        if not args:
            return
        my_inputs = list()
        assert isinstance(args, (tuple, list))
        batch_size = args[0].shape[0]
        for arg in args:
            if arg.shape[0] != batch_size:
                raise ValueError("Inputs of different batch sizes (using 0-th index).")
        if batch_size != self._previous_batch_size:
            self._update_batch_size(batch_size)
        for arg, inpt_code in zip(args, self._input_codes):
            shmem = g.inputs.shmems[inpt_code]
            if shmem is None:
                alloc_write_shmem(arg, inpt_code)
            else:
                # check if they are already the same memory (based on first element)
                arg_addr, _ = arg.__array_interface__["data"]
                shmem_addr, _ = shmem.__array_interface__["data"]
                if arg_addr == shmem_addr:
                    if arg.__array_interface__["strides"] is not None:
                        raise ValueError("Cannot accept strided view of existing shared memory as input.")
                else:
                    if arg.shape[1:] != shmem.shape[1:] or batch_size > shmem.shape[0]:
                        # new shape or bigger batch
                        alloc_write_shmem(arg, inpt_code)
                    else:
                        shmem[:batch_size] = arg  # already enough shared memory
            my_inputs.append(arg[self._my_idx[0]:self._my_idx[1]])
        return my_inputs

    def _update_batch_size(self, batch_size):
        assign_idx = np.ceil(np.linspace(0, batch_size, g.n_gpu + 1)).astype(int)
        g.sync.assign_idx[self._code, :] = assign_idx
        self._my_idx = (assign_idx[g.master_rank], assign_idx[g.master_rank + 1])
        self._previous_batch_size = batch_size

    def _set_worker_signal(self):
        self._sync.exec_type.value = FUNCTION
        self._sync.func_code.value = self._code
        self._sync.barriers.exec_in.wait()

    def _reduce_results(self, my_results):
        for r in my_results:
            g.gpu_comm.reduce(r, op=self._reduce_op, dest=r)  # (in-place)
        results = list(my_results)
        return results

    def _gather_results(self, my_results):
        results = list()
        for r in my_results:
            results.append(g.gpu_comm.all_gather(r))  # I think this will return a single GPU Array
        return results

    def _closed_call(self, *args):
        raise RuntimeError("Synkhronos already closed, can only call Theano function.")

    def _pre_distributed_call(self, *args):
        raise RuntimeError("Synkhronos functions have not been distributed to workers, can only call Theano function.")

    def _synk_call(self, *inputs):
        """
        This needs to:
        1. Share input data.
        2. Signal to workers to start and what to do.
        3. Call the local theano function on data.
        4. Collect result from workers and return it.

        NOTE: Barriers happen INSIDE master function call.
        FIXME: handle kwargs?
        """
        my_inputs = self._share_inputs(inputs)
        self._set_worker_signal()
        my_results = self._call_theano_function(my_inputs)  # always a tuple
        results = self._collect_results(my_results)  # returns a list
        for idx in [i for i, val in enumerate(self.outputs_to_cpu) if val]:
            results[idx] = np.array(results[idx])
        results = results[0] if len(results) == 1 else tuple(results)
        self._sync.barriers.exec_out.wait()  # NOTE: Not sure if keeping this--yes
        return results


def function(inputs, outputs=None, updates=None, name=None,
             collect_mode="reduce", reduce_op="avg"):
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

    reduce_op = check_collect(collect_mode, reduce_op)

    # Keep the outputs on the GPU; remember which to eventually send to CPU.
    gpu_outputs, outputs_to_cpu = handling.gpu_outputs(outputs)

    # TODO: Probably still need to do something about updates and givens.
    theano_function = theano.function(inputs=inputs,
                                      outputs=gpu_outputs,
                                      updates=updates,
                                      name=name,
                                      )
    g.theano_functions.append(theano_function)
    input_codes, shared_codes = handling.register_inputs(theano_function,
                                                         g.inputs, g.shareds)
    synk_function = Function(name=name,
                             code=len(g.synk_functions),  # Fcn can ID itself...?
                             theano_function=theano_function,
                             input_codes=input_codes,
                             shared_codes=shared_codes,
                             outputs_to_cpu=outputs_to_cpu,
                             collect_mode=collect_mode,
                             reduce_op=reduce_op
                             )
    g.synk_functions.append(synk_function)
    return synk_function


###############################################################################
#
#                      GPU Collectives.
#
###############################################################################


def get_shared_codes(synk_functions=None, shared_names=None):
    if synk_functions is None and shared_names is None:
        return tuple(range(g.shareds.num))  # default is all shareds
    else:
        # All type and existence checking.
        if not isinstance(synk_functions, (list, tuple)):
            synk_functions = (synk_functions,)
        for synk_fcn in synk_functions:
            if not isinstance(synk_fcn, Function):
                raise TypeError("Expected Synkhronos function(s).")
        if not isinstance(shared_names, (list, tuple)):
            shared_names = (shared_names,)
        for name in shared_names:
            if name not in g.shareds.names:
                raise ValueError("Unrecognized name for shared variable: ", name)

        shared_codes = list()
        for synk_fcn in synk_functions:
            shared_codes += synk_fcn.shared_codes
        for name in shared_names:
            shared_codes.append(g.shareds.names.index(name))
    return tuple(sorted(set(shared_codes)))


def gpu_comm_function(gpu_comm_func, comm_code, has_op=False):
    def build_comm_procedure(f):
        @functools.wraps(f)
        def gpu_comm_procedure(*args, functions=None, shared_names=None, **kwargs):
            if g.closed:
                raise RuntimeError("synk already closed--cannot call comm function.")
            g.sync.exec_type = GPU_COMM
            g.sync.comm_code = comm_code
            shared_codes = get_shared_codes(functions, shared_names)
            n_shared = len(shared_codes)
            g.sync.shared_codes[:n_shared] = shared_codes
            g.sync.n_shared.value = n_shared
            if has_op:
                op, op_code = check_op(kwargs.pop("op", "avg"))
                g.sync.comm_op.value = op_code
                args = (*args, op)
            g.sync.barriers.exec_in.wait()
            results = list()
            for idx in shared_codes:
                r = gpu_comm_func(g.shareds.gpuarrays[idx], *args, **kwargs)
                results.append(r)
            g.sync.barriers.exec_out.wait()
            return results  # (mostly just in case of non-in-place operation)
        return gpu_comm_procedure
    return build_comm_procedure


def reduce_func(src, op, in_place=True, dest=None):
    if in_place:
        g.gpu_comm.reduce(src, op=op, dest=src)
        return src
    else:
        return g.gpu_comm.reduce(src, op=op, dest=dest)  # makes a new gpuarray


def all_reduce_func(src, op):
    # workers can't get new arrays, so everyone (including master) will overwrite src
    g.gpu_comm.all_reduce(src, op=op, dest=src)


@gpu_comm_function(g.gpu_comm.broadcast, BROADCAST)
def broadcast(functions=None, shared_names=None):
    """broadcast docstring"""
    pass


@gpu_comm_function(all_reduce_func, ALL_REDUCE, has_op=True)
def all_reduce(functions=None, shared_names=None, op="avg"):
    """all_reduce docstring"""


@gpu_comm_function(reduce_func, REDUCE, has_op=True)
def reduce(functions=None, shared_names=None, op="avg", in_place=True, dest=None):
    """reduce docstring"""


@gpu_comm_function(g.gpu_comm.all_gather, ALL_GATHER)
def all_gather(functions=None, shared_names=None, dest=None, nd_up=1):
    """all_gather docstring"""


###############################################################################
#
#                       Initializing and Exiting.
#
###############################################################################


def close():
    """
    It needs to:
    1. Signal all the workers to quit.
    2. join() the subprocesses.
    3. Turn off the synk functions.
    """
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
        for synk_fcn in g.synk_functions:
            synk_fcn._close()  # Turn off all the functions.
        g.closed = True


def fork(n_gpu=None, master_rank=None):
    """
    Call this in the master process at the very beginning (possibly before
    importing theano, in order to use different GPUs)

    It needs to do:
    1. Build shared variables according to inputs (sizes & types).
       a. perhaps inputs is a list, which each entry a dict with: name, shape,
          type
       b. I can make this a separate object which the user simply populates, all
          format / layout is controlled.
       c. And maybe having these inputs stored in standardized way, easy to
          refer to them and check that they exist when making a function.
    2. Build synchronization (barriers, semaphores, etc.)
    3. Build whatever comms necessary for later creating functions & shared
       variables in subprocesses.
    4. Set os.environ THEANO_FLAGS for cuda devices, and fork subprocesses
       a. Don't necessarily need to start them now?
    """
    if g.forked:
        raise RuntimeError("Only fork once.")
    from worker import worker_exec

    n_gpu, master_rank = util.n_gpu(n_gpu, master_rank)
    sync = util.build_sync(n_gpu)

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
    g.sync = sync

    util.use_gpu(master_rank)


def distribute_functions():
    """
    Call this in the master after having built all functions.

    It needs to do:
    1. Pickle all the functions.
    2. Make a map of which theano shared variables are the same, between
       functions.
    3. Pass the filename and the theano shared variable map to subprocesses.
    4. Tell subprocesses to get the functions and how to match theano shareds.
       a. (subprocesses will start with same shared values as in master)
    5. Wait until subprocesses are done.
    """
    if not g.forked:
        raise RuntimeError("Need to fork before distributing functions.")
    if g.distributed:
        raise RuntimeError("Can distribute only once.")

    # NOTE: pickle all functions together in one list to preserve
    # correspondences among shared variables in different functions.
    with open(PKL_FILE, "wb") as f:
        pickle.dump(g.theano_functions, f, pickle.HIGHEST_PROTOCOL)

    g.sync.shared_codes = ShmemRawArray('i', g.shareds.num, SH_ARRAY_TAG)
    g.sync.input_tag_codes = ShmemRawArray('i', g.inputs.num, INPUT_TAG_CODES_TAG)
    asgn_rows = len(g.functions)
    asgn_cols = g.n_gpu + 1
    g.sync.assign_idx = np.ctypeslib.as_array(ShmemRawArray(
        'i', asgn_rows * asgn_cols, ASGN_IDX_TAG)
        ).reshape((asgn_rows, asgn_cols))
    gpu_comm, comm_id = util.init_gpu_comm(g.n_gpu, g.master_rank)
    g.sync.dict["comm_id"] = comm_id
    g.sync.distributed.value = True  # let the workers know this part succeeded
    g.sync.barriers.distribute.wait()  # signal workers to receive & join comm
    for synk_fcn in g.synk_functions:
        synk_fcn._set_normal_call()
    g.gpu_comm = gpu_comm
    g.distributed = True
