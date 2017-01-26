"""
Run theano functions in parallel on multiple GPUs (data parallelism).

This file has everything the master does.
"""

import pickle
import numpy as np
import multiprocessing as mp
import theano
import functools

from common import struct, Inputs, Shareds, SynkFunction
from common import (use_gpu, init_gpu_comm, register_inputs, build_vars_sync,
                    allocate_shmem)
from common import (PKL_FILE, FUNCTION, GPU_COMM, BROADCAST, REDUCE, ALL_REDUCE,
                    ALL_GATHER, COLLECT_MODES, REDUCE_OPS)


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
#                                                                             #
#                           Building Functions.                               #
#                                                                             #
###############################################################################


def alloc_write_shmem(input_arg, input_ID):
    shape = list(input_arg.shape)
    shape[0] = int(np.ceil(shape[0] * 1.05))  # ( a little extra)
    tag_ID = np.max(g.sync.input_tag_IDs) + 1
    shmem = allocate_shmem(input_ID, g.inputs, shape, tag_ID)
    shmem[:input_arg.shape[0]] = input_arg  # (copy arg data into shared buffer)
    g.sync.input_tag_IDs[input_ID] = tag_ID
    return shmem


class Function(SynkFunction):

    def __init__(self, outputs_to_cpu, input_names, *args, **kwargs):
        super(Function).__init__(*args, **kwargs)
        self._call = self._pre_distributed_call
        self._outputs_to_cpu = outputs_to_cpu
        self._input_names = input_names
        self._previous_batch_size = None
        self._my_idx = None
        self._n_inputs = len(self._input_IDs)

    def __call__(self, *args, **kwargs):
        self._call(*args, **kwargs)  # What this refers to is set dynamically

    @property
    def outputs_to_cpu(self):
        return self._outputs_to_cpu

    def _set_normal_call(self):
        self._call = self._synk_call

    def _close(self):
        self._call = self._closed_call

    def _order_inputs(self, *args, **kwargs):
        ordered_inputs = list(args)
        if not kwargs:
            output_subset = None
        else:
            output_subset = kwargs.pop("output_subset", None)
            if output_subset is not None:
                raise NotImplementedError
            ordered_inputs += [None] * len(kwargs)
            try:
                for name, val in kwargs.iteritems():
                    ordered_inputs[self._input_names.index(name)] = val
            except ValueError as e:
                raise e("Input passed as keyword arg not found in input names of function.")
        if len(ordered_inputs) != self._n_inputs:
            raise TypeError("Incorrect number of inputs to synkhronos function.")
        return ordered_inputs, output_subset

    def _update_batch_size(self, ordered_inputs):
        batch_size = ordered_inputs[0].shape[0]
        for inpt in ordered_inputs[1:]:
            if inpt.shape[0] != batch_size:
                raise ValueError("Inputs of different batch sizes (using 0-th index).")
        if batch_size != self._previous_batch_size:
            assign_idx = np.ceil(np.linspace(0, batch_size, g.n_gpu + 1)).astype(int)
            g.sync.assign_idx[self._ID, :] = assign_idx
            self._my_idx = (assign_idx[g.master_rank], assign_idx[g.master_rank + 1])
            self._previous_batch_size = batch_size

    def _update_shmem(self, inpt, inpt_ID):
        shmem = g.inputs.shmems[inpt_ID]
        if shmem is None:
            shmem = alloc_write_shmem(inpt, inpt_ID)
        else:
            # check if they are already the same memory (based on first element)
            inpt_addr, _ = inpt.__array_interface__["data"]
            shmem_addr, _ = shmem.__array_interface__["data"]
            if inpt_addr == shmem_addr:
                if inpt.__array_interface__["strides"] is not None:
                    raise ValueError("Cannot accept strided view of existing shared memory as input.")
            else:
                if inpt.shape[1:] != shmem.shape[1:] or inpt.shape[0] > shmem.shape[0]:
                    # new shape or bigger batch
                    shmem = alloc_write_shmem(inpt, inpt_ID)
                else:
                    shmem[:inpt.shape[0]] = inpt  # already enough shared memory
        return shmem

    def _share_inputs(self, *args, **kwargs):
        """
        Can separately be used to allocate, write, and return input shared
        memory arrays without executing the function.
        """
        if not args and not kwargs:
            return None, None, None
        ordered_inputs, output_subset = self._order_inputs(*args, **kwargs)
        self._update_batch_size(ordered_inputs)
        my_inputs = list()
        input_shmems = list()
        for inpt, inpt_ID in zip(ordered_inputs, self._input_IDs):
            shmem = self._update_shmem(inpt, inpt_ID)
            my_inputs.append(shmem[self._my_idx[0]:self._my_idx[1]])
            input_shmems.append(shmem)
        return tuple(my_inputs), output_subset, input_shmems

    def _set_worker_signal(self):
        self._sync.exec_type.value = FUNCTION
        self._sync.func_ID.value = self._ID
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

    def _synk_call(self, *args, **kwargs):
        """
        This needs to:
        1. Share input data.
        2. Signal to workers to start and what to do.
        3. Call the local theano function on data.
        4. Collect result from workers and return it.

        NOTE: Barriers happen INSIDE master function call.
        """
        return_shmems = kwargs.pop("return_shmems", False)
        my_inputs, output_subset, input_shmems = self._share_inputs(*args, **kwargs)  # ugh now i have to get the right index
        self._set_worker_signal()
        my_results = self._call_theano_function(my_inputs, output_subset)  # always a list
        results = self._collect_results(my_results)  # returns a list
        for idx in [i for i, val in enumerate(self.outputs_to_cpu) if val]:
            results[idx] = np.array(results[idx])
        self._sync.barriers.exec_out.wait()  # NOTE: Not sure if keeping this--yes
        if return_shmems:
            results += input_shmems  # append the list of results with tuple of shmems
        if len(results) == 1:
            results = results[0]
        return results

    def get_input_shmems(self, *args, **kwargs):
        if self._call is not self._synk_call:
            raise RuntimeError("Cannot call this method on inactive synkhronos function.")
        input_shmems = list()
        if not args and not kwargs:
            # gather existing
            for inpt_ID in self._input_IDs:
                input_shmems.append(g.inputs.shmems[inpt_ID])
        else:
            # make new ones according to inputs
            _, _, input_shmems = self._share_inputs(*args, **kwargs)
        return input_shmems


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
    gpu_outputs, outputs_to_cpu = vars_as_gpu(outputs)

    # TODO: Probably still need to do something about updates and givens.
    theano_function = theano.function(inputs=inputs,
                                      outputs=gpu_outputs,
                                      updates=updates,
                                      name=name,
                                      )
    g.theano_functions.append(theano_function)
    input_IDs, shared_IDs, input_names = register_inputs(theano_function,
                                                         g.inputs,
                                                         g.shareds,
                                                         )
    synk_function = Function(name=name,
                             ID=len(g.synk_functions),  # Fcn can ID itself...?
                             theano_function=theano_function,
                             input_IDs=input_IDs,
                             input_names=input_names,
                             shared_IDs=shared_IDs,
                             outputs_to_cpu=outputs_to_cpu,
                             collect_mode=collect_mode,
                             reduce_op=reduce_op
                             )
    g.synk_functions.append(synk_function)
    return synk_function


###############################################################################
#                                                                             #
#                      GPU Collectives.                                       #
#                                                                             #
###############################################################################


def get_shared_IDs(synk_functions=None, shared_names=None):
    if synk_functions is None and shared_names is None:
        return tuple(range(g.shareds.num))  # default is all shareds
    else:
        # Type and existence checking.
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

        shared_IDs = list()
        for synk_fcn in synk_functions:
            shared_IDs += synk_fcn.shared_IDs
        for name in shared_names:
            shared_IDs.append(g.shareds.names.index(name))
    return tuple(sorted(set(shared_IDs)))


def gpu_comm_function(gpu_comm_func, comm_ID, has_op=False):
    def build_comm_procedure(f):
        @functools.wraps(f)  # (preserves signature and docstring of wrapped)
        def gpu_comm_procedure(functions=None, shared_names=None, op=None, **kwargs):
            if g.closed:
                raise RuntimeError("synk already closed--cannot call comm function.")
            g.sync.exec_type = GPU_COMM
            g.sync.comm_ID = comm_ID
            shared_IDs = get_shared_IDs(functions, shared_names)
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
                r = gpu_comm_func(g.shareds.gpuarrays[idx], **kwargs)
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
    return src


@gpu_comm_function(g.gpu_comm.broadcast, BROADCAST)
def broadcast(functions=None, shared_names=None):
    """broadcast docstring"""
    pass


@gpu_comm_function(all_reduce_func, ALL_REDUCE, has_op=True)
def all_reduce(functions=None, shared_names=None, op="avg"):
    """all_reduce docstring"""
    pass


@gpu_comm_function(reduce_func, REDUCE, has_op=True)
def reduce(functions=None, shared_names=None, op="avg", in_place=True, dest=None):
    """reduce docstring"""
    pass


@gpu_comm_function(g.gpu_comm.all_gather, ALL_GATHER)
def all_gather(functions=None, shared_names=None, dest=None, nd_up=1):
    """all_gather docstring"""
    pass


###############################################################################
#                                                                             #
#                       Initializing and Exiting.                             #
#                                                                             #
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


def fork(n_gpu=None, master_rank=0):
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

    n_gpu, master_rank = n_gpu(n_gpu, master_rank)
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
    g.sync = sync

    use_gpu(master_rank)


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

    g.sync.vars = build_vars_sync(g.inputs, g.shareds, len(g.functions), g.n_gpu)
    gpu_comm, comm_id = init_gpu_comm(g.n_gpu, g.master_rank)
    g.sync.dict["comm_id"] = comm_id  # workers need this to join gpu comm clique
    g.sync.distributed.value = True  # let the workers know this part succeeded
    g.sync.barriers.distribute.wait()  # signal workers to receive & join comm
    for synk_fcn in g.synk_functions:
        synk_fcn._set_normal_call()
    g.gpu_comm = gpu_comm
    g.distributed = True


###############################################################################
#                                                                             #
#                       Utilities (no accesses to global g)                   #
#                                                                             #
###############################################################################


def n_gpu_getter(mp_n_gpu):
    """
    Call in a subprocess because it prevents future subprocesses from using GPU.
    """
    from pygpu import gpuarray
    mp_n_gpu.value = gpuarray.count_devices("cuda", 0)


def n_gpu(n_gpu, master_rank):
    if n_gpu is not None:
        n_gpu = int(n_gpu)
    master_rank = int(master_rank)

    if n_gpu is None:
        #  Detect the number of devices present and use all.
        mp_n_gpu = mp.RawValue('i', 0)
        p = mp.Process(target=n_gpu_getter, args=(mp_n_gpu))
        p.start()
        p.join()
        n_gpu = mp_n_gpu.value
        if n_gpu == 0:
            raise RuntimeError("No cuda GPU detected by pygpu.")
        elif n_gpu == 1:
            raise RuntimeWarning("Only one GPU detected; undetermined behavior (but I could make it revert to regular Theano?)")
        else:
            print("Detected and attempting to use {} GPUs.".format(n_gpu))

    if master_rank not in list(range(n_gpu)):
        raise ValueError("Invalid value for master rank: ", master_rank)

    return n_gpu, master_rank


def build_sync(n_gpu):
    from ctypes import c_bool

    mgr = mp.Manager()
    dictionary = mgr.dictionary()
    barriers = struct(
        distribute=mp.Barrier(n_gpu),
        exec_in=mp.Barrier(n_gpu),
        exec_out=mp.Barrier(n_gpu),
    )
    sync = struct(
        dict=dictionary,  # use for setup e.g. Clique comm_id; serializes.
        quit=mp.RawValue(c_bool, False),
        distributed=mp.RawValue(c_bool, False),
        exec_type=mp.RawValue('i', 0),
        func_ID=mp.RawValue('i', 0),
        comm_ID=mp.RawValue('i', 0),
        n_shared=mp.RawValue('i', 0),
        barriers=barriers,
    )
    return sync


def check_collect(collect_mode, reduce_op):
    if collect_mode not in COLLECT_MODES:
        raise ValueError("Unrecognized collect_mode: ", collect_mode)
    if collect_mode == "reduce":
        if reduce_op not in REDUCE_OPS:
            raise ValueError("Unrecognized reduce_op: ", reduce_op)
    else:
        reduce_op = None
    return reduce_op


def check_op(op):
    if op not in REDUCE_OPS:
        raise ValueError("Unrecognized reduction operator: ", op,
            ", must be one of: ", [k for k in REDUCE_OPS.keys()])
    elif op in ["avg", "average"]:
        raise NotImplementedError
    return REDUCE_OPS[op]


def vars_as_gpu(variables):
    """
    Change all vars to be on GPU, if not already.  Record which were
    previously on CPU (so they can be transfered after collecting, in the case
    of outputs).
    """
    if variables is None:
        return None, None
    else:
        from theano.gpuarray.type import GpuArrayVariable
        if not isinstance(variables, (list, tuple)):
            variables = (variables,)
        variables = list(variables)
        variables_to_cpu = list()
        for idx, var in enumerate(variables):
            if isinstance(var, GpuArrayVariable):
                variables_to_cpu.append(False)
            else:
                variables_to_cpu.append(True)
                variables[idx] = var.transfer(None)
        return tuple(variables), tuple(variables_to_cpu)
