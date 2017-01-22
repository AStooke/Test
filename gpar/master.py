"""
Run theano functions in parallel on multiple GPUs (data parallelism).

This file has everything the master does.
"""

import pickle
import numpy as np
import multiprocessing as mp
import theano


from Input import Input
from Function import MasterFunction as Function
import util
import handling
from util import struct
from util import (FUNCTION, GPU_COMM, BROADCAST, REDUCE, ALL_REDUCE,
                  PKL_FILE, MASTER_RANK, OPS, SH_ARRAY_TAG)
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
    inputs=list(),  # list of shmemarray objects
    named_inputs=dict(),  # name --> position in list
    shareds=list(),  # list of theano.link.Container objects
    named_shareds=dict(),  # name --> position in list
    theano_functions=list(),
    # GPU
    gpar_functions=list(),
    n_gpu=None,
    gpu_comm=None,
    master_rank=None,
    )


###############################################################################
#
#                           Building Functions.
#
###############################################################################


def input(name, max_shape, typecode='f'):
    """
    Constructs a gpar input object, used to set up multiprocessing shared
    memory.  (User calls this multiple times to register all inputs before
    fork).

    name -- must match the name of a to-be-created theano variable to be used
    """
    if g.forked:
        raise RuntimeError("Cannot register new inputs after forking.")
    for h_inpt in g.inputs:
        if h_inpt.name == name:
            raise ValueError("Already have an input named ", name)
    assert isinstance(max_shape, (tuple, list))
    mp_array = np.ctypeslib.as_array(
        mp.RawArray(typecode, int(np.prod(max_shape)))).reshape(*max_shape)
    new_input = Input(name, mp_array, max_shape, typecode)
    g.inputs.append(new_input)
    print("Input ", name, " successfully registered.")
    return new_input.mp_array  # numpy wrapper around multiprocessing shared array


def function(inputs, outputs=None, updates=None, name=None):
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
        raise RuntimeError("Probably should fork before calling function, which uses theano.")
    if g.distributed:
        raise RuntimeError("Cannot make new functions after distributing.")

    # Check that all inputs have been registered and associate corresponding
    # multiprocessing shared memory with this function.
    # mp_inputs, worker_indeces = handling.inputs_handling(inputs)

    fcn_input_codes = handling.register_inputs(inputs, g.inputs, g.named_inputs)

    # Keep the outputs on the GPU; remember which to eventually send to CPU.
    gpu_outputs, outputs_to_cpu = handling.gpu_outputs(outputs)

    # TODO: Probably still need to do something about updates and givens.
    theano_function = theano.function(inputs=inputs,
                                      outputs=gpu_outputs,
                                      updates=updates,
                                      name=name,
                                      )

    g.theano_functions.append(theano_function)

    # Check for any shared values used in this function and keep track of them.
    shared_codes = handling.register_shareds(theano_function, g.shareds, g.named_shareds)

    gpar_function = Function(name=name,
                             theano_function=theano_function,
                             input_codes=fcn_input_codes,
                             outputs_to_cpu=outputs_to_cpu,
                             shared_codes=shared_codes,
                             mp_indeces=worker_indeces[0],
                             code=len(g.gpar_functions),  # Fcn can ID itself.
                             gpu_comm=g.gpu_comm,
                             sync=g.sync,
                             )
    g.gpar_functions.append(gpar_function)
    return gpar_function


###############################################################################
#
#                      GPU Collectives.
#
###############################################################################


def gpar_fcn_typecheck(gpar_function):
    if not isinstance(gpar_function, Function):
        raise TypeError("Expected gpar function(s).")


def shared_name_check(name):
    if name not in g.named_shareds:
        raise ValueError("Unrecognized name for shared variable: ", name)


def op_check(op):
    if op not in OPS:
        raise ValueError("Unrecognized reduction operator: ", op,
            ", must be one of: ", [k for k in OPS.keys()])
    elif op in ["avg", "average"]:
        raise NotImplementedError
    return op, OPS[op]


def get_shared_codes(gpar_functions=None, shared_names=None):
    if gpar_functions is None and shared_names is None:
        return list(range(len(g.shareds)))
    else:
        if isinstance(gpar_functions, (list, tuple)):
            for gpar_fcn in gpar_functions:
                gpar_fcn_typecheck(gpar_fcn)
        else:
            gpar_fcn_typecheck(gpar_fcn)
        if isinstance(shared_names, (list, tuple)):
            for name in shared_names:
                shared_name_check(name)
        else:
            shared_name_check(name)
        shared_codes = list()
        if isinstance(gpar_functions, (list, tuple)):
            for gpar_fcn in gpar_functions:
                shared_codes += gpar_fcn.shared_codes
        else:
            shared_codes += gpar_fcn.shared_codes
        if isinstance(shared_names, (list, tuple)):
            for name in shared_names:
                shared_codes.append(g.named_shareds[name])
        else:
            shared_codes.append(g.named_shareds[name])
    return tuple(sorted(set(shared_codes)))


def gpu_comm_function(gpu_comm_func, comm_code, has_op=False):
    def build_comm_procedure(f):
        @functools.wraps(f)
        def gpu_comm_procedure(*args, functions=None, shared_names=None, **kwargs):
            if g.closed:
                raise RuntimeError("Gpar already closed--cannot call comm function.")
            g.sync.exec_type = GPU_COMM
            g.sync.comm_code = comm_code
            shared_codes = get_shared_codes(functions, shared_names)
            n_shared = len(shared_codes)
            g.sync.shared_codes[:n_shared] = shared_codes
            g.sync.n_shared.value = n_shared
            if has_op:
                op, op_code = op_check(kwargs.pop("op", "avg"))
                g.sync.comm_op.value = op_code
                args = (*args, op)
            g.sync.barriers.exec_in.wait()
            results = list()
            for idx in shared_codes:
                results.append(gpu_comm_func(g.shareds[idx].data, *args, **kwargs))
            g.sync.barriers.exec_out.wait()
            return results  # (mostly just in case of non-in-place operation)
        return gpu_comm_procedure
    return build_comm_procedure


def reduce_func(src, op, in_place=True, dest=None):
    if in_place:
        g.gpu_comm.reduce(src=src, op=op, dest=src)
        return src
    else:
        return g.gpu_comm.reduce(src=src, op=op, dest=dest)  # makes a new gpuarray


def all_reduce_func(src, op):
    g.gpu_comm.all_reduce(src=src, op=op, dest=src)


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
    3. Turn off the gpar functions.
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
        for gpar_fcn in g.gpar_functions:
            gpar_fcn._close()  # Turn off all the functions.
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

    # NOTE: probably get rid of this.
    for inpt in g.inputs:
        inpt.assign_indeces(n_gpu)

    for rank in (r for r in range(n_gpu) if r != master_rank):
        args = tuple(rank, n_gpu, master_rank, sync, g.inputs)
        g.processes.append(mp.Process(target=worker_exec, args=args))
    for p in g.processes:
        p.start()

    import atexit
    atexit.register(close)
    
    Function._sync = sync  # endow all functions
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

    # Make a shared array of ints, size equal to number of shared variables.
    sync.shared_codes = ShmemRawArray('i', len(g.shareds), SH_ARRAY_TAG)
    gpu_comm, comm_id = util.init_gpu_comm(g.n_gpu, g.master_rank)
    g.sync.dict["comm_id"] = comm_id
    g.sync.distributed.value = True  # let the workers know this part succeeded
    g.sync.barriers.distribute.wait()  # signal workers to receive & join comm
    Function._gpu_comm = gpu_comm  # endow all functions
    for gpar_fcn in g.gpar_functions:
        gpar_fcn._set_normal_call()
    g.gpu_comm = gpu_comm
    g.distributed = True
