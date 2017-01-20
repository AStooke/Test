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
import handling as h
from util import struct
from util import (FUNCTION, GPU_COMM, BROADCAST, REDUCE, ALL_REDUCE,
                       PKL_FILE, MASTER_RANK)


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
    inputs=list(),
    shareds=list(),
    named_shareds=dict(),
    theano_functions=list(),
    # GPU
    gpar_functions=list(),
    n_gpu=None,
    gpu_comm=None,
    master_rank=None,
    )


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
    mp_inputs, worker_indeces = h.inputs_handling(inputs)

    # Keep the outputs on the GPU; remember which to eventually send to CPU.
    outputs, outputs_to_cpu = h.outputs_handling(outputs)

    # TODO: Probably still need to do something about updates and givens.
    theano_function = theano.function(inputs=inputs,
                                      outputs=outputs,
                                      updates=updates,
                                      name=name)

    g.theano_functions.append(theano_function)

    # Check for any shared values used in this function and keep track of them.
    shared_codes = h.shareds_handling(theano_function)

    # Make each function aware of its unique code (idx in overall list).
    next_fcn_code = len(g.gpar_functions)
    gpar_function = Function(name=name,
                             theano_function=theano_function,
                             mp_inputs=mp_inputs,
                             outputs_to_cpu=outputs_to_cpu,
                             shared_codes=shared_codes,
                             mp_indeces=worker_indeces[0],
                             code=next_fcn_code,
                             gpu_comm=g.gpu_comm,
                             sync=g.sync,
                             )
    g.gpar_functions.append(gpar_function)
    return gpar_function

# TODO: figure out how to organize collectives in master.


def gpar_fcn_typecheck(gpar_function):
    if not isinstance(gpar_function, Function):
        raise TypeError("Expected gpar function(s).")


def shared_name_check(name):
    if name not in g.named_shareds:
        raise ValueError("Unrecognized name for shared variable: ", name)


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


def gpu_comm_procedure(comm_func, comm_code):
    def gpar_comm_fcn(functions=None, shared_names=None, **kwargs):
        if g.closed:
            raise RuntimeError("Cannot call GPU collectives after session closed.")
        g.sync.exec_type.value = GPU_COMM
        g.sync.exec_code.value = comm_code
        shared_codes = get_shared_codes(functions, shared_names)
        n_shareds = len(shared_codes)
        g.sync.shared_codes[:n_shareds] = shared_codes
        g.sync.n_shareds.value = n_shareds
        g.sync.barriers.exec_in.wait()
        for idx in shared_codes:
            comm_func(idx, **kwargs)
        g.sync.barriers.exec_out.wait()
    return gpar_comm_fcn


@gpu_comm_procedure(BROADCAST)
def broadcast(idx, root=None):
    g.gpu_comm.broadcast(src=g.shareds[idx].data, root=root)

@gpu_comm_procedure(REDUCE)
def reduce(idx, op=None, root=None):
    if op is None:
        raise ValueError("Must provide a reduction operation.")
    g.gpu_comm.reduce(src=g.shareds[idx].data, op=op, root=root)

# TODO: make sure this is coming out right....and then fill in with all the
# correct call signatures.



# def outer_wrapper(func, comm_code):
#     def outer_wrapped(*args, **kwargs):
#         if g.closed:
#             raise RuntimeError("Cannot call GPU collectives after session closed.")
#         g.sync.exec_type.value = GPU_COMM
#         g.sync.exec_code.value = comm_code
#         func(*args, **kwargs)
#         g.sync.barriers.exec_out.wait()
#     return outer_wrapped

# def shared_wrapper(func):
#     def shared_wrapped(functions=None, shared_names=None, **kwargs):
#         shared_codes = get_shared_codes(functions, shared_names)
#         n_shareds = len(shared_codes)
#         g.sync.shared_codes[:n_shareds] = shared_codes
#         g.sync.n_shareds.value = n_shareds
#         g.sync.barriers.exec_in.wait()
#         for idx in shared_codes:
#             func(idx, **kwargs)
#     return shared_wrapped

# @outer_wrapper(BROADCAST)
# @shared_wrapper
# def broadcast(idx, root=None):
#     gpu_comm.broadcast(g.shareds[idx].data, root)


def broadcast(functions=None, shared_names=None):
    if g.closed:
        raise RuntimeError("Cannot call GPU collectives after session closed.")
    g.sync.exec_type.value = GPU_COMM
    g.sync.exec_code.value = BROADCAST

    shared_codes = get_shared_codes(functions, shared_names)
    n_shareds = len(shared_codes)
    g.sync.shared_codes[:n_shareds] = shared_codes
    g.sync.n_shareds.value = n_shareds  # FIXME: set up shared list?
    g.sync.barriers.exec_in.wait()
    for idx in shared_codes:
        # Here tell the worker what to do and then release it.
        g.gpu_comm.broadcast(g.shareds[idx].data)
    g.sync.barriers.exec_out.wait()


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

    Function._sync = sync  # endow all functions
    g.forked = True
    g.n_gpu = n_gpu
    g.master_rank = master_rank
    g.sync = sync

    # Initialize disinct GPU.
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

    gpu_comm, comm_id = util.init_gpu_comm(g.n_gpu, g.master_rank)
    g.sync.dict["comm_id"] = comm_id
    g.sync.barriers.distribute.wait()  # signal workers to receive & join comm
    Function._gpu_comm = gpu_comm  # endow all functions
    for gpar_fcn in g.gpar_functions:
        gpar_fcn._set_normal_call()
    g.gpu_comm = gpu_comm
    g.distributed = True


def close():
    """
    TODO: make this happen automatically on exit (but still allow user to call it)
    Call this in the master at the end of the program.

    It needs to:
    1. Signal all the workers to quit.
    2. join() the subprocesses.
    3. Turn off the gpar functions.
    """
    if (not g.forked) or (not g.distributed):
        raise RuntimeError("Cannot close before forking and distributing.")
    if not g.closed:
        g.sync.quit.value = True
        g.sync.barriers.exec_in.wait()
        for p in g.processes:
            p.join()
        for gpar_fcn in g.gpar_functions:
            gpar_fcn._close()  # Turn off all the functions.
        g.closed = True
