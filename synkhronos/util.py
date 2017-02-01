"""
Classes and functions used by master but which don't MODIFY globals
(Might still read from globals passed explicitly as parameter).
"""

import numpy as np
import multiprocessing as mp

from common import REDUCE_OPS, AVG_ALIASES
from common import SynkFunction


COLLECT_MODES = ["reduce", "gather"]


class struct(dict):

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def build_sync(n_gpu):
    from ctypes import c_bool

    mgr = mp.Manager()
    dictionary = mgr.dict()
    barriers = struct(
        distribute=mp.Barrier(n_gpu),
        exec_in=mp.Barrier(n_gpu),
        exec_out=mp.Barrier(n_gpu),
    )
    sync = struct(
        dict=dictionary,  # use for setup e.g. Clique comm_id; serializes.
        quit=mp.RawValue(c_bool, False),
        n_user_fcns=mp.RawValue('i', 0),
        distributed=mp.RawValue(c_bool, False),
        exec_type=mp.RawValue('i', 0),
        func_ID=mp.RawValue('i', 0),
        comm_ID=mp.RawValue('i', 0),
        n_shared=mp.RawValue('i', 0),
        barriers=barriers,
    )
    return sync


###############################################################################
#                           Counting GPUs                                     #


def n_gpu_getter(mp_n_gpu):
    """
    Call in a subprocess because it prevents future subprocesses from using GPU.
    """
    from pygpu import gpuarray
    mp_n_gpu.value = gpuarray.count_devices("cuda", 0)


def get_n_gpu(n_gpu, master_rank):
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
            raise RuntimeError("Only one GPU detected; just use Theano.)")
        else:
            print("Detected and attempting to use {} GPUs.".format(n_gpu))

    if master_rank not in list(range(n_gpu)):
        raise ValueError("Invalid value for master rank: ", master_rank)

    return n_gpu, master_rank


###############################################################################
#                           Building functions                                #


def check_collect(outputs, collect_modes, reduce_ops):
    if outputs is None:
        if collect_modes is not None or reduce_ops is not None:
            raise RuntimeWarning("No function outputs, ignoring collect_modes \
                and reduce_ops parameters.")
        return [], []
    n_outputs = len(outputs)
    if not isinstance(collect_modes, (list, tuple)):
        collect_modes = [collect_modes] * n_outputs
    if len(collect_modes) != n_outputs:
        raise ValueError("Number of collect modes does not match number of \
            outputs (or enter a single string to be used for all outputs).")
    for mode in collect_modes:
        if mode not in COLLECT_MODES:
            raise ValueError("Unrecognized collect_mode: ", mode)
    if not isinstance(reduce_ops, (list, tuple)):
        tmp_ops = list()
        for mode in collect_modes:
            if mode == "reduce":
                tmp_ops.append(reduce_ops)
            else:
                tmp_ops.append(None)
        reduce_ops = tmp_ops
    if "reduce" not in collect_modes and \
            any([op is not None for op in reduce_ops]):
        raise RuntimeWarning("Reduce op(s) provided but ignored--no reduced \
            outputs.")
    if len(reduce_ops) != n_outputs:
        raise ValueError("Number of reduce ops does not match number of \
            outputs (use None for non-reduce outputs, or a single string for \
            all reduced outputs).")
    else:
        for idx, op in enumerate(reduce_ops):
            if collect_modes[idx] == "reduce":
                if op not in REDUCE_OPS:
                    raise ValueError("Unrecognized reduce op: ", op)
            else:
                if op is not None:
                    raise RuntimeWarning("Reduce op provided but ignored for \
                        non-reduce collect mode.")
    return collect_modes, reduce_ops


def check_op(op):
    if op not in REDUCE_OPS:
        raise ValueError("Unrecognized reduction operator: ", op,
            ", must be one of: ", [k for k in REDUCE_OPS.keys()])
    return REDUCE_OPS[op]


def get_worker_reduce_ops(synk_functions):
    reduce_ops_all = [fn.reduce_ops for fn in synk_functions]
    for ops in reduce_ops_all:
        for idx, op in enumerate(ops):
            if op in AVG_ALIASES:
                ops[idx] = "sum"
    return reduce_ops_all


def check_func_scatter(inputs, broadcast_inputs, scatter_inputs):
    if broadcast_inputs is not None and scatter_inputs is not None:
        raise ValueError("May specify either broadcast_inputs or scatter_inputs but not both.")
    if broadcast_inputs is None and scatter_inputs is None:
        inputs_scatter = [True] * len(inputs)  # (default is to scatter all)
    elif broadcast_inputs is not None:
        if not isinstance(broadcast_inputs, (tuple, list)):
            raise TypeError("Optional param broadcast_inputs must be list or tuple.")
        inputs_scatter = [True] * len(inputs)
        for bc_inpt in broadcast_inputs:
            if bc_inpt not in inputs:
                raise ValueError("Elements of param broadcast_inputs must also be inputs.")
            inputs_scatter[inputs.index(bc_inpt)] = False
    else:  # (scatter_inputs is not None)
        if not isinstance(scatter_inputs, (list, tuple)):
            raise TypeError("Optional param scatter_inputs must be list or tuple.")
        inputs_scatter = [False] * len(inputs)
        for sc_inpt in scatter_inputs:
            if sc_inpt not in inputs:
                raise ValueError("Elements of param scatter_inputs must also be inputs.")
            inputs_scatter[inputs.index(sc_inpt)] = True
    return inputs_scatter


###############################################################################
#                           GPU Collectives                                   #


def get_shared_IDs(g_shareds, synk_functions=None, shared_vars=None):
    """ this one is clean from global accesses """
    if synk_functions is None and shared_vars is None:
        return tuple(range(g_shareds.num))  # default is all shareds
    else:
        # Type and existence checking.
        if not isinstance(synk_functions, (list, tuple)):
            synk_functions = (synk_functions,)
        for synk_fcn in synk_functions:
            if not isinstance(synk_fcn, SynkFunction):
                raise TypeError("Expected Synkhronos function(s).")
        if not isinstance(shared_vars, (list, tuple)):
            shared_vars = (shared_vars,)
        for var in shared_vars:
            if var is None:
                raise ValueError("Received None for one or mored shared variables")
            if var not in g_shareds.names and var not in g_shareds.vars:
                raise ValueError("Unrecognized shared variable or name: ", var)

        shared_IDs = list()
        for synk_fcn in synk_functions:
            shared_IDs += synk_fcn.shared_IDs
        for var in shared_vars:
            if var in g_shareds.names:
                shared_IDs.append(g_shareds.names.index(var))
            else:
                shared_IDs.append(g_shareds.vars.index(var))
    return tuple(sorted(set(shared_IDs)))


###############################################################################
#                       CPU Comm                                              #


def check_shared_var(g_shareds, shared_var):
    """ this one clean of global accesses """
    if shared_var not in g_shareds.vars and shared_var not in g_shareds.names:
        raise ValueError("Unrecog_ized theano shared variable or name: ",
            shared_var)
    if shared_var in g_shareds.vars:
        shared_ID = g_shareds.vars.index(shared_var)
    else:
        shared_ID = g_shareds.names.index(shared_var)
        shared_var = g_shareds.vars[shared_ID]
    return shared_var, shared_ID


def check_scatter_sources(g_shareds, n_gpu, sources, shared_ID):
    """ this one clean of global accesses """
    if not isinstance(sources, (tuple, list)):
        raise TypeError("Param sources must be a list or tuple of arguments to shared.set_value().")
    if len(sources) != n_gpu:
        raise ValueError("Source list must have as many elements as there are GPUs.")
    for src in sources:
        if not isinstance(src, np.ndarray):
            raise TypeError("For now...Must provide a numpy ndarray for each source.")
    shared_shape = g_shareds.gpuarrays[shared_ID].shape
    shared_dtype = g_shareds.vars[shared_ID].type.dtype
    for idx, src in enumerate(sources):
        if not isinstance(src, np.ndarray):
            src = np.asarray(src, dtype=shared_dtype)
            sources[idx] = src
        elif src.dtype != shared_dtype:
            raise TypeError("Must provide the same data type as the shared var: ",
                shared_dtype)
        if src.shape != shared_shape:
            raise ValueError("Source is not same shape as shared variable: ",
                shared_shape)
    return sources