"""
Constansts, classes, and functions used in both master and workers.
"""

import ctypes
import os
import numpy as np

from shmemarray import ShmemRawArray


###############################################################################
#                                                                             #
#                           Constants                                         #
#                                                                             #
###############################################################################

PID = str(os.getpid())

# Exec types
FUNCTION = 0
GPU_COMM = 1

# GPU_COMM IDs
BROADCAST = 0
REDUCE = 1
ALL_REDUCE = 2
ALL_GATHER = 3

# Where to put functions on their way to workers
# (possibly need to make this secure somehow?)
PKL_FILE = "synk_function_dump_" + PID + ".pkl"

PRE = "/synk_" + PID
SH_ARRAY_TAG = PRE + "_active_theano_shareds"  # (shouldn't be a conflict!)
INPUT_TAGS_TAG = PRE + "_input_tag_IDs"
ASGN_IDX_TAG = PRE + "_assign_idx_"
SHAPES_TAG = PRE + "_shapes_"
SHMEM_TAG_PRE = PRE + "_"


REDUCE_OPS = {"+": 0,
              "sum": 0,
              "add": 0,
              "*": 1,
              "prod": 1,
              "product": 1,
              "max": 2,
              "maximum": 2,
              "min": 3,
              "minimum": 3,
              "avg": 4,
              "average": 4,
              "mean": 4,
              }

AVG_ALIASES = ["avg", "average", "mean"]

WORKER_OPS = {0: "sum",
              1: "prod",
              2: "max",
              3: "min",
              4: "avg",
              }

NP_TO_C_TYPE = {'float64': ctypes.c_double,
                'float32': ctypes.c_float,
                'float16': None,
                'int8': ctypes.c_byte,
                'int16': ctypes.c_short,
                'int32': ctypes.c_int,
                'int64': ctypes.c_longlong,
                'uint8': ctypes.c_ubyte,
                'uint16': ctypes.c_ushort,
                'uint32': ctypes.c_uint,
                'uint64': ctypes.c_ulonglong,
                'bool': ctypes.c_bool,
                }

COLLECT_MODES = ["reduce", "gather"]


###############################################################################
#                                                                             #
#                                Classes                                      #
#                                                                             #
###############################################################################

class struct(dict):

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


class Inputs(struct):

    def __init__(self, **kwargs):
        super(Inputs).__init__(self, **kwargs)
        self.shmems = list()  # numpy arrays wrapping shared memory
        self.names = list()  # strings (None if no name given)
        self.dtypes = list()  # numpy / theano type
        self.ctypes = list()  # ctypes needed for making shared array
        self.tags = list()  # current tag used for shared memory array
        self.ndims = list()
        self.num = 0

    def append(self, store):
        input_ID = self.num
        self.names.append(store.name)
        if store.name is None:
            raise RuntimeWarning("Synkhronos encountered un-named input; \
                shared memory management is improved if inputs used in \
                multiple functions are named.")
        self.dtypes.append(store.type.dtype)
        c_type = NP_TO_C_TYPE.get(store.type.dtype, None)
        if c_type is None:
            raise TypeError("Numpy/Theano type: ", store.type.dtype,
                " not supported.")
        self.ctypes.append(c_type)
        self.tags.append(ID)
        self.ndims.append(store.type.ndim)
        self.shmems.append(None)
        self.num += 1
        return input_ID


class Shareds(struct):

    def __init__(self, **kwargs):
        super(Shareds).__init__(self, **kwargs)
        self.vars = list()
        self.gpuarrays = list()
        self.names = list()
        self.avg_functions = list()
        self.num = 0

    def append(self, shared_var):
        shared_ID = self.num
        self.vars.append(shared_var)
        self.gpuarrays.append(shared_var.container.data)
        self.names.append(shared_var.name)
        self.num += 1
        return shared_ID


class SynkFunction(object):

    def __init__(self,
                 ID,
                 theano_function,
                 input_IDs,
                 shared_IDs,
                 collect_modes,
                 name=None,
                 reduce_ops=None,
                 avg_fac=None,
                 ):
        self._ID = ID
        self._theano_function = theano_function
        self._input_IDs = input_IDs
        self._shared_IDs = shared_IDs
        self._collect_modes = collect_modes
        self._name = name
        self._reduce_ops = reduce_ops
        self._avg_fac = avg_fac

    @property
    def name(self):
        return self._name

    @property
    def theano_function(self):
        return self._theano_function

    @property
    def collect_modes(self):
        return self._collect_modes

    @property
    def reduce_ops(self):
        return self._reduce_ops

    def _call_theano_function(self, inputs, output_subset=None):
        results = self._theano_function(*inputs, output_subset=output_subset)
        if not isinstance(results, list):
            results = [results, ]
        return results  # (always returns a list, even if length 1)

    # def _reduce_results(self, *args, **kwargs):
    #     """ Different for master vs worker """
    #     raise NotImplementedError

    # def _gather_results(self, *args, **kwargs):
    #     """ Different for master vs worker """
    #     raise NotImplementedError


###############################################################################
#                                                                             #
#                               Functions                                     #
#                                                                             #
###############################################################################

def use_gpu(rank):
    dev_str = "cuda" + str(rank)
    import theano.gpuarray
    theano.gpuarray.use(dev_str)


def init_gpu_comm(n_gpu, rank, comm_id=None):
    import theano.gpuarray
    from pygpu import collectives as gpu_coll

    gpu_ctx = theano.gpuarray.get_context()
    clique_id = gpu_coll.GpuCommCliqueId(gpu_ctx)
    if comm_id is not None:
        clique_id.comm_id = comm_id
    gpu_comm = gpu_coll.GpuComm(clique_id, n_gpu, rank)

    if comm_id is None:
        return gpu_comm, clique_id.comm_id  # (in master)
    else:
        return gpu_comm  # (in worker)


def register_inputs(theano_function, inputs_global, shareds_global):
    input_IDs = list()
    input_names = list()
    shared_IDs = list()
    shareds = theano_function.get_shareds()
    for shared in shareds:
        if shared in shareds_global.vars:
            shared_IDs.append(shareds_global.vars.index(shared))
        else:
            shared_ID = shareds_global.append(shared)
            shared_IDs.append(shared_ID)
    for store in theano_function.input_storage:
        if not store.implicit:  # (so an explicit input)
            input_names.append(store.name)
            if store.name is None or store.name not in inputs_global.names:
                inpt_ID = inputs_global.append(store)
                input_IDs.append(inpt_ID)
            else:
                input_IDs.append(inputs_global.names.index(store.name))
    return tuple(input_IDs), tuple(shared_IDs), tuple(input_names)


def build_vars_sync(inputs_global, shareds_global, n_func, n_gpu, create=True):
    assign_idx = [ShmemRawArray('i', n_gpu + 1, ASGN_IDX_TAG + str(idx), create)
                    for idx in range(n_func)]
    shapes = [ShmemRawArray('i', ndim, SHAPES_TAG + str(idx), create)
                for idx, ndim in enumerate(inputs_global.ndims)]
    vars_sync = struct(
        input_tags=ShmemRawArray('i', inputs_global.num, INPUT_TAGS_TAG, create),
        shared_IDs=ShmemRawArray('i', shareds_global.num, SH_ARRAY_TAG, create),
        assign_idx=assign_idx,
        shapes=shapes,
    )
    return vars_sync


def allocate_shmem(input_ID, inputs_global, shape, tag_ID, create=True):
    shmem = np.ctypeslib.as_array(
        ShmemRawArray(
            inputs_global.ctypes[input_ID],
            int(np.prod(shape)),
            SHMEM_TAG_PRE + str(tag_ID),
            create,
        )
    ).reshape(shape)
    inputs_global.shmems[input_ID] = shmem
    inputs_global.tags[input_ID] = tag_ID
    return shmem
