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
              }

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
        ID = self.num
        self.names.append(store.name)
        if store.name is None:
            raise RuntimeWarning("Synkhronos encountered un-named input; shared memory management is improved if inputs used in multiple functions are named.")
        self.dtypes.append(store.type.dtype)
        c_type = NP_TO_C_TYPE.get(store.type.dtype, None)
        if c_type is None:
            raise TypeError("Numpy/Theano type: ", store.type.dtype, " not supported.")
        self.ctypes.append(c_type)
        self.tags.append(ID)
        self.ndims.append(store.type.ndim)
        self.shmems.append(None)
        self.num += 1
        return ID


class Shareds(struct):

    def __init__(self, **kwargs):
        super(Shareds).__init__(self, **kwargs)
        self.gpuarrays = list()
        self.names = list()
        self.num = 0

    def append(self, store):
        ID = self.num
        self.names.append(store.name)
        assert store.data is not None
        self.gpuarrays.append(store.data)
        self.num += 1
        return ID


class SynkFunction(object):

    def __init__(self,
                 ID,
                 theano_function,
                 input_IDs,
                 shared_IDs,
                 collect_mode,
                 reduce_op=None,
                 name=None,
                 ):
        self._ID = ID
        self._theano_function = theano_function
        self._input_IDs = input_IDs
        self._shared_IDs = shared_IDs
        self._name = name
        if collect_mode == "reduce":
            self._collect_results = self._reduce_results
            self._reduce_op = reduce_op
        elif collect_mode == "gather":
            self._collect_results = self._gather_results
            self._reduce_op = None
        else:
            raise RuntimeError("Unrecognized collect mode in function: ",
                collect_mode)
        self._collect_mode = collect_mode

    @property
    def name(self):
        return self._name

    @property
    def theano_function(self):
        return self._theano_function

    @property
    def collect_mode(self):
        return self._collect_mode

    @property
    def reduce_op(self):
        return self._reduce_op

    def _call_theano_function(self, inputs, output_subset=None):
        results = self._theano_function(*inputs, output_subset=output_subset)
        if not isinstance(results, list):
            results = [results, ]
        return results  # (always returns a list, even if length 1)

    def _reduce_results(self, *args, **kwargs):
        """ Different for master vs worker """
        raise NotImplementedError

    def _gather_results(self, *args, **kwargs):
        """ Different for master vs worker """
        raise NotImplementedError


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
    for store in theano_function.input_storage:
        if store.implicit:  # (a shared variable)
            for idx, gpuarray in enumerate(shareds_global.gpuarrays):
                if store.data is gpuarray:  # (a previously registered shared)
                    shared_IDs.append(idx)
                    break
            else:  # (does not match any previously registered)
                sh_ID = shareds_global.append(store)
                shared_IDs.append(sh_ID)
        else:  # (an explicit input)
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
