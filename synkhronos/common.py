"""
Constansts, classes, and functions used in both master and workers.
"""

import ctypes
import os
import numpy as np
import theano

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
CPU_COMM = 2

# GPU_COMM IDs
BROADCAST = 0
REDUCE = 1
ALL_REDUCE = 2
ALL_GATHER = 3

# CPU_COMM IDs
SCATTER = 0

# Where to put functions on their way to workers
# (possibly need to make this secure somehow?)
PKL_FILE = "synk_function_dump_" + PID + ".pkl"

PRE = "/synk_" + PID
SHRD_ARRAY_TAG = PRE + "_active_theano_shareds"  # (shouldn't be a conflict!)
INPUT_TAGS_TAG = PRE + "_input_tag_IDs"
ASGN_IDX_TAG = PRE + "_assign_idx_"
SHAPES_TAG = PRE + "_shapes_"
MAX_INPT_IDX_TAG = PRE + "_max_idx"
INPT_SHMEM_TAG_PRE = PRE + "_INPT_"
SHRD_SHMEM_TAG_PRE = PRE + "_SHRD_"

AVG_FAC_NAME = "__synk_avg_fac__"


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
        self.vars = list()
        self.shmems = list()  # numpy arrays wrapping shared memory
        self.names = list()  # strings (None if no name given)
        self.dtypes = list()  # numpy / theano type
        self.ctypes = list()  # ctypes needed for making shared array
        self.tags = list()  # current tag used for shared memory array
        self.ndims = list()
        self.num = 0
        self.sync = None

    def include(self, var):
        if var in self.vars:  # (already have var, just return its ID)
            return self.vars.index(var)
        else:  # ( don't have var yet, append it)
            input_ID = self.num
            self.names.append(var.name)  # (even if None)
            self.dtypes.append(var.type.dtype)
            ctype = NP_TO_C_TYPE.get(var.type.dtype, None)
            if ctype is None:
                raise TypeError("Numpy/Theano type: ", var.type.dtype,
                    " not supported.")
            self.ctypes.append(ctype)
            self.tags.append(input_ID)
            self.ndims.append(var.type.ndim)
            self.shmems.append(None)
            self.num += 1
            return input_ID

    def register_func(self, theano_function):
        inpt_IDs = list()
        inpt_names = list()
        for theano_In in theano_function.inv_finder.values():
            if not theano_In.implicit:  # (then it is explicit input)
                var = theano_In.variable
                inpt_names.append(var.name)
                inpt_IDs.append(self.include(var))
        return tuple(inpt_IDs), tuple(inpt_names)

    def alloc_shmem(self, input_ID, shape, tag_ID, create=True):
        shmem = np.ctypeslib.as_array(
            ShmemRawArray(
                self.ctypes[input_ID],
                int(np.prod(shape)),
                INPT_SHMEM_TAG_PRE + str(tag_ID),
                create,
            )
        ).reshape(shape)
        self.shmems[input_ID] = shmem
        self.tags[input_ID] = tag_ID  # (needs to be dynamic for changing size)
        return shmem

    def build_sync(self, n_func, n_gpu, create=True):
        if self.sync is not None:
            raise RuntimeError("Tried to build inputs sync a second time.")
        assign_idx = [ShmemRawArray('i', n_gpu + 1, ASGN_IDX_TAG + str(idx), create)
                        for idx in range(n_func)]
        shapes = [ShmemRawArray('i', ndim, SHAPES_TAG + str(idx), create)
                    for idx, ndim in enumerate(self.ndims)]
        max_idx = ShmemRawArray('i', self.num, MAX_INPT_IDX_TAG, create)
        inpt_sync = struct(
            input_tags=ShmemRawArray('i', self.num, INPUT_TAGS_TAG, create),
            assign_idx=assign_idx,
            shapes=shapes,
            max_idx=max_idx,
        )
        self.sync = inpt_sync
        return inpt_sync


class Shareds(struct):

    def __init__(self, **kwargs):
        super(Shareds).__init__(self, **kwargs)
        self.vars = list()
        self.gpuarrays = list()
        self.names = list()
        self.dtypes = list()
        self.ctypes = list()
        self.avg_functions = list()
        self.avg_facs = list()
        self.shmems = list()  # for cpu scatter, will be diff in master/worker
        self.shapes = list()
        self.num = 0
        self.sync = None

    def include(self, var):
        if var in self.vars:
            return self.vars.index(var)
        else:
            shared_ID = self.num
            self.vars.append(var)
            self.gpuarrays.append(var.container.data)
            self.names.append(var.name)
            self.dtypes.append(var.type.dtype)
            ctype = NP_TO_C_TYPE.get(var.type.dtype, None)
            if ctype is None:
                raise TypeError("Numpy/Theano type: ", var.type.dtype,
                    " not supported for CPU-based scatter of shared var.")
            self.ctypes.append(ctype)
            self.shmems.append(None)
            self.shapes.append(var.container.data.shape)

            # avg func built later?  yes, only in master
            self.num += 1
            return shared_ID

    def register_func(self, theano_function):
        shared_IDs = list()
        shareds = theano_function.get_shareds()
        for shared in shareds:
            shared_IDs.append(self.include(shared))
        return tuple(shared_IDs)

    def alloc_shmem(self, shared_ID, rank, create=True):
        shape = self.shapes[shared_ID]
        shmem = np.ctypeslib.as_array(
            ShmemRawArray(
                self.ctypes[shared_ID],
                int(np.prod(shape)),
                SHRD_SHMEM_TAG_PRE + str(shared_ID) + "_" + str(rank),
                create,
            )
        ).reshape(shape)
        if not create:
            self.shmems[shared_ID] = shmem  # (in workers only)
        return shmem

    def build_shmems(self, shared_ID, n_gpu, master_rank):
        shmems = list()
        for rank in range(n_gpu):
            if rank == master_rank:
                shmems.append(None)
            else:
                shmems.append(self.alloc_shmem(shared_ID, rank))
        self.shmems[shared_ID] = shmems
        return shmems

    def build_sync(self, create=True):
        if self.sync is not None:
            raise RuntimeError("Tried to build sync on shareds a second time.")
        shrd_sync = struct(
            shared_IDs=ShmemRawArray('i', self.num, SHRD_ARRAY_TAG, create),
        )
        self.sync = shrd_sync
        return shrd_sync

    def build_avg_functions(self):
        """
        Not done automatically in register because only the master calls this
        method.
        """
        avg_funcs = list()
        avg_facs = list()
        for var in self.vars:
            avg_fac = theano.shared(np.array(1, dtype=var.type.dtype),
                                    name=AVG_FAC_NAME)
            avg_funcs.append(
                theano.function([], updates={var: var * avg_fac}))
            avg_facs.append(avg_fac)
        self.avg_funcs = avg_funcs
        self.avg_facs = avg_facs
        return avg_funcs

    def set_avg_facs(self, n_gpu):
        for avg_fac in self.avg_facs:
            avg_fac.set_value(1 / n_gpu)


class SynkFunction(object):

    def __init__(self,
                 ID,
                 theano_function,
                 input_IDs,
                 inputs_scatter,
                 shared_IDs,
                 collect_modes,
                 name=None,
                 reduce_ops=None,
                 avg_fac=None,
                 ):
        self._ID = ID
        self._theano_function = theano_function
        self._input_IDs = input_IDs
        self._inputs_scatter = inputs_scatter
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

    @property
    def inputs_scatter(self):
        return self._inputs_scatter

    def _call_theano_function(self, inputs, output_subset=None):
        results = self._theano_function(*inputs, output_subset=output_subset)
        if not isinstance(results, list):
            results = [results, ]
        return results  # (always returns a list, even if length 1)


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
