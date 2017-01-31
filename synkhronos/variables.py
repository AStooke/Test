"""
Classes for managing variables.  Inputs and Shareds used in both master and
workers.  Outputs used only in master.
"""

import ctypes
import numpy as np
import theano

from util import struct
from common import PID
from shmemarray import ShmemRawArray

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

PRE = "/synk_" + PID
SHRD_ARRAY_TAG = PRE + "_active_theano_shareds"  # (shouldn't be a conflict!)
INPUT_TAGS_TAG = PRE + "_input_tag_IDs"
ASGN_IDX_TAG = PRE + "_assign_idx_"
SHAPES_TAG = PRE + "_shapes_"
MAX_INPT_IDX_TAG = PRE + "_max_idx"
INPT_SHMEM_TAG_PRE = PRE + "_INPT_"
SHRD_SHMEM_TAG_PRE = PRE + "_SHRD_"

AVG_FAC_NAME = "__synk_avg_fac__"


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
            self.vars.append(var)
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
        input_IDs = list()
        input_names = list()
        input_vars = list()
        for theano_In in theano_function.inv_finder.values():
            if not theano_In.implicit:  # (then it is explicit input)
                var = theano_In.variable
                input_vars.append(var)
                input_names.append(var.name)
                input_IDs.append(self.include(var))
        return tuple(input_IDs), tuple(input_names), tuple(input_vars)

    def alloc_shmem(self, input_ID, shape, tag_ID=None, create=True):
        _tag_ID = np.max(self.tags) + 1 if tag_ID is None else tag_ID
        shmem = np.ctypeslib.as_array(
            ShmemRawArray(
                self.ctypes[input_ID],
                int(np.prod(shape)),
                INPT_SHMEM_TAG_PRE + str(_tag_ID),
                create,
            )
        ).reshape(shape)
        self.shmems[input_ID] = shmem
        self.tags[input_ID] = tag_ID  # (needs to be dynamic for changing size)
        if tag_ID is None:
            return shmem, tag_ID  # (in master when making new tag)
        else:
            return shmem  # (in worker)

    def build_sync(self, n_func, n_gpu, create=True):
        if self.sync is not None:
            raise RuntimeError("Tried to build inputs sync a second time.")
        assign_idx = [ShmemRawArray('i', n_gpu + 1, ASGN_IDX_TAG + str(idx), create)
                        for idx in range(n_func)]
        shapes = [ShmemRawArray('i', ndim, SHAPES_TAG + str(idx), create)
                    for idx, ndim in enumerate(self.ndims)]
        max_idx = ShmemRawArray('i', self.num, MAX_INPT_IDX_TAG, create)
        inpt_sync = struct(
            tags=ShmemRawArray('i', self.num, INPUT_TAGS_TAG, create),
            assign_idx=assign_idx,
            shapes=shapes,
            max_idx=max_idx,
        )
        self.sync = inpt_sync
        return inpt_sync

    def update_shmem(self, input_ID, input_data):
        """ Master-only """
        shmem = self.shmems[input_ID]
        if not check_memory(shmem, input_data):
            shape = list(input_data.shape)
            shape[0] = int(np.ceil(shape[0] * 1.05))   # (a little extra)
            shmem, tag_ID = self.alloc_shmem(input_ID, shape)
            self.sync.tags[input_ID] = tag_ID
        shmem[:input_data.shape[0]] = input_data
        self.sync.max_idx[input_ID] = input_data.shape[0]  # (in case broadcast)
        return shmem

    def check_inputs(self, input_IDs, ordered_inputs):
        """ Master-only """
        for idx, (input_ID, input_data) in enumerate(zip(input_IDs, ordered_inputs)):
            if not isinstance(input_data, np.ndarray):
                input_data = np.asarray(input_data, dtype=self.dtypes[input_ID])
                ordered_inputs[idx] = input_data
            elif input_data.dtype != self.dtypes[input_ID]:
                raise TypeError("Wrong data type provided for input ", idx,
                    ": ", input_data.dtype)
            if input_data.ndim != self.ndims[input_ID]:
                raise TypeError("Wrong data ndim provided for input ", idx,
                    ": ", input_data.ndim)
        return ordered_inputs  # (now as numpy arrays)


def check_memory(shmem, input_data):
    memory_OK = False
    if shmem is not None:
        input_addr, _ = input_data.__array_interface__["data"]
        shmem_addr, _ = shmem.__array_interface__["data"]
        if input_addr == shmem_addr:
            if input_data.__array_interface__["strides"] is not None:
                raise Warning("Cannot use strided view of memory as input, \
                    will copy into new shmem array.")
            elif input_data.shape[1:] == shmem.shape[1:] and \
                    input_data.shape[0] <= shmem.shape[0]:
                memory_OK = True
    return memory_OK


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

    def include(self, var, build_avg_func):
        if var in self.vars:
            return self.vars.index(var)
        else:
            shared_ID = self.num
            self.vars.append(var)
            self.gpuarrays.append(var.container.data)
            self.names.append(var.name)
            dtype = var.type.dtype
            self.dtypes.append(dtype)
            ctype = NP_TO_C_TYPE.get(dtype, None)
            if ctype is None:
                raise TypeError("Numpy/Theano type: ", dtype,
                    " not supported for CPU-based scatter of shared var.")
            self.ctypes.append(ctype)
            self.shmems.append(None)
            self.shapes.append(var.container.data.shape)
            if build_avg_func:   # (only in master)
                avg_fac = theano.shared(np.array(1, dtype=dtype),
                                        name=AVG_FAC_NAME)
                avg_func = theano.function([], updates={var: var * avg_fac})
                self.avg_facs.append(avg_fac)
                self.avg_funcs.append(avg_func)
            self.num += 1
            return shared_ID

    def register_func(self, theano_function, build_avg_func=True):
        shared_IDs = list()
        shareds = theano_function.get_shareds()
        for shared in shareds:
            shared_IDs.append(self.include(shared, build_avg_func))
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

    def set_avg_facs(self, n_gpu):
        for avg_fac in self.avg_facs:
            avg_fac.set_value(1 / n_gpu)

    def unpack_avg_facs(self):
        """ Worker only (and only if later changing avg_fac dynamically) """
        for fcn in self.avg_functions:
            for fcn_shared in fcn.get_shared():
                if fcn_shared.name == AVG_FAC_NAME:
                    self.avg_facs.append(fcn_shared)
                    break
                else:
                    raise RuntimeError("Could not identify shared var's \
                        average factor.")


class Outputs(struct):
    """ Master only. """

    def __init__(self, **kwargs):
        super(Outputs).__init__(self, **kwargs)
        self.vars = list()
        self.gpu_vars = list()
        self.to_cpu = list()
        self.avg_funcs = list()
        self.avg_facs = list()

    def include(self, var):
        if var in self.vars:  # (already have this var, just retrieve it)
            output_ID = self.vars.index(var)
            gpu_var = self.gpu_vars[output_ID]
            avg_func = self.avg_funcs[output_ID]
            to_cpu = self.to_cpu[output_ID]
        else:
            from theano.gpuarray.type import GpuArrayVariable
            self.vars.append(var)
            to_cpu = False if isinstance(var, GpuArrayVariable) else True
            self.to_cpu.append(to_cpu)
            gpu_var = var.transfer(None)
            self.gpu_vars.append(gpu_var)
            avg_fac = theano.shared(np.array(1, dtype=var.type.dtype))
            avg_otpt = (avg_fac * gpu_var).transfer(None)
            avg_func = theano.function([gpu_var], avg_otpt)
            self.avg_funcs.append(avg_func)
        return gpu_var, avg_func, to_cpu

    def register(self, outputs):
        gpu_outputs = list()
        output_avg_funcs = list()
        outputs_to_cpu = list()
        for var in outputs:
            gpu_output, avg_func, to_cpu = self.include(var)
            gpu_outputs.append(gpu_output)
            output_avg_funcs.append(avg_func)
            outputs_to_cpu.append(to_cpu)
        return gpu_outputs, output_avg_funcs, outputs_to_cpu

    def set_avg_facs(self, n_gpu):
        for avg_fac in self.avg_facs:
            avg_fac.set_value(1 / n_gpu)
