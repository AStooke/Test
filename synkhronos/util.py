
import multiprocessing as mp
import ctypes
import os

PID = str(os.getpid())

MASTER_RANK = 0  # default if none other selected

# Exec types
FUNCTION = 0
GPU_COMM = 1

# GPU_COMM codes
BROADCAST = 0
REDUCE = 1
ALL_REDUCE = 2
ALL_GATHER = 3

# Where to put functions on their way to workers
# (possibly need to make this secure somehow?)
PKL_FILE = "function_dump.pkl"

SH_ARRAY_TAG = "/synk_" + PID + "_active_theano_shareds"  # (shouldn't be a conflict!)
INPUT_TAG_CODES_TAG = "/synk_" + PID + "_input_tag_codes"
ASGN_IDX_TAG = "/synk_" + PID + "_assign_idx"
SHMEM_TAG_PRE = "/synk_" + PID + "_"


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


class struct(dict):

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


class Inputs(struct):

    def __init__(self, **kwargs):
        super(Inputs).__init__(self, **kwargs)
        self.shmems = list()  # numpy arrays wrapping shared memory
        self.names = list()  # strings (None if no name given)
        self.ctypes = list()  # ctypes needed for making shared array
        self.tags = list()  # current tag used for shared memory array
        self.num = 0

    def append(self, store):
        code = self.num
        self.names.append(store.name)
        if store.name is None:
            raise RuntimeWarning("Synkhronos encountered un-named input; shared memory management is improved if inputs used in multiple functions are named.")
        c_type = NP_TO_C_TYPE.get(store.type.dtype, None)
        if c_type is None:
            raise TypeError("Numpy/Theano type: ", store.type.dtype, " not supported.")
        self.ctypes.append(c_type)
        self.tags.append(code)
        self.shmems.append(None)
        self.num += 1
        return code


class Shareds(struct):

    def __init__(self, **kwargs):
        super(Shareds).__init__(self, **kwargs)
        self.gpuarrays = list()
        self.names = list()
        self.num = 0

    def append(self, store):
        code = self.num
        self.names.append(store.name)
        assert store.data is not None
        self.gpuarrays.append(store.data)
        self.num += 1
        return code


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


def n_gpu_getter(mp_n_gpu):
    """
    Call in a subprocess because it prevents future subprocesses from using GPU.
    """
    from pygpu import gpuarray
    mp_n_gpu.value = gpuarray.count_devices("cuda", 0)


def n_gpu(n_gpu, master_rank):
    if n_gpu is not None:
        n_gpu = int(n_gpu)
    if master_rank is None:
        master_rank = MASTER_RANK
    else:
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
        func_code=mp.RawValue('i', 0),
        comm_code=mp.RawValue('i', 0),
        n_shared=mp.RawValue('i', 0),
        barriers=barriers,
    )
    return sync


def use_gpu(rank):
    dev_str = "cuda" + str(rank)
    import theano.gpuarray
    theano.gpuarray.use(dev_str)


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
    return op, REDUCE_OPS[op]
