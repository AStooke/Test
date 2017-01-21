
import multiprocessing as mp


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

SH_ARRAY_TAG = "active_shared_variable_shared_array_tag"  # (shouldn't be a conflict!)

OPS = {"+": 0,
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


class struct(dict):

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


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
    mp_n_gpu.value = gpuarray.count_devices('cuda', 0)


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
        if n_gpu == -1:
            raise ImportError("Unable to import pycuda to detect GPU count.")
        elif n_gpu == 0:
            raise RuntimeError("No GPU detected by pycuda.")
        elif n_gpu == 1:
            raise RuntimeWarning("Only one GPU detected; undetermined behavior (but I should make it revert to regular Theano)")
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



