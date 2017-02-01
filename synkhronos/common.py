"""
Constansts, classes, and functions used across master and worker.
"""

import os
import sys


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
GATHER = 4

# CPU_COMM IDs
SCATTER = 0

# Where to put functions on their way to workers
# (possibly need to make this secure somehow?)
PKL_FILE = "synk_function_dump_" + PID + ".pkl"

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

WORKER_OPS = {0: "sum",  # Make sure this is inverse of REDUCE_OPS
              1: "prod",
              2: "max",
              3: "min",
              4: "avg",
              }

AVG_ALIASES = ["avg", "average", "mean"]


###############################################################################
#                                                                             #
#                               Functions                                     #
#                                                                             #
###############################################################################


def use_gpu(rank, sync):
    dev_str = "cuda" + str(rank)
    err = None
    try:
        import theano.gpuarray
        theano.gpuarray.use(dev_str)
    except:
        err = sys.exc_info()[1]
        sync.gpu_inited.value = False  # (let others know it failed)
    sync.barriers.gpu_init.wait()
    if err is not None:
        if not isinstance(err, ImportError):
            raise err
    # TODO


    if not import_success or err is not None:
        sync.gpu_inited.value = False  # (let others know it failed)
    sync.barriers.gpu_init.wait()
    if err is not None:
        raise err
    return sync.gpu_inited.value


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
