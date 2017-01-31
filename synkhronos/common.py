"""
Constansts, classes, and functions used across master and worker.
"""

import os


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
#                                Classes                                      #
#                                                                             #
###############################################################################


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
