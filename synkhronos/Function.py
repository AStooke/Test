"""
Still a question....how to write these without refering to h; sync
...or maybe that's OK?
"""

import numpy as np

from util import FUNCTION, MASTER_RANK, PID, NP_TO_C_TYPE, SHMEM_TAG_PRE
from shmemarray import ShmemRawArray


class SynkFunction(object):

    def __init__(self,
                 code,
                 theano_function,
                 input_codes,
                 shared_codes,
                 name=None,
                 ):
        self._code = code
        self._theano_function = theano_function
        self._input_codes = input_codes
        self._shared_codes = shared_codes  # (tuple)
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def theano_function(self):
        return self._theano_function

    def _call_theano_function(self, inputs=None):
        if inputs is not None:
            return self._theano_function(*inputs)
        else:
            return self._theano_function()


# class MasterFunction(SynkFunction):

#     _sync = None
#     _gpu_comm = None
#     _n_gpu = None

#     def __init__(self, *args, **kwargs):
#         super(MasterFunction).__init__(*args, **kwargs)
#         self._call = self._pre_distributed_call

#     def __call__(self, *args, **kwargs):
#         self._call(*args, **kwargs)  # What this refers to is set dynamically

#     def _set_normal_call(self):
#         self._call = self._synk_call

#     def _close(self):
#         self._call = self._closed_call

#     def _share_inputs(self, args):
#         if not args:
#             return
#         my_inputs = list()
#         assert isinstance(args, (tuple, list))
#         batch_size = args[0].shape[0]
#         for arg in args:
#             if arg.shape[0] != batch_size:
#                 raise ValueError("Inputs of different batch sizes (using 0-th index).")
#         if batch_size != self._previous_batch_size:
#             assign_idx = np.ceil(np.linspace(
#                 0, batch_size, self._n_gpu + 1)).astype(int)
#             self._sync.assign_idx[self._code, :] = assign_idx
#             self._my_idx = (assign_idx[self._rank], assign_idx[self._rank + 1])
#             self._previous_batch_size = batch_size
#         for idx, (arg, shmem) in enumerate(zip(args, self._input_shmems)):
#             if shmem is None:
#                 self._alloc_write_shmem(arg, idx)
#             else:
#                 # check if they are already the same memory (based on first element)
#                 arg_addr, _ = arg.__array_interface__["data"]
#                 shmem_addr, _ = shmem.__array_interface__["data"]
#                 # if they do start at the same memory, assume nothing to do.
#                 if arg_addr != shmem_addr:
#                     if arg.shape[1:] != shmem.shape[1:] or batch_size > shmem.shape[0]:
#                         # new shape or bigger batch
#                         self._alloc_write_shmem(arg, idx)
#                     else:
#                         shmem[:batch_size] = arg  # already enough shared memory
#             my_inputs.append(arg[self._my_idx[0]:self._my_idx[1]])
#         return my_inputs

#     def _alloc_write_shmem(self, arg, idx):
#         c_type = NP_TO_C_TYPE.get(arg.dtype.name, None)
#         if c_type is None:
#             raise TypeError("Numpy type: ", arg.dtype.name, " not supported.")
#         shape = list(arg.shape)
#         shape[0] = int(np.ceil(shape[0] * 1.05))  # (a little extra)
#         tag_code = np.max(self._sync.input_tag_codes) + 1
#         # FIXME: this is possibly a bad idea, might hit some max length for tag code?
#         self._sync.input_tag_codes[self._input_codes[idx]] = tag_code
#         shmem = np.ctypeslib.as_array(ShmemRawArray(
#             c_type, int(np.prod(shape), SHMEM_TAG_PRE + str(tag_code)))
#             ).reshape(shape)
#         shmem[:arg.shape[0]] = arg  # (copy arg data into shared memory buffer)
#         self._input_shmems[idx] = shmem

#     def _set_worker_signal(self):
#         self._sync.exec_type.value = FUNCTION
#         self._sync.func_code.value = self._code
#         self._sync.barriers.exec_in.wait()

#     def _collect_results(self, results):
#         if isinstance(results, (list, tuple)):
#             for r in results:
#                 self._gpu_comm.reduce(r, 'sum', r)
#             for idx, r in enumerate(results):
#                 if self.outputs_to_cpu[idx]:
#                     results[idx] = np.array(r)
#         else:
#             self._gpu_comm.reduce(results, 'sum', results)
#             if self.outputs_to_cpu:
#                 results = np.array(results)
#         self._sync.barriers.exec_out.wait()
#         return results

#     def _closed_call(self, *args):
#         raise RuntimeError("Synkhronos already closed, can only call Theano function.")

#     def _pre_distributed_call(self, *args):
#         raise RuntimeError("Synkhronos functions have not been distributed to workers, can only call Theano function.")

#     def _synk_call(self, *inputs):
#         """
#         This needs to:
#         1. Share input data.
#         2. Signal to workers to start and what to do.
#         3. Call the local theano function on data.
#         4. Collect result from workers and return it.

#         NOTE: Barriers happen INSIDE master function call.
#         FIXME: handle kwargs?
#         """
#         my_inputs = self._share_inputs(inputs)
#         self._set_worker_signal()
#         my_results = self._call_theano_function(my_inputs)
#         return self._collect_results(my_results)


# class WorkerFunction(SynkFunction):

#     sync = None
#     gpu_comm = None
#     master_rank = None

#     def __call__(self):
#         """
#         This needs to:
#         1. Gather the right inputs from mp shared values.
#         2. Execute local theano function on those inputs.
#         3. Send results back to master.

#         NOTE: Barriers happen OUTSIDE worker function call.
#         """
#         inputs = self._receive_inputs()
#         results = self._call_theano_function(inputs)
#         self._send_results(results)

#     def _receive_inputs(self):
#         my_inputs = list()
#         my_idx = (self.sync.assign_idx[self.rank],
#             self.sync.assign_idx[self.rank + 1])
#         for inpt in self.input_shmems




#         for inpt in self.mp_inputs:
#             my_inputs.append(inpt[self.s_ind:self.e_ind])  # a view
#         return my_inputs

#     def _send_results(self, results):
#         if isinstance(results, (list, tuple)):
#             for r in results:
#                 self._gpu_comm.reduce(r, 'sum', root=self.master_rank)
#         else:
#             self._gpu_comm.reduce(r, 'sum', root=self.master_rank)
