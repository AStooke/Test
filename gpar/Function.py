"""
Still a question....how to write these without refering to h; sync
...or maybe that's OK?
"""

import numpy as np

from constants import FUNCTION, MASTER_RANK


class BaseFunction(object):

    _gpu_comm = None
    _sync = None

    def __init__(self,
                 name,
                 theano_function,
                 mp_inputs,
                 outputs_to_cpu,
                 shared_codes,
                 mp_indeces,  # probably get rid of this..make it dynamic
                 ):
        self._name = name
        self._theano_function = theano_function
        self._shared_codes = shared_codes  # (tuple)
        self._mp_inputs = mp_inputs  # (tuple)
        self._outputs_to_cpu = outputs_to_cpu
        self._s_ind = mp_indeces[0]
        self._e_ind = mp_indeces[1]

    @property
    def name(self):
        return self._name

    @property
    def theano_function(self):
        return self._theano_function

    @property
    def shared_names(self):
        return self._shared_names

    @property
    def outputs_to_cpu(self):
        return self._outputs_to_cpu

    def _call_local_function(self, inputs=None):
        if inputs:
            return self._theano_function(*inputs)
        else:
            return self._theano_function()


class MasterFunction(BaseFunction):

    def __init__(self, code, *args, **kwargs):
        super(MasterFunction).__init__(*args, **kwargs)
        self._code = code
        self._call = self._pre_dict_call

    def __call__(self, *args, **kwargs):
        self._call(*args, **kwargs)

    def _set_normal_call(self):
        self._call = self._gpar_call

    def _close(self):
        self._call = self._closed_call

    def _share_inputs(self, args):
        # FIXME: this needs to get better, handle different sizes
        if args:
            assert isinstance(args, (tuple, list))
        my_inputs = list()
        for a, share in zip(args, self.mp_inputs):
            # ipdb.set_trace()
            if a is not share:
                share[:] = a  # later make this take up to size max, but also less
            my_inputs.append(share[self.s_ind:self.e_ind])  # a view
        return my_inputs

    def _set_worker_signal(self):
        self._sync.exec_type.value = FUNCTION
        self._sync.exec_code.value = self._code
        self._sync.barriers.exec_in.wait()

    def _collect_results(self, results):
        if isinstance(results, (list, tuple)):
            for r in results:
                self._gpu_comm.reduce(r, 'sum', r)
            for idx, r in enumerate(results):
                if self.outputs_to_cpu[idx]:
                    results[idx] = np.array(r)
        else:
            self._gpu_comm.reduce(results, 'sum', results)
            if self.outputs_to_cpu:
                results = np.array(results)
        self._sync.barriers.exec_out.wait()
        return results

    def _closed_call(self, *args):
        raise RuntimeError("GPar already closed, can only call Theano function.")

    def _pre_dist_call(self, *args):
        raise RuntimeError("Gpar functions have not been distributed to workers, can only call Theano function.")

    def _gpar_call(self, *args):
        """
        This needs to:
        1. Share input data.
        2. Signal to workers to start and what to do.
        3. Call the local theano function on data.
        4. Collect result from workers and return it.

        NOTE: Barriers happen INSIDE master function call.
        FIXME: handle kwargs?
        """
        my_args = self._share_inputs(args)
        self._set_worker_signal()
        results = self._call_local_function(my_args)
        return self._collect_results(results)


class WorkerFunction(BaseFunction):

    master_rank = None

    def __call__(self):
        """
        This needs to:
        1. Gather the right inputs from mp shared values.
        2. Execute local theano function on those inputs.
        3. Send results back to master.

        NOTE: Barriers happen OUTSIDE worker function call.
        """
        inputs = self._receive_inputs()
        results = self._call_local_function(inputs)
        self._send_results(results)

    def _receive_inputs(self):
        my_inputs = list()
        for inpt in self.mp_inputs:
            my_inputs.append(inpt[self.s_ind:self.e_ind])  # a view
        return my_inputs

    def _send_results(self, results):
        if isinstance(results, (list, tuple)):
            for r in results:
                self._gpu_comm.reduce(r, 'sum', root=master_rank)
        else:
            self._gpu_comm.reduce(r, 'sum', root=master_rank)
