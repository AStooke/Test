

class SynkFunction(object):

    def __init__(self,
                 code,
                 theano_function,
                 input_codes,
                 shared_codes,
                 collect_mode,
                 reduce_op=None,
                 name=None,
                 ):
        self._code = code
        self._theano_function = theano_function
        self._input_codes = input_codes
        self._shared_codes = shared_codes
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

    def _call_theano_function(self, inputs):
        if inputs:
            results = self._theano_function(*inputs)
        else:
            results = self._theano_function()
        if not isinstance(results, tuple):
            results = (results,)
        return results  # (always returns a packed tuple, even if length 1)

    def _reduce_results(self, *args, **kwargs):
        """ Different for master vs worker """
        raise NotImplementedError

    def _gather_results(self, *args, **kwargs):
        """ Different for master vs worker """
        raise NotImplementedError
