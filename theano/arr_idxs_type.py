from __future__ import absolute_import, print_function, division
#
# Slice type and Op. None Type and NoneConst.
#

import numpy as np

import theano
from theano.gof import Apply, Constant, Generic, Op, Type, hashtype
from theano.gradient import DisconnectedType


class MakeArrIdxs(Op):

    __props__ = ()

    def make_node(self, arr_idxs, stop=None, step=None):
        # Uses tuple type to tell whether to make slice
        if arr_idxs is None or isinstance(arr_idxs, int):
            slc = [arr_idxs, stop, step]
            inp = tuple(map(as_int_none_variable, slc))
        else:
            assert stop is None
            assert step is None
            if isinstance(arr_idxs, slice):
                slc = [arr_idxs.start, arr_idxs.stop, arr_idxs.step]
                inp = tuple(map(as_int_none_variable, slc))
            elif isinstance(arr_idxs, list):
                assert all(isinstance(i, int) for i in arr_idxs)
                inp = arr_idxs
            elif isinstance(arr_idxs, tuple):
                assert all(isinstance(i, int) for i in arr_idxs)
                inp = list(arr_idxs)
            elif isinstance(arr_idxs, np.ndarray):
                assert arr_idxs.ndim == 1
                assert np.issubdtype(arr_idxs[0], int)
                inp = arr_idxs
        return Apply(self, inp, [arr_idxs_type()])
         
    def perform(self, node, inp, out_):
        # Uses tuple type to know whether to make slice
        out, = out_
        if isinstance(inp, tuple):
            out[0] = slice(*inp)
        else:
            out[0] = inp

    def grad(self, inputs, grads):
        return [DisconnectedType()() for i in inputs]

make_arr_idxs = MakeArrIdxs()


class ArrIdxsType(Type):

    def filter(self, x, strict=False, allow_downcast=None):
        if isinstance(x, slice):
            return x
        if isinstance(x, (list, tuple)):
            if not all(isinstance(i, int) for i in x):
                raise TypeError("Elements of list must be type int.")
            return x
        if isinstance(x, np.ndarray):
            if not x.ndim == 1:
                raise TypeError("Numpy array must be one-dimensional.")
            if not np.issubdtype(x[0], int):
                raise TypeError("Numpy array must be integer dtype.")
            return x
        else:
            raise TypeError("Expected a slice, or a list, tuple, or 1-D numpy array of ints.")

    def __str__(self):
        return "arr_idxs_type"

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hashtype(self)

    @staticmethod
    def may_share_memory(a, b):
        if not isinstance(a, np.ndarray) and not isinstance(b, np.ndarray):
            # slices and lists/tuples of integers never share memory between objects
            return a is b
        else:
            return np.may_share_memory(a, b)

arr_idxs_type = ArrIdxsType()

# Need definition of constant type?


#
# vvvvv    original from theano.tensor.type_other.py    vvvvvv
#

def as_int_none_variable(x):
    if x is None:
        return NoneConst
    elif NoneConst.equals(x):
        return x
    x = theano.tensor.as_tensor_variable(x, ndim=0)
    if x.type.dtype not in theano.tensor.integer_dtypes:
        raise TypeError('index must be integers')
    return x


class MakeSlice(Op):

    __props__ = ()

    def make_node(self, slc, stop=None, step=None):
        # We need to accept and handle in make_node inputs the node
        # inputs to allow redoing a new op elsewhere in the graph by
        # optimization.
        if isinstance(slc, slice):
            assert stop is None
            assert step is None
            inp = [slc.start, slc.stop, slc.step]
        else:
            inp = [slc, stop, step]
        return Apply(self,
                     list(map(as_int_none_variable, inp)),
                     [slicetype()])

    def perform(self, node, inp, out_):
        out, = out_
        out[0] = slice(*inp)

    def grad(self, inputs, grads):
        return [DisconnectedType()() for i in inputs]

make_slice = MakeSlice()


class SliceType(Type):

    def filter(self, x, strict=False, allow_downcast=None):
        if isinstance(x, slice):
            return x
        else:
            raise TypeError('Expected a slice!')

    def __str__(self):
        return "slice"

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hashtype(self)

    @staticmethod
    def may_share_memory(a, b):
        # Slices never shared memory between object
        return isinstance(a, slice) and a is b

slicetype = SliceType()


class SliceConstant(Constant):
    def __init__(self, type, data, name=None):
        assert isinstance(data, slice)
        # Numpy ndarray aren't hashable, so get rid of them.
        if isinstance(data.start, np.ndarray):
            assert data.start.ndim == 0
            assert str(data.start.dtype) in theano.tensor.integer_dtypes
            data = slice(int(data.start), data.stop, data.step)
        elif isinstance(data.stop, np.ndarray):
            assert data.stop.ndim == 0
            assert str(data.stop.dtype) in theano.tensor.integer_dtypes
            data = slice(data.start, int(data.stop), data.step)
        elif isinstance(data.step, np.ndarray):
            assert data.step.ndim == 0
            assert str(data.step.dtype) in theano.tensor.integer_dtypes
            data = slice(data.start, int(data.stop), data.step)
        Constant.__init__(self, type, data, name)

    def signature(self):
        return (SliceConstant, self.data.start, self.data.stop, self.data.step)

    def __str__(self):
        return "%s{%s, %s, %s}" % (self.__class__.__name__,
                                   self.data.start,
                                   self.data.stop,
                                   self.data.step)
SliceType.Constant = SliceConstant


class NoneTypeT(Generic):
    """
    Inherit from Generic to have c code working.
    """

    def filter(self, x, strict=False, allow_downcast=None):
        if x is None:
            return x
        else:
            raise TypeError('Expected None!')

    @staticmethod
    def may_share_memory(a, b):
        # None never share memory between object, in the sence of DebugMode.
        # Python None are singleton
        return False

none_type_t = NoneTypeT()

# This is a variable instance. It can be used only once per fgraph.
# So use NoneConst.clone() before using it in a Theano graph.
# Use NoneConst.equals(x) to check if two variable are NoneConst.
NoneConst = Constant(none_type_t, None, name='NoneConst')
