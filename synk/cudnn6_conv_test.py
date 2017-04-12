
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as LN
import lasagne.init as LI


INPUT_SHAPE = (1, 24, 24)
input_var = T.tensor4('input_var', dtype='float32')


def wrapped_conv(*args, **kwargs):
    copy = dict(kwargs)
    copy.pop("image_shape", None)
    copy.pop("filter_shape", None)
    assert copy.pop("filter_flip", False)

    input, W, input_shape, get_W_shape = args
    if theano.config.device == 'cpu':
        return theano.tensor.nnet.conv2d(*args, **kwargs)
    try:
        # return theano.sandbox.cuda.dnn.dnn_conv(
        return theano.gpuarray.dnn.dnn_conv(  # FIXME
            input.astype('float32'),
            W.astype('float32'),
            **copy
        )
    except Exception as e:
        print("falling back to default conv2d")
        return theano.tensor.nnet.conv2d(*args, **kwargs)



l_in = L.InputLayer(shape=(None, *INPUT_SHAPE), input_var=input_var)

l_conv = L.Conv2DLayer(
    l_in,
    num_filters=10,
    filter_size=(4, 4),
    stride=(2, 2),
    pad=(0, 0),
    nonlinearity=LN.rectify,
    name="conv",
    convolution=wrapped_conv,
)

l_dense = L.DenseLayer(
    l_conv,
    num_units=32,
    nonlinearity=LN.rectify,
    name="hidden",
    W=LI.GlorotUniform(),
    b=LI.Constant(0.)
)

l_out = L.DenseLayer(
    l_dense,
    num_units=2,
    nonlinearity=LN.rectify,
    name="out",
    W=LI.GlorotUniform(),
    b=LI.Constant(0.),
)

output = L.get_output(l_out)

print("calling theano.function")
f = theano.function([input_var], output)

d = np.random.rand(5, *INPUT_SHAPE).astype('float32')

r = f(d)

print(r)
