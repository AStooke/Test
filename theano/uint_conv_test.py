
import theano
import theano.tensor as T
import numpy as np
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as LN
# import ipdb

x_uint = T.tensor4('x_uint', dtype='uint8')
x_float = x_uint * (1 / 255.)
# x_float = T.tensor4('x_float', dtype='float32')

x_shape = (1, 1, 8, 8)


l_in = L.InputLayer(shape=x_shape, input_var=x_uint)
# ipdb.set_trace()
l_in = L.standardize(l_in, offset=np.array([0.,], dtype='float32'),
    scale=np.array([255.,], dtype='float32'))

l_conv = L.Conv2DLayer(l_in,
    num_filters=2,
    filter_size=4,
    stride=(2, 2),
    pad=(0, 0),
    nonlinearity=LN.linear,
    name="conv0",
)
l_out = L.DenseLayer(l_conv,
    num_units=1,
    nonlinearity=LN.linear,
    name="output"
)


# x_dat = np.random.randn(*x_shape).astype('float32')
x_dat = np.random.randint(0, 255, x_shape, dtype='uint8')

check = theano.function(inputs=[x_uint], outputs=L.get_output(l_in))

r = check(x_dat)
print("\nx_dat: \n", x_dat)
print("\nr: \n", r.dtype, r)
print("\nx_dat / 255.: \n", x_dat / 255.)
print("all_close: ", np.allclose(r, x_dat / 255.))

# output = T.sum(L.get_output(l_out))
# # print(output)
# params = L.get_all_params(l_out)

# # f = theano.function(inputs=[x_float], outputs=[output, L.get_output(l_conv)])
# f = theano.function(inputs=[x_uint], outputs=[output, L.get_output(l_conv)])


# r = f(x_dat)
# print("\n function result:\n", r)
# # ipdb.set_trace()
# grads = theano.grad(output, wrt=params)
# print("\ngrads:\n", grads)
# # ipdb.set_trace()
# # h = theano.function(inputs=[x_float], outputs=grads)
# h = theano.function(inputs=[x_uint], outputs=grads)

# r = h(x_dat)
# print("\ngrad result:\n", r)
