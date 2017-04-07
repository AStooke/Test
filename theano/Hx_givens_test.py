
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as LN
import lasagne.init as LI
import numpy as np

# import ipdb

# Build a simple MLP
input_dim = 10
output_dim = 10
hidden_size = 32
W_init = LI.GlorotUniform()
b_init = LI.Constant(0.)
input_var = T.matrix('input_var', dtype=theano.config.floatX)
l_in = L.InputLayer(shape=(None, input_dim), input_var=input_var)
l_hid_0 = L.DenseLayer(
    l_in,
    num_units=hidden_size,
    nonlinearity=LN.tanh,
    name="hidden_0",
    W=W_init,
    b=b_init,
)
l_hid_1 = L.DenseLayer(
    l_hid_0,
    num_units=hidden_size,
    nonlinearity=LN.tanh,
    name="hidden_1",
    W=W_init,
    b=b_init,
)
l_out = L.DenseLayer(
    l_hid_1,
    num_units=output_dim,
    nonlinearity=None,
    name="output",
    W=W_init,
    b=b_init,
)
params = L.get_all_params(l_out)
# ipdb.set_trace()
n_params = sum([p.get_value().size for p in params])

# Measure something to do with the output.
output_var = L.get_output(l_out)
cost_vars = T.square(output_var)
mean_var = T.sum(cost_vars)
# ipdb.set_trace()

# Build the Hx function.
grad_splits = theano.grad(mean_var, wrt=params)
grad_flat = T.concatenate([T.flatten(g) for g in grad_splits])
x = T.vector('x')
Hx_splits = T.grad(grad_flat.dot(x), wrt=params, disconnected_inputs='warn')
Hx_flat = T.concatenate([T.flatten(h) for h in Hx_splits])


# Build another one with inputs replaced by givens.
f = theano.function(inputs=[input_var, x], outputs=Hx_flat)

input_0 = np.random.rand(5, output_dim).astype(theano.config.floatX)
x_0 = np.random.rand(n_params).astype(theano.config.floatX)

r = f(input_0, x_0)
print(r)

input_shared = theano.shared(value=input_0, name='input_shared')
f_shared_input = theano.function(inputs=[x], outputs=Hx_flat, givens={input_var: input_shared})
r_shared_input = f_shared_input(x_0)
assert np.allclose(r, r_shared_input)

x_shared = theano.shared(value=x_0, name='x_shared')
f_shared_x = theano.function(inputs=[input_var], outputs=Hx_flat, givens={x: x_shared})
r_shared_x = f_shared_x(input_0)
assert np.allclose(r, r_shared_x)

print("all assertions passed")
