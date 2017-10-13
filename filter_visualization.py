
import theano
import theano.tensor as T
import numpy as np
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as LN

import matplotlib.pyplot as plt

# import ipdb


l_in = L.InputLayer(shape=(None, 5, 24, 24))
num_filters = 3
l_conv = L.Conv2DLayer(l_in, num_filters=num_filters,
    filter_size=(8, 8), stride=3, nonlinearity=LN.rectify)

l_out = L.DenseLayer(l_conv, num_units=1, nonlinearity=LN.rectify)

params = L.get_all_params(l_out)

conv_params = params[0]

assert conv_params.ndim > 2

conv_vals = conv_params.get_value()

fig, axes = plt.subplots(1, num_filters)
for i, ax in enumerate(axes):
    pos = ax.imshow(conv_vals[i][0], vmin=-.3, vmax=+.3, cmap="coolwarm")
fig.colorbar(pos)
# ipdb.set_trace()
fig.suptitle("yeah layer 1")
plt.show()
