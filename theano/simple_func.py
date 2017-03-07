
import theano
import theano.tensor as T
import numpy as np

N = 5000

x_var = T.matrix('x', dtype='float32')
y_var = T.matrix('y', dtype='float32')

x = np.random.randn(N, N).astype('float32')
y = np.random.randn(N, N).astype('float32')

z_var = x_var.dot(y_var)

f = theano.function([x_var, y_var], z_var, profile=True)

f(x, y)
for _ in range(10):
    f(x, y)

f.profile.summary()
