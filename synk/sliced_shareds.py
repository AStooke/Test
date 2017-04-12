
import numpy as np
import theano
import theano.tensor as T
import synkhronos as synk

synk.fork()

s = theano.shared(np.random.rand(10, 2).astype('float32'), 's')
x = T.matrix('x', dtype='float32')
d = np.random.rand(10, 3).astype('float32')
print("d: \n", d)
print("s: \n", s.get_value())

f = synk.function([], (2 * T.sum(s, axis=0), "sum"), sliceable_shareds=[s])
g = synk.function([x], (2 * T.sum(x, axis=0), "sum"))
h = synk.function([], (2 * T.sum(x, axis=0), "sum"), sliceable_shareds=[(x, s)])


synk.distribute()

r = f()
print("f result: \n", r)

r2 = f(num_slices=2)
print("f slice2 result: \n", r2)


sd = g.build_inputs(d)
r = g(sd)
print("g result: \n", r)

r2 = g(sd, num_slices=2)
print("g slice2 result: \n", r2)

r = h()
print("h result: \n", r)

r2 = h(num_slices=2)
print("h slice2 result: \n", r2)
