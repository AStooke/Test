
import theano
import theano.tensor as T
import numpy as np

x = T.matrix('x', dtype='float32')
s = theano.shared(np.random.rand(10, 2).astype('float32'), name='s')
start = T.lscalar()
end = T.lscalar()
out = T.sum(x, axis=0)
v = T.lvector('v')

f = theano.function([start, end], out, givens=[(x, s[start:end])])
g = theano.function([v], out, givens=[(x, s[v])])
f.trust_input = True
d = s.get_value()
print("s value: \n", d)

def make_idx(a, b):
    return (np.array(a), np.array(b))

idx = make_idx(0, 2)
print("f(0, 2): \n", f(*idx))
print("np.sum(d[0:2], axis=0): \n", np.sum(d[0:2], axis=0))

idx = make_idx(0, 10)
print("f(0, 10): \n", f(*idx))
print("np.sum(d[0:10], axis=0): \n", np.sum(d[0:10], axis=0))

idx = make_idx(8, 10)
print("f(8, 10): \n", f(*idx))
print("np.sum(d[8:10], axis=0): \n", np.sum(d[8:10], axis=0))

idx = [0, 9, 1, 8]
print("g([0, 10, 1, 9]): \n", g(idx))
print("np.sum(d[idx], axis=0): \n", np.sum(d[idx], axis=0))
