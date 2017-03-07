

import theano
import theano.gpuarray
import multiprocessing as mp
import numpy as np


def worker(f):
    theano.gpuarray.use('cuda0')
    r = f()

    print("worker r: ", r)


x = theano.shared(np.ones([3, 3], dtype='float32'))

f = theano.function([], x.dot(x))
theano.gpuarray.use('cuda0')

p = mp.Process(target=worker, args=(f,))
p.start()


r = f()
print("master r: ", r)

x.set_value(np.zeros([3, 3], dtype='float32'))

r = f()
print("master 2nd r: ", r)

p.join()
