import numpy
import theano
from timeit import default_timer as timer
import time


v01 = theano.shared(numpy.random.random((1024, 1024)).astype('float32'),
                    target='dev0')
v02 = theano.shared(numpy.random.random((1024, 1024)).astype('float32'),
                    target='dev0')
v11 = theano.shared(numpy.random.random((1024, 1024)).astype('float32'),
                    target='dev1')
v12 = theano.shared(numpy.random.random((1024, 1024)).astype('float32'),
                    target='dev1')

time.sleep(0.5)
f = theano.function([], [theano.tensor.dot(v01, v02).transfer('dev0'),
                         theano.tensor.dot(v11, v12).transfer('dev1')])
r = f()
time.sleep(0.5)
t0 = timer()
for i in range(1000):
    r = f()
t1 = timer()
# print(f.maker.fgraph.toposort())
# print("r[0] type: ", type(r[0]))
# nr = numpy.asarray(r[0])
# print("nr[0] type: ", type(nr))
# time.sleep(0.5)
t1a = timer()
nr0 = numpy.asarray(r[0])
t2 = timer()
nr1 = numpy.asarray(r[1])
t3 = timer()

theano.printing.debugprint(f)
print("dot time: ", t1 - t0)
print("nr0 time: ", t2 - t1a)
print("nr1 time: ", t3 - t2)
print("tots time: ", t3 - t0)
