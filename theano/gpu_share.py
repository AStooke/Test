from theano import function, config, shared, tensor
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
# x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
# f = function([], tensor.exp(x).transfer('gpu'))
# print(f.maker.fgraph.toposort())
# r = []
# f()
# t0 = time.time()
# for i in range(iters):
#     r.append(f())
# t1 = time.time()
# print("Looping %d times took %f seconds" % (iters, t1 - t0))
# print("Result is %s" % (numpy.asarray(r),))
# t2 = time.time()
# print("Total time to last result transfered: ", t2 - t0)
# if numpy.any([isinstance(x.op, tensor.Elemwise) and
#               ('Gpu' not in type(x.op).__name__)
#               for x in f.maker.fgraph.toposort()]):
#     print('Used the cpu')
# else:
#     print('Used the gpu')


x_s = shared(numpy.asarray(rng.rand(vlen), config.floatX))
x = tensor.exp(x_s)
y = tensor.scalar('y')
z = x * y
f = function([y], [x.transfer('gpu'), z.transfer('gpu')])  # This makes a big difference, 10x
print(f.maker.fgraph.toposort())
r = []
t0 = time.time()
for i in range(iters):
    r.append(f(i))
t1 = time.time()
# time.sleep(0.01)
# t1 = time.time()
print("\nLooping %d times took %f seconds" % (iters, t1 - t0))
print(type(r[-1][0]))
for i in range(iters):
    out_1 = numpy.asarray(r[i][0])
    out_2 = numpy.asarray(r[i][1])
t2 = time.time()
print("Total time to last result transfered: ", t2 - t0)

# print("Result 1 is %s" % (numpy.asarray(r[1][0]),))
# print("Result -1 is %s" % (numpy.asarray(r[-1][0]),))
if numpy.any([isinstance(x.op, tensor.Elemwise) and
              ('Gpu' not in type(x.op).__name__)
              for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')
