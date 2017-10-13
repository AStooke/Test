
import numpy as np
import theano
import theano.gpuarray
import time


theano.gpuarray.use("cuda")

SIZE = 200
NUM = 20
LOOPS = 200

xs = [np.zeros([SIZE, SIZE], dtype='float32') for _ in range(NUM)]

total_x = np.zeros([NUM, SIZE, SIZE], dtype='float32')


shareds = [theano.shared(x) for x in xs]
total_shared = theano.shared(total_x)

ys = [None] * len(xs)


for _ in range(5):
    for i, s in enumerate(shareds):
        ys[i] = s.get_value()

t0 = time.time()
for _ in range(LOOPS):
    for i, s in enumerate(shareds):
        ys[i] = s.get_value()
t1 = time.time()
print("Time to read small variables: {:.3f}".format(t1 - t0))

for _ in range(5):
    total_y = total_shared.get_value()

t0 = time.time()
for _ in range(LOOPS):
    total_y = total_shared.get_value()
t1 = time.time()
print("Time to read large variable: {:.3f}".format(t1 - t0))


for _ in range(5):
    for i, s in enumerate(shareds):
        s.set_value(ys[i])

t0 = time.time()
for _ in range(LOOPS):
    for i, s in enumerate(shareds):
        s.set_value(ys[i])
t1 = time.time()
print("Time to write small variables: {:.3f}".format(t1 - t0))

for _ in range(5):
    total_shared.set_value(total_y)

t0 = time.time()
for _ in range(LOOPS):
    total_shared.set_value(total_y)
t1 = time.time()
print("Time to write large variable: {:.3f}".format(t1 - t0))
