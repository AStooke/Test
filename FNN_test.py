"""
Some FNN stuff
"""

import gtimer as gt
import numpy as np
import keras
import multiprocessing as mp
import psutil

from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf


in_size = 100
out_size = 10
# with tf.device('/gpu:0'):
model = Sequential()
model.add(Dense(300, input_shape=(in_size,)))
model.add(Activation('relu'))
model.add(Dense(300))
model.add(Activation('relu'))
model.add(Dense(300))
model.add(Activation('relu'))
model.add(Dense(out_size))
model.compile(loss='mse', optimizer='sgd')

weights = model.get_weights()
w_mbytes = 0
for w in weights:
    w_mbytes += w.nbytes / 1000000
print("w type: ", w[0].dtype, "\nw nMbytes: ", w_mbytes)
batch_size = 1000
num_loops = 100

x = []
for i in range(num_loops):
    x.append(np.random.rand(batch_size, in_size).astype(np.float32))
# x = np.random.rand(batch_size, in_size).astype(np.float32)
x_mbytes = x[0].nbytes / 1000000
print("x[0] nMbytes: ", x_mbytes)
print("total MBytes per batch + model: ", x_mbytes + w_mbytes)

gt.stamp('prep')
# y = model.predict(x[0])
# t0 = gt.stamp('first')


# # SERIAL / GPU
# for i in range(num_loops):
#     y = model.predict(x[i])
# t1 = gt.stamp('predict')
# print("Pred/s: {}".format(batch_size * num_loops / (t1 - t0)))
# print(gt.report())


nb_epoch = 1
y = []
for i in range(num_loops):
    y.append(np.random.randn(batch_size, out_size).astype(np.float32))

model.fit(x[-1], y[-1], batch_size=batch_size, nb_epoch=1, verbose=0, shuffle=False)

t0 = gt.stamp('prep_Fit')
for i in range(num_loops):
    model.fit(x[i], y[i], batch_size=batch_size, nb_epoch=nb_epoch, verbose=0, shuffle=False)
t1 = gt.stamp('fit')
print("Fit Speed: {}".format(batch_size * nb_epoch * num_loops / (t1 - t0)))








#  PARALLEL
# n_procs = 8

# def worker(rank):
#     p = psutil.Process()
#     p.cpu_affinity([rank])
#     gt.blank_stamp()
#     for i in range(num_loops):
#         y = model.predict(x[i])
#     t1 = gt.stamp('predict')
#     print("Rank: {}, Pred/s: {}".format(rank, batch_size * num_loops / (t1 - t0)))

#     gt.stop()


# procs = [mp.Process(target=worker, args=(rank,)) for rank in range(n_procs)]
# for p in procs:
#     p.start()
# for p in procs:
#     p.join()


