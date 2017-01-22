"""
Some convnet stuff
"""

import numpy as np
import keras
import multiprocessing as mp
import psutil

from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense, Activation
import tensorflow as tf
import gtimer as gt

with tf.device('/cpu:0'):
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 256, 256)))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))

    weights = model.get_weights()
    w_mbytes = 0
    for w in weights:
        w_mbytes += w.nbytes / 1000000
    print("w type: ", w[0].dtype, "\nw nMbytes: ", w_mbytes)
    batch_size = 12
    num_loops = 50
    x = []

    for i in range(num_loops):
        x.append(np.random.rand(batch_size, 3, 256, 256).astype(np.float32))
    # x = np.random.rand(batch_size, 3, 256, 256).astype(np.float32)
    x_mbytes = x[0].nbytes / 1000000
    print("x[0] nMbytes: ", x_mbytes)
    print("total MBytes per batch + model: ", x_mbytes + w_mbytes)

    gt.start()
    y = model.predict(x[0])
    t0 = gt.stamp('first')

    for i in range(num_loops):
        y = model.predict(x[0])
    t1 = gt.stamp('predict')
    print("Pred/s: {}".format(batch_size * num_loops / (t1 - t0)))



# # Parallel (CPU)

# n_procs = 1


# def worker(rank):
#     print("In worker: ", rank)
#     # p = psutil.Process()
#     # p.cpu_affinity([rank])
#     gt.start()
#     y = model.predict(x[0])
#     t0 = gt.stamp('first')

#     gt.blank_stamp()
#     for i in range(num_loops):
#         print("worker: ", rank, "in loop: ", i)
#         y = model.predict(x[i])
#     t1 = gt.stamp('predict')
#     print("Rank: {}, Pred/s: {}".format(rank, batch_size * num_loops / (t1 - t0)))

#     gt.stop()


# procs = [mp.Process(target=worker, args=(rank,)) for rank in range(n_procs)]
# for p in procs:
#     p.start()
# for p in procs:
#     p.join()


# print(gt.report())


# model.train_on_batch()  # single gradient update
