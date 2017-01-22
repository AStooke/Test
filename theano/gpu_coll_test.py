"""
Make sure I can use gpu collectives with theano shared variables.
"""

import os
import numpy as np
import multiprocessing as mp
# import ipdb
from timeit import default_timer as timer
import time

SIZE = [500, 5000]
N_ITR = 1

def target(d, b):
    os.environ["THEANO_FLAGS"] = "device=cuda1"
    import theano
    import pygpu
    from pygpu import collectives as gpucoll

    gpuctx = theano.gpuarray.get_context(None)

    local_id = gpucoll.GpuCommCliqueId(gpuctx)

    b.wait()

    local_id.comm_id = d["comm_id"]
    # print("worker local_id: \n", local_id)

    print("Worker at GpuComm")

    local_comm = gpucoll.GpuComm(local_id, 2, 1)

    print("Worker past GpuComm")

    # x_bare = pygpu.empty([10, 10], dtype='float32', context=gpuctx)
    # z_bare = pygpu.empty([10, 10], dtype='float32', context=gpuctx)
    # x = theano.shared(x_bare)
    # z = theano.shared(z_bare)
    # x.set_value(np.ones([10, 10], dtype='float32'))
    # res = local_comm.all_reduce(x_bare, 'sum', z_bare)
    # x = theano.shared(np.ones([10, 10], dtype='float32'))
    # z = theano.shared(np.zeros([10, 10], dtype='float32'))
    # y = theano.shared(np.zeros(SIZE, dtype='float32'))

    # x = theano.tensor.matrix('x')
    # y = theano.tensor.matrix('y')
    # z = x.dot(y)
    # f = theano.function([x, y], z.transfer(None))
    # x_dat = np.ones([3, 3], dtype='float32')
    # r = f(x_dat, x_dat)
    # local_comm.reduce(r, 'sum', root=0)

    x = theano.shared(np.ones([5, 5], dtype='float32'))
    f = theano.function([], x ** 2)
    x_is = f.input_storage[0]
    # print("\n worker shared before: \n", x.get_value())
    local_comm.reduce(x_is.data, 'sum', root=0)
    # print("\n worker shared after: \n", x.get_value())

    # # res = local_comm.all_reduce(x.container.data, 'sum', z.container.data)
    # time.sleep(1)
    # b.wait()
    # # time.sleep(1)
    # t0 = timer()
    # # for _ in range(N_ITR):
    # local_comm.broadcast(y.container.data, root=0)
    # y_after = y.get_value()
    # # print(y_after[-1,-1])
    # t1 = timer()
    # print("worker time: ", t1 - t0)
    # print(z.get_value())
    # print(y.get_value())

mgr = mp.Manager()
d = mgr.dict()
b = mp.Barrier(2)
p = mp.Process(target=target, args=(d,b))
p.start()


os.environ["THEANO_FLAGS"] = "device=cuda0"

import theano
import pygpu
from pygpu import collectives as gpucoll

gpuctx = theano.gpuarray.get_context(None)

local_id = gpucoll.GpuCommCliqueId(gpuctx)

d["comm_id"] = local_id.comm_id

b.wait()

print("Master at GpuComm")

local_comm = gpucoll.GpuComm(local_id, 2, 0)

print("Master got past GpuComm")

x = theano.shared(np.ones([5, 5], dtype='float32'))
f = theano.function([], x ** 2)
x_is = f.input_storage[0]
print("\nmaster before: \n", x.get_value())
local_comm.reduce(x_is.data, 'sum', x_is.data)
print("\nmaster after: \n", x.get_value())




# x = theano.tensor.matrix('x')
# y = theano.tensor.matrix('y')
# z = x.dot(y)
# f = theano.function([x, y], z.transfer(None))
# x_dat = np.ones([3, 3], dtype='float32')
# r = f(x_dat, x_dat)
# print(r)
# local_comm.reduce(r, 'sum', r, root=0)
# print(r)



# x_bare = pygpu.empty([10, 10], dtype='float32', context=gpuctx)
# z_bare = pygpu.empty([10, 10], dtype='float32', context=gpuctx)
# x = theano.shared(x_bare)
# z = theano.shared(z_bare)
# x.set_value(2 * np.ones([10, 10], dtype='float32'))
# res = local_comm.all_reduce(x_bare, 'sum', z_bare)
# x = theano.shared(3 * np.ones(SIZE, dtype='float32'))
# z = theano.shared(np.zeros([10, 10], dtype='float32'))
# # res = local_comm.all_reduce(x.container.data, 'sum', z.container.data)
# time.sleep(1)
# b.wait()
# # time.sleep(1)
# t0 = timer()
# # for _ in range(N_ITR):
# local_comm.broadcast(x.container.data)
# t1 = timer()

# tm = t1 - t0
# byt = np.prod(SIZE) * 4
# byt_tot = byt * N_ITR
# print("Master time: ", tm)
# print("bytes per move: {:,}".format(byt))
# print("total bytes moved: {:,}".format(byt_tot))
# print("bytes per time: {:,}".format(byt_tot / tm))
# THESE TIMES ARE BOGUS, THE BROADCAST CALL RETURNS IMMEDIATELY

# res = local_comm.all_reduce(y, 'sum', z)
# ipdb.set_trace()

p.join()


