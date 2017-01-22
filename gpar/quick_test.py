

import multiprocessing as mp

# import pycuda
# import pycuda.driver as cuda
import theano
# import theano.gpuarray
import time

def worker():
    import theano.gpuarray
    theano.gpuarray.use("cuda1")
    time.sleep(8)


# cuda.init()
# n_gpu = cuda.Device.count()
from pygpu import gpuarray
n_gpu = gpuarray.count_devices('cuda', 0)

print("n_gpu: ", n_gpu)

p = mp.Process(target=worker, args=())

p.start()
import theano.gpuarray
theano.gpuarray.use("cuda0")

time.sleep(8)

p.join()
