

import theano
import theano.tensor as T
import multiprocessing as mp
import time
import numpy as np
# import lasagne

def target():
    # time.sleep(1)
    import theano.sandbox.cuda
    print("target about to use")
    theano.sandbox.cuda.use('gpu1')
    print("target is using")
    import lasagne
    time.sleep(15)
    print("target is exiting")

x = T.scalar('x', dtype='float32')

p = mp.Process(target=target)

p.start()

time.sleep(1)
# import lasagne
import theano.sandbox.cuda
print("master about to use")
theano.sandbox.cuda.use('gpu0')
print("master is using")
import lasagne
time.sleep(4)
print("master will join")

p.join()
print("master is exiting")
