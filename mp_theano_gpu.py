
import theano
import theano.tensor as T
import theano.sandbox
import multiprocessing as mp
import time

# ANY OF THESE CAUSES THE ERROR
# import theano.sandbox.cuda
# from theano.sandbox.rng_mrg import MRG_RandomStreams
# import lasagne

x = T.scalar('x', dtype='float32')  # just exercising theano.tensor

def target():
    import theano.sandbox.cuda
    print("target about to use")
    theano.sandbox.cuda.use('gpu1')
    print("target is using")
    import lasagne
    time.sleep(15)  # (make it long enough to see master acquire GPU)
    print("target is exiting")

p = mp.Process(target=target)
p.start()
import theano.sandbox.cuda
print("master about to use")
theano.sandbox.cuda.use('gpu0')
print("master is using")
import lasagne
p.join()
print("master is exiting")
