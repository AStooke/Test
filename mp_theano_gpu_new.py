"""
Now running the new Theano GPU Array backend:
i.e. environment variable:
THEANO_FLAGS="dev0->cuda0;dev1->cuda1"
to get multiple GPUs on one process.
Still want to see if another process can be spawned OK.
"""
import numpy
import multiprocessing as mp
import time

########  OLD  ####################3
# import theano
# import theano.tensor as T
# import theano.sandbox

# ANY OF THESE CAUSES THE ERROR
# import theano.sandbox.cuda
# from theano.sandbox.rng_mrg import MRG_RandomStreams
# import lasagne

# x = T.scalar('x', dtype='float32')  # just exercising theano.tensor

######################################

# import theano
# import lasagne
import importlib


def do_theano(theano):
    v01 = theano.shared(numpy.random.random((1024, 1024)).astype('float32'),
                        target=None)
    v02 = theano.shared(numpy.random.random((1024, 1024)).astype('float32'),
                        target=None)
    v11 = theano.shared(numpy.random.random((1024, 1024)).astype('float32'),
                        target=None)
    v12 = theano.shared(numpy.random.random((1024, 1024)).astype('float32'),
                        target=None)

    f = theano.function([], [theano.tensor.dot(v01, v02),
                             theano.tensor.dot(v11, v12)])
    r = f()


def target():
    print("target starts")
    import theano
    importlib.reload(theano)
    time.sleep(3)
    do_theano(theano)
    # do_theano()
    print("target done")

p = mp.Process(target=target)
p.start()

import theano
do_theano(theano)
# do_theano()

# import theano.sandbox.cuda
# print("master about to use")
# theano.sandbox.cuda.use('gpu0')
# print("master is using")
# import lasagne
p.join()
print("master is exiting")

