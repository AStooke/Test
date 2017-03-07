
import multiprocessing as mp
import time
import os
import signal
import theano
import theano.tensor as T
import numpy as np
import threading

THEANO = True
THREADED = True


def sleep_worker():

    t = 5
    print("worker starting to sleep for {} s".format(t))
    t0 = time.time()
    try:
        time.sleep(t)
    except KeyboardInterrupt:
        print("worker interrupted")
    t1 = time.time()
    print("worker elapsed: ", t1 - t0)


def theano_worker(f, x):
    print("worker starting function execution")
    t0 = time.time()
    try:
        r = f(x)
        print(r[0])
    except KeyboardInterrupt:
        print("got interrupted")
    finally:
        t1 = time.time()
        print("worker elapsed: ", t1 - t0)


def thread_worker(f, x):
    print("thread worker starting function")
    t0 = time.time()
    thrd = threading.Thread(target=f, args=(x,))
    thrd.daemon = True
    thrd.start()
    while thrd.is_alive():
        thrd.join(0.1)
    # try:
    #     thrd = threading.Thread(target=f, args=(x,))
    #     thrd.daemon = True
    #     thrd.start()
    #     while thrd.is_alive():
    #         thrd.join(0.1)
    # except KeyboardInterrupt as e:
    #     print("worker interrupted")
    #     t1 = time.time()
    #     print("worker elapsed: ", t1 - t0)
    #     raise e
    t1 = time.time()
    print("worker elapsed: ", t1 - t0)


if not THEANO:
    p = mp.Process(target=sleep_worker)
else:
    x = T.matrix('x')
    x_dat = np.ones([5000, 5000], dtype='float32')
    f = theano.function([x], x.dot(x))
    if not THREADED:
        target = theano_worker
    else:
        target = thread_worker
    p = mp.Process(target=target, args=(f, x_dat))


t = 2
print("master starting process and interrupting after {} s".format(t))
t0 = time.time()
p.start()
time.sleep(t)
os.kill(p.pid, signal.SIGINT)
p.join()
t1 = time.time()
print("master elapsed: ", t1 - t0)
