import zmq
import time
import multiprocessing as mp
import numpy as np
from timeit import default_timer as timer
PORTS = ["5556", "5555"]
COPY = False
TRACK = False
SLEEPY = 6
PAUSE_AFTER_SEND = False

sleeping = " ..sleeping {}".format(SLEEPY)


def server(barrier):
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind("tcp://*:%s" % PORTS[0])
    print("Running server on port: ", PORTS[0])
    # c2 = zmq.Context()
    socket2 = context.socket(zmq.PAIR)
    socket2.bind("tcp://*:%s" % PORTS[1])
    print("Running server on port: ", PORTS[1])

    x = np.ones([10 ** 8, 10], dtype='float32')
    x_size_MB = x.dtype.itemsize * x.size // 1024 // 1024
    print("SERVER made x, size: {} MB".format(x_size_MB) + sleeping)
    print("SERVER x dtype: {}, shape: {}".format(x.dtype, x.shape))
    time.sleep(SLEEPY)
    t_s = timer()
    z = x.copy()
    t_c = timer() - t_s
    print("SERVER copied x in {} seconds".format(t_c) + sleeping)
    time.sleep(SLEEPY)
    print("SERVER sending, copy: {}".format(COPY))
    md = dict(dtype=x.dtype.name, shape=x.shape)
    socket.send_json(md)
    socket2.send_json(md)
    socket.send(x, copy=COPY, track=TRACK)
    socket2.send(x, copy=COPY, track=TRACK)
    if PAUSE_AFTER_SEND:
        print("SERVER" + sleeping)
        time.sleep(SLEEPY)
    barrier.wait()
    barrier.wait()


def client(barrier, port):

    context = zmq.Context()
    print("Connecting to server with port %s" % port)
    socket = context.socket(zmq.PAIR)
    socket.connect("tcp://localhost:%s" % port)
    print("CLIENT at first barrier")
    md = socket.recv_json()
    barrier.wait()
    t_s = timer()
    msg = socket.recv(copy=COPY, track=TRACK)
    t_msg = timer() - t_s
    print("CLIENT msg time: {} seconds".format(t_msg))
    x = np.frombuffer(msg, dtype=md['dtype'])
    x = x.reshape(md['shape'])
    print("CLIENT dtype: {},  shape: {}".format(x.dtype, x.shape))
    print("CLIENT" + sleeping)
    time.sleep(SLEEPY)
    barrier.wait()


if __name__ == "__main__":

    barrier = mp.Barrier(2)
    mp.Process(target=server, args=(barrier,)).start()
    [mp.Process(target=client, args=(barrier, port)).start() for port in PORTS]

