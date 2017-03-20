
import zmq
import time
import multiprocessing as mp
import numpy as np
from timeit import default_timer as timer


PORT = "5554"
N_SUB = 8
N_MSG = 1
DTYPE = 'float32'
SHAPE = (10 ** 7,)

def publisher(barriers):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:%s" % PORT)
    xs = [np.ones(SHAPE, dtype=DTYPE) for _ in range(N_MSG)]
    print("PUBLISHER Ready")
    time.sleep(5)  # still needs a little extra time to connect
    t_s = timer()
    print("PUBLISHER Starting loop")
    for i, x in enumerate(xs):
        socket.send(x, copy=False)
        print("PUBLISHER Sent array # ", i)
    t_send = timer() - t_s
    print("PUBLISHER time to send: {} seconds.".format(t_send))
    barrier.wait()
        # time.sleep(.1)
        # barriers[i].wait()
    print("PUBLISHER Finished loop")
    time.sleep(10)


def subscriber(rank, barrier):
    # time.sleep(1)
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:%s" % PORT)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    print("SUBSCRIBER {} Ready".format(rank))
    barrier.wait()
    # time.sleep(5)
    print("SUBSCIBER {} Starting loop".format(rank))
    xs = list()
    t_s = timer()
    for i in range(N_MSG):
        msg = socket.recv(copy=False)
        xs.append(np.frombuffer(msg, dtype=DTYPE))
        print("SUBSCIBER {}, msg # received: {}".format(rank, i))
        # time.sleep(1)
    t_r = timer() - t_s
    print("SUBSCRIBER {} time to receive: {} seconds".format(rank, t_r))
    time.sleep(10)


if __name__ == "__main__":

    barrier = mp.Barrier(N_SUB + 1)
    mp.Process(target=publisher, args=(barrier,)).start()
    for rank in range(N_SUB):
        mp.Process(target=subscriber, args=(rank, barrier)).start()
