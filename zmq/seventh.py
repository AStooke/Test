
import zmq
import numpy
import multiprocessing as mp
import numpy as np
from timeit import default_timer as timer
import time


PORT = "5555"
N = 100


def server(barrier):
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind("tcp://*:%s" % PORT)
    time.sleep(4)
    # xs = [np.ones(3) for _ in range(N)]
    xs = [np.array(1) for _ in range(N)]
    # barrier.wait()
    # print("SERVER start loop")
    # for x in xs:
    #     md = dict(dtype=x.dtype.name, shape=x.shape)
    #     socket.send_json(md)
    # print("SERVER done with json loop")
    barrier.wait()
    print("SERVER start array loop")
    for x in xs:
        socket.send_string(x.dtype.name)
        sh_str = str(x.shape).lstrip('(').rstrip(')').rstrip(',')
        socket.send_string(sh_str)
        socket.send(x, copy=False)
    barrier.wait()
    print("SERVER start single string loop")
    for x in xs:
        socket.send_string(str(x))



def client(barrier):
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect("tcp://localhost:%s" % PORT)
    # barrier.wait()
    # time.sleep(2)
    # t_s = timer()
    # for i in range(N):
    #     msg = socket.recv_json()
    # t_r = timer() - t_s
    # print("CLIENT json receive time: ", t_r)

    barrier.wait()
    time.sleep(2)
    t_s = timer()
    for i in range(N):
        dtype = socket.recv_string()
        shape = socket.recv_string()
        if not shape:
            shape = ()
        else:
            shape = [int(s) for s in shape.split(',')]
        x = np.frombuffer(socket.recv(copy=False), dtype=dtype)
    t_r = timer() - t_s
    print("CLIENT string receive and parse time: ", t_r)
    print(shape)
    barrier.wait()
    time.sleep(2)
    t_s = timer()
    for i in range(N):
        x = socket.recv_string()
    t_r = timer() - t_s
    print("CLIENT single string receive and float time: ", t_r)


if __name__ == "__main__":

    barrier = mp.Barrier(2)
    mp.Process(target=server, args=(barrier,)).start()
    mp.Process(target=client, args=(barrier,)).start()
