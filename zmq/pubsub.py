
import zmq
import time
import multiprocessing as mp

PORT = "5556"
N_SUB = 5
N_MSG = 10

def publisher(barriers):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:%s" % PORT)
    print("PUBLISHER Ready")
    barriers[N_MSG].wait()
    time.sleep(3)
    print("PUBLISHER Starting loop")
    for i in range(N_MSG):
        socket.send_string(str(i))
        print("PUBLISHER Sent # ", i)
        # barriers[i].wait()
    print("PUBLISHER Finished loop")


def subscriber(rank, barriers):
    # time.sleep(1)
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:%s" % PORT)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    print("SUBSCRIBER {} Ready".format(rank))
    barriers[N_MSG].wait()
    # time.sleep(5)
    print("SUBSCIBER {} Starting loop".format(rank))
    for i in range(N_MSG):
        # barriers[i].wait()
        time.sleep(0.1)
        msg = socket.recv_string()
        print("SUBSCIBER {}, msg received: {}".format(rank, msg))


if __name__ == "__main__":

    barriers = [mp.Barrier(N_SUB + 1) for _ in range(N_MSG + 1)]
    mp.Process(target=publisher, args=(barriers,)).start()
    for rank in range(N_SUB):
        mp.Process(target=subscriber, args=(rank, barriers)).start()
