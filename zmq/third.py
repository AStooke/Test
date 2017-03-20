import zmq
import time
import sys
from  multiprocessing import Process

def server(port="5556"):
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind("tcp://*:%s" % port)
    print("Running server on port: ", port)
    # serves only 5 request and dies
    for msgnum in range(5):
        socket.send_string("Server message to client")
        msg = socket.recv_string()
        print(msg)
        time.sleep(1)


def client(port="5556"):

    context = zmq.Context()
    print( "Connecting to server with port %s" % port)
    socket = context.socket(zmq.PAIR)
    socket.connect ("tcp://localhost:%s" % port)
    for msgnum in range (5):
        msg = socket.recv_string()
        print(msg)
        socket.send_string("client message #" + str(msgnum))
        time.sleep(1)

if __name__ == "__main__":
    # Now we can run a few servers
    Process(target=server, args=()).start()


    # Now we can connect a client to all these servers
    Process(target=client, args=()).start()

