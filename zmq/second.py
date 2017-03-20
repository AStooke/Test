import zmq
import time
import sys
from  multiprocessing import Process
import numpy as np

def server(port="5556"):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)
    print("Running server on port: ", port)
    # serves only 5 request and dies
    x = np.ones([3, 3])
    print(x)
    md = dict(shape=x.shape)
    socket.send_pyobj(md)
    socket.send(x)

    # for reqnum in range(5):
    #     # Wait for next request from client
    #     message = socket.recv_string()
    #     print("Received request #%s: %s" % (reqnum, message))
    #     socket.send_string("World from %s" % port)

def client(port="5556"):

    context = zmq.Context()
    print( "Connecting to server with port %s" % ports)
    socket = context.socket(zmq.REQ)
    socket.connect ("tcp://localhost:%s" % port)
    md = socket.recv_pyobj(flags=0)
    print(md)
    msg = socket.recv()
    buf = buffer(msg)
    x = np.frombuffer(buf, dtype='float64')
    x = x.reshape(md['shape'])
    print(x)



    # for request in range (20):
    #     print( "Sending request ", request,"...")
    #     socket.send_string("Hello")
    #     message = socket.recv_string()
    #     print( "Received reply ", request, "[", message, "]")
    #     time.sleep (1)


if __name__ == "__main__":
    # Now we can run a few servers
    server_ports = range(5550,5558,2)
    for server_port in server_ports:
        Process(target=server, args=(server_port,)).start()

    # Now we can connect a client to all these servers
    Process(target=client, args=(server_ports,)).start()

