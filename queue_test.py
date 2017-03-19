
import multiprocessing as mp
import time

N = 10
R = 5


def queue_worker(queue, rank):

    for i in range(N):
        time.sleep(0.1)
        queue.put(rank)



def run_queue():
    queue = mp.Queue()

    procs = [mp.Process(target=queue_worker, args=(queue, rank)) for rank in range(R)]

    [p.start() for p in procs]

    n_recv = 0
    while n_recv < N * R:
        print(queue.get())
        n_recv += 1

    [p.join() for p in procs]


def run_simple_queue():
    sq = mp.SimpleQueue()
    procs = [mp.Process(target=queue_worker, args=(sq, rank)) for rank in range(R)]
    [p.start() for p in procs]
    n_recv = 0
    while n_recv < N * R:
        print(sq.get())
        n_recv += 1
    [p.join() for p in procs]


def pipe_worker(conn_send, lock, rank):

    for i in range(N):
        time.sleep(0.1)
        with lock:
            conn_send.send(rank)

def run_pipe():
    conn_recv, conn_send = mp.Pipe(False)
    lock = mp.Lock()
    procs = [mp.Process(target=pipe_worker, args=(conn_send, lock, rank)) for rank in range(R)]
    [p.start() for p in procs]
    n_recv = 0
    while n_recv < N * R:
        print(conn_recv.recv())
        n_recv += 1
    [p.join() for p in procs]
