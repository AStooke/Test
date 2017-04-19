import multiprocessing as mp
import time



def worker(rank, stopper, pipe_read, lock):
    print("worker started, rank: {}".format(rank))
    i = 0
    while True:
        with lock:
            if not stopper.value:
                break
            k = pipe_read.recv()
            print("rank: {}, msg no: {}, msg val: {}".format(rank, i, k))
        time.sleep(0.01)


stopper = mp.RawValue('i', 1)
pipe_read, pipe_write = mp.Pipe(False)
lock = mp.Lock()

procs = list()
for rank in range(4):
    procs.append(mp.Process(target=worker, args=(rank, stopper, pipe_read, lock)))

for p in procs: p.start()

time.sleep(1)
for i in range(100):
    pipe_write.send(i)

while pipe_read.poll():
    time.sleep(0.1)
stopper.value = 0

# for i in range(100, 200):
    # pipe_write.send(i)
pipe_write.send("end")

for p in procs: p.join()
