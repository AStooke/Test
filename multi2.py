from multiprocessing import Process, Lock
import os
import time

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name, lock, p_time=0):
    time.sleep(p_time) # hack so that they each start at about the same time
    for i in range(5):
        # lock.acquire()
        # info('function f')
        print('{0} {1}\n'.format(name, i))
        
        # finally:
        # lock.release()
        time.sleep(0.0001) # hack so that other threads may operate in between iterations

if __name__ == '__main__':
    info('main line')
    pro = []
    names = ('bob','joe','sue','alice')
    pauses = (0.005, 0.004, 0.003, 0.002) # hack so that they each start at about the same time
    lock = Lock()
    for i in range(4):
        p = Process(target=f, args=(names[i],lock))
        pro.append(p)
    for i in range(4):
        pro[i].start()

    for i in range(4):
        pro[i].join()