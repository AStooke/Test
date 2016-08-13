import time
from multiprocessing import Process, Semaphore
from multiprocessing.managers import BaseManager

"""
Example of reusable barrier implemented with semaphores.
"""

class Barrier:
    def __init__(self, n):
        self.n = n # make equal to the number of threads that need to adhere to the barrier.
        self.count = 0 # number of threads having reached the barrier.
        self.mutexIn = Semaphore(1) # this one is initialized as 'availalble'.
        self.mutexOut = Semaphore(1) 
        self.barrier = Semaphore(0) # this one is initialized as 'blocked'.

    def wait(self):
        self.mutex.acquire() # mutex is used to make sure that only one thread can increment the counter at a time.
        self.count += 1 # another thread has reached the barrier.
        print 'Barrier count incremented to: {}'.format(self.count)
        
        if self.count == self.n: # if all the threads are at the barrier (i.e. the one executing is the last one).
            print('had count equal n')
            self.barrier.release() # release barrier so that one other thread can pass the barrier.acqurire() below.
            # (and in this case, keep the mutex blocked.)
        else:
            self.mutex.release() # let other threads pass the mutex.acquire() at the beginning and increment the count.
            
        print('about to acquire barrier')
        self.barrier.acquire() # this is where all threads will wait until the release in the if statement above is executed.
        self.count -= 1 # one less thread is still at the barrier (before releasing, so only one at a time here)
        print('about to release barrier')
        self.barrier.release() # whenever getting past the acquire, re-release so that another thread can get by (cascade of releases).
        if self.count == 0: # if all the threads have gotten past the barrier (i.e. the one executing is the last one).
            self.barrier.acquire() # block the barrier again so that the next time around the threads don't get past it.
            self.mutex.release() # allow the barrier to be re-used: now a thread can reach the counter increment (this has been blocked since count reached the value n.)
            print('barrier reset \n')


class MyManager(BaseManager):
    pass

MyManager.register('bar',Barrier)


def func1(b):
    print('Entered Func1')
    time.sleep(1)
    for i in range(4):
        
        print 'Func1 done sleeping: {}'.format(i)
        #
        b.wait()
    #
        print 'func1 past barrier: {}'.format(i)
    return 

def func2(b):
    print('Entered Func2')
    time.sleep(2)
    for i in range(4):
        
        print 'Func2 done sleeping: {}'.format(i)
        #
        b.wait()
    #
        print 'func2 past barrier: {}'.format(i)
    return    

if __name__ == '__main__':

    manager = MyManager()
    manager.start()

    b = manager.bar(2)

    p1 = Process(target=func1, args=(b,))
    p2 = Process(target=func2, args=(b,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()  
