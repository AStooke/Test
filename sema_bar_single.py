import time
from multiprocessing import Process, Semaphore
from multiprocessing.managers import BaseManager

"""
Example of single-use barrier implemented with semaphores.
"""

class Barrier:
    def __init__(self, n):
        self.n = n # make equal to the number of threads that need to adhere to the barrier.
        self.count = 0 # number of threads having reached the barrier.
        self.mutex = Semaphore(1) # this one is initialized as 'availalble'.
        self.barrier = Semaphore(0) # this one is initialized as 'blocked'.

    def wait(self):
        self.mutex.acquire() # mutex is used to make sure that only one thread can increment the counter at a time.
        self.count += 1 # another thread has reached the barrier.
        print 'Barrier count incremented to: {}'.format(self.count)
        if self.count == self.n: # if all the threads are at the barrier (i.e. the one executing is the last one).
            print('had count equal n')
            self.barrier.release() # release barrier so that one other thread can pass the barrier.acqurire() below.
        self.mutex.release() # let other threads pass the mutex.acquire() at the beginning and increment the count.
        print('about to acquire barrier')
        self.barrier.acquire() # this is where all threads will wait until the release in the if statement above is executed.
        print('about to release barrier')
        self.barrier.release() # whenever getting past the acquire, re-release so that another thread can get by (cascade of releases).


class MyManager(BaseManager):
    pass


def func1(b):
    print('Entered Func1')
    time.sleep(1)
    print('Func1 done sleeping')
    #
    b.wait()
    #
    print('func1 past barrier')
    return 

def func2(b):
    print('Entered Func2')
    time.sleep(3)
    print('Func2 done sleeping')
    #
    b.wait()
    #
    print('func2 past barrier')
    return    

if __name__ == '__main__':
    
    # Some people write this outside of main(), but it works here, too.
    MyManager.register('bar',Barrier) # need to set up a custom object to be shared
    manager = MyManager()
    manager.start()
    b = manager.bar(2) # create the shared barrier object.

    p1 = Process(target=func1, args=(b,))
    p2 = Process(target=func2, args=(b,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()  
