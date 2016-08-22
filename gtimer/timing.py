import time
from timeit import default_timer as timer

def procedure():
    time.sleep(2.5)


# what does default_timer do
t0 = timer()
procedure()
print timer() - t0, "default_timer"

# measure process time
t0 = time.clock()
procedure()
print time.clock() - t0, "seconds process time"


# measure wall time
t0 = time.time()
procedure()
print time.time() - t0, "seconds wall time"