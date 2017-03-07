
import time

print("starting")
t0 = time.time()
try:
    time.sleep(5)
except KeyboardInterrupt:
    print("got interrupted")
t1 = time.time()
print("ending.  time elapsed: ", t1 - t0)
