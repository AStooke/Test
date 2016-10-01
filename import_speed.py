from timeit import default_timer as timer
import importlib
import numpy

t_start = timer()
n = 1000
for _ in range(n):
    importlib.reload(numpy)
t_end = timer()

print("\nDid {} reloads of numpy in {:.2f}s.".format(n, t_end - t_start))
