import theano
import numpy as np
# from timeit import default_timer as timer
import time
import gtimer as gt

SIZE = [1, 1]
N_ITR = 1
x = theano.shared(np.ones(SIZE, dtype='float32'))

time.sleep(5)
t0 = gt.start()
# for _ in range(N_ITR):
y = x.get_value()
# print(y[0, 0])
t1 = gt.stamp("done", qp=True)
tm = t1 - t0
byt = np.prod(SIZE) * 4
byt_tot = byt * N_ITR
byt_per_tm = byt_tot / tm
tm_per_itr = tm / N_ITR

print("bytes per: {:,}".format(byt))
print("total bytes: {:,}".format(byt_tot))
print("bytes per time: {:,}".format(byt_per_tm))
print("time per itr: {}".format(tm_per_itr))
