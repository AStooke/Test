import gtimer as gt
import time
import globalholder as g

time.sleep(0.1)
gt.stamp('first')
for i in gt.timed_for([1, 2, 3]):
    time.sleep(0.1)
    gt.l_stamp('l_first')
    for j in gt.timed_for([1, 2, 3]):
        time.sleep(0.02)
        gt.l_stamp('l_second')
time.sleep(0.05)
gt.stamp('second')
gt.stop()

print g.tif.times.stamps
print g.tif.times.stamps_itrs

