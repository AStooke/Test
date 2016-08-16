from timer4 import Timer
import time


t = Timer('steve')
time.sleep(0.1)

t.stamp('first')
time.sleep(0.2)
t.stamp('second')
for i in t.timed_for([1, 2], ['l_1']):
    print i
    time.sleep(0.1)
    t.l_stamp('l_1')
t.stamp('after loop')


for i in t.timed_for([1, 2, 3], ['l_2']):
    print i
    time.sleep(0.05)
    t.l_stamp('l_2')
t.stop()
t.print_report()
