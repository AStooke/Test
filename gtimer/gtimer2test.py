# gtimer2test
import gtimer2 as gt
import time

def a(x=0.1):
    time.sleep(x)

t1 = gt.Timer()

a()
t1.stamp('first')

for i in t1.timed_for([1, 2, 3]):
    a()
    t1.l_stamp('one')
    a(0.05)
    t1.l_stamp('two')



a()
t1.stamp('after')
t1.stop()
t1.print_report()



# named loops.