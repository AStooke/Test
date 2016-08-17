from gtimer import G_Timer
import time

t1, t2 = G_Timer([x for x in ['whoa', 'there']])
print "t1 in timer_test_b: ", t1
print "t2 in timer_test_b: ", t2


def funky():
    t1.clear()
    for i in t1.timed_for([1, 2, 3], ['booya']):
        time.sleep(0.2)
        t1.l_stamp('booya')
    t1.stop()


def monkey():
    t2.clear()
    time.sleep(0.05)
    t2.stamp('kashah')
