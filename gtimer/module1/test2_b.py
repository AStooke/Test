import gtimer as gt
import time


@gt.timer_wrap
def monkey():
    print "in monkey"
    time.sleep(0.1)
    gt.stamp('oh-whoa')
