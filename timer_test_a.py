from timer5 import G_Timer, g_timers
import time
import timer_test_b

t1 = G_Timer('yeah')
print "t1 in timer_test_a: ", t1

time.sleep(0.1)

t1.stamp('first')

for i in [1,2,3]:
    timer_test_b.funky()
    t1.graft('whoa', 'second')
    time.sleep(0.1)
    timer_test_b.monkey()
    time.sleep(0.1)
    t1.graft('there', 'second')

t1.stamp('second')

t1.stop()

# t1.graft('whoa', 'second')

# timer_test_b.funky()

# t1.graft('whoa', 'second')

t1.print_report()

g_timers['there'].print_report()

t1.print_structure()