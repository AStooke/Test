import gtimer as gt
import time
import test2_b
# import data_glob as g


@gt.timer_wrap
def funky():
    print "inside funky"
    time.sleep(0.1)
    gt.stamp('wrapped_1')


time.sleep(0.1)
gt.stamp('first')
# for i in [1, 2, 3]:
for i in gt.timed_for([1, 2, 3]):
    print "i: ", i
    # print "g.lf: ", g.lf
    # print "g.lf.stamps: ", g.lf.stamps
    time.sleep(0.1)
    funky()
    gt.stamp('l1_first')
    for j in gt.timed_for([1, 2], 'loop2'):
        # for j in [1, 2, 3]:
        print "j: ", j
        # print "g.lf: ", g.lf
        # print "g.lf.stamps: ", g.lf.stamps
        time.sleep(0.1)
        gt.stamp('l2_first')
        funky()

        funky()
        gt.stamp('after_funky')
        # for k in gt.timed_for([1, 2], 'loop3'):
        #     print "k: ", k
        #     time.sleep(0.03)
        #     gt.stamp('l3_first')
        #     funky()
        #     gt.stamp('l3_after_funky')
        # for m in gt.timed_for([1, 2], 'loop4'):
        #     print "m: ", m
        #     time.sleep(0.01)
        #     gt.stamp('l_fourth')
        time.sleep(0.1)
        gt.stamp('l2_second')
time.sleep(0.1)
gt.stamp('second')
gt.open_new_timer('yeah')
time.sleep(0.1)
gt.stamp('yeah_1')
test2_b.monkey()
time.sleep(0.1)
gt.stamp('yeah_2')
gt.close_last_timer()
gt.stop('third')

gt.print_report()
# print g.tif.times.stamps
# print g.tif.times.stamps_itrs


