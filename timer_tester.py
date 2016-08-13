# Timer Tester.
from sandbox.adam.timer3 import Timer
import time



# t = Timer()
# with t:
#     time.sleep(1)
#     t.stamp('first')
#     time.sleep(1)
#     t.interval('interval')
#     time.sleep(1)
#     t.stamp(2)
#     time.sleep(1)
#     t.interval('interval')
#     time.sleep(1)
#     t.stamp(3)
#     t.interval('interval')
#     time.sleep(1)
#     t.interval('interval')
#     t.stamp("last")

# # t.stamp('outside')
# print t.times

t = Timer(name='BigPoppa')
print t.times.self

time.sleep(0.1)
t.stamp('before loop')

# t.interval('loopdi')
# print 'start'
# t.enter_loop('wheel')
# for i in [1, 2, 3]:
#     print i
#     t.loop_start()
#     time.sleep(0.1)
#     t.l_stamp('first')
#     time.sleep(0.1)
#     t.l_interval('yeah')
#     # t.interval('during loop')
#     time.sleep(0.1)
#     if i == 2:
#         time.sleep(0.15)
#         t.l_stamp('branch')
#     t.l_stamp('second')
#     t.l_interval('yeah')
#     time.sleep(0.1)
#     t.loop_end('last')
# print 'outside the loop'
# t.exit_loop()
# t.interval('loopdi')
i = 0
for _ in t.timed_while('wheel'):
    i += 1
    # if i == 2:
    #     time.sleep(0.02)
    #     t.while_condition = False   # These two lines are
    #     continue                    # equivalent to 'break'
    time.sleep(0.1)
    t.l_stamp('first')
    time.sleep(0.1)
    t.l_interval('yeah')
    time.sleep(0.1)
    # if i == 2:
    #     time.sleep(0.15)
    #     t.l_stamp('branch')
    t.l_stamp('second')
    t.l_interval('yeah')
    time.sleep(0.1)
    t.l_stamp('last')
    if i > 3:                      # Just put the while condition at the end of
        t.while_condition = False  # everything, then it will behave the same!

t.stop()


t2 = Timer('LilMomma', save_itrs=False)
print 'start2222222'
for i in t2.timed_for('spoke2', [1, 2, 3]):
    print i
    time.sleep(0.1)
    t2.l_stamp('first2')
    time.sleep(0.1)
    t2.l_interval('yeah2')
    # t.interval('during loop')
    time.sleep(0.1)
    if i == 2:
        time.sleep(0.15)
        t2.l_stamp('branch2')
    t2.l_stamp('second2')
    t2.l_interval('yeah2')
    time.sleep(0.1)
    t2.l_stamp('last2')
print 'outside the loop22222'


# t.enter_loop('cartwheel')
# for i in [1,2,3]:
#     t.loop_start()
#     time.sleep(0.1)
#     t.loop_end()
# t.exit_loop()

# time.sleep(0.1)
# t.stamp('after loop')
t2.stop()
# t.stop()
# time.sleep(0.1)
# t.stamp('whoops')
# print t.times, "\n"
# print t.times.total, "\n"
# print t.times.stamps, "\n"
# print t.times.intervals, "\n"
# print t.times.loops, "\n"
# print 'wheel \n\n'
# print t.times.loops['wheel'].stamps, "\n"
# print t.times.loops['wheel'].intervals, "\n"
# # print t.times.loops['wheel'].stamps_itrs, "\n"
# # print t.times.loops['wheel'].intervals_itrs, "\n"
# print t.times.loops['wheel'].total, "\n"
# # print t.times.loops['wheel'].total_itrs, "\n"
# print 'spoke2 \n\n'
# print t.times.loops['spoke2'].stamps, "\n"
# print t.times.loops['spoke2'].intervals, "\n"
# # print t.times.loops['spoke2'].stamps_itrs, "\n"
# # print t.times.loops['spoke2'].intervals_itrs, "\n"
# print t.times.loops['spoke2'].total, "\n"
# print t.times.loops['spoke2'].total_itrs, "\n"
# print t.times
# print "\n"
# print t.loop_times
# print "\n"
# print t.loop_time_sums

print t.times.stamps

t.times.absorb(t2.times, 'first')

print t.times.children
print t2.times

print t.times
print t2.times.parent

t.times.absorb(t2.times, 'first')

print t.times.children
x = t.times.children['first'][0]
print '\n\n'
print x.stamps, '\n'
print x.intervals, '\n'
print x.total, '\n'
print x.self, '\n'
print x.loops, '\n'
# print x.loops['spoke2'].total_itrs, '\n'
print x.loops['spoke2'].total, '\n'
# print x.loops['spoke2'].stamps_itrs, '\n'
print x.loops['spoke2'].stamps, '\n'
# print x.loops['spoke2'].intervals_itrs, '\n'
print x.loops['spoke2'].intervals, '\n'