from timer4 import Timer
import time


def a(sleep_time=0.1):
    time.sleep(sleep_time)


def print_attr(obj):
    print obj
    for k, v in obj.__dict__.iteritems():
        print k, ":\t\t", v

t = Timer('original')

a()
t.stamp('first')
a(0.2)
t.stamp('second')
# i = 0
# a()
l_stamp_list = ['l_first', 'l_second', 'l_third']
for i in t.timed_for([1, 2, 3], l_stamp_list):
    a()
    t.l_stamp('l_first')
    a(0.2)
    if i == 2:
        t.l_stamp('l_second')
    else:
        a()
        t.l_stamp('l_third')
    # a()
a()
t.stamp('third')
t.stop()

t2 = Timer('t2')
a()
t2.stamp('2first')
a()
t2.stamp('2second')
t2.stop()

t3 = Timer('t3')
a(0.3)
t3.stamp('3first')
for i in t3.timed_for([1, 2, 3], ['3l_1sdfasldfi12345', '3l_2']):
    a()
    t3.l_stamp('3l_1sdfasldfi12345')
    a(0.05)
    t3.l_stamp('3l_2')
t3.stop()

t.times.print_report()


print "\n t.times atrributes"
print_attr(t.times)

print "\n t2.times atrributes"
print_attr(t2.times)

print "\n t3.times atrributes"
print_attr(t3.times)

# print "\n\nabsorbing t3 into t2 at '2first'\n\n"
# t2.absorb(t3, '2first')

# print "\n\nabsorbing t2 into t at 'first'\n\n"
# t.absorb(t2, 'first')

# print "\n t.times atrributes"
# print_attr(t.times)

# print "\n t.times.children[0] attributes"
# print_attr(t.times.children[0])

# print "\n t.times.children[0].children[0] attributes"
# print_attr(t.times.children[0].children[0])

# print "\nabsorbing t2 into t at 'first'\n\n"
# t.absorb(t2, 'first')

# print "\n t.times atrributes"
# print_attr(t.times)

# print "\n t.times.children[0] attributes"
# print_attr(t.times.children[0])

# print "\n t.times.children[0].children[0] attributes"
# print_attr(t.times.children[0].children[0])

# print "\n messing with t3\n"
# t3.times.stamps['wowza'] = 1.2

# print "\n t2.times.children[0]"
# print_attr(t2.times.children[0])

# print "\n t2.times.children[0]"


print "\nabsorbing t3 into t2 at '2first'\n"
t2.absorb(t3, '2first')

print "\n t2.times attributes\n"
print_attr(t2.times)

print "\nmerging t2 into t\n"
t.merge(t2)

print "\n t_t.times attributes\n"
print_attr(t.times)

print "\n t2.times.children[0] attributes\n"
print_attr(t2.times.children[0])

print "\n t.times.children[0] attributes\n"
print_attr(t.times.children[0])

t.times.print_report()

t.times.print_structure()

# print "\n\nabsorbing t2 into t at 'first'\n\n"
# t.absorb(t2, 'first')

# print "\n t.times atrributes"
# print_attr(t.times)

# print "\n t2.times atrributes"
# print_attr(t2.times)

# print "\n t.times.children[0] attributes"
# print_attr(t.times.children[0])

# print "\n\naltering t2 and child differently"
# t2.times.stamps['new'] = 7.6
# t.times.children[0].stamps['whoops'] = 2.3

# print "\n t2.times atrributes"
# print_attr(t2.times)

# print "\n t.times.children[0] attributes"
# print_attr(t.times.children[0])

# print "\n\n absorbing t3 into t2 at '2first'\n\n"
# t2.absorb(t3, '2first')

# print "\n t2.times atrributes"
# print_attr(t2.times)

# print "\n t.times.children[0] attributes"
# print_attr(t.times.children[0])



# print "t.times: ", t.times, "\n"
# print "t.times.total: ", t.times.total, "\n"
# print "t.times.self_: ", t.times.self_, "\n"
# print "t.times.self_agg: ", t.times.self_agg, "\n"
# print "t.times.stamps_sum: ", t.times.stamps_sum, "\n"
# print "t.times.stamps: ", t.times.stamps, "\n"
# print "t.times.stamps_itrs: ", t.times.stamps_itrs, "\n"
# print "absorbing t2 into t at 'first'\n\n"
# t.absorb(t2, 'first')
# print "t.times.self_agg: ", t.times.self_agg, "\n"
# print "t2.times.self_agg: ", t2.times.self_agg, "\n"
# print "t.times.children: ", t.times.children, "t2.times: ", t2.times, "\n"
# print "t.times.children['first'][0]: ", t.times.children['first'][0], "\n"
# print "t.times.children['first'][0].stamps: ", t.times.children['first'][0].stamps, "\n"
# print "absorbing t3 into t2 at '2first' \n\n"
# t2.absorb(t3, '2first')
# print "dir(t.times)", dir(t.times)
# print "t.times.__dict__", t.times.__dict__
# for k, v in t.times.__dict__.iteritems():
#     print k, "\t\t", v


