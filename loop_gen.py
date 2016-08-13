def loop(seq):
    print 'preamble'
    for i in seq:
        print 'before'
        yield i
        print 'after'
    print 'epilogue'


def while_loop(condition_holder):
    print 'preamble'
    while condition_holder.satisfied:
        print 'before'
        yield None
        print 'after'
    print 'epilogue'


class ConditionHolder(object):

    def __init__(self):
        self.__dict__.update(satisfied=True)



some_var = ConditionHolder()
x = 1

for _ in while_loop(some_var):
    print 'Went Around'
    x += 1
    print x
    if x > 7:
        some_var.satisfied = False
    if x > 8:
        raise RuntimeError("x went past 8")



# for i in loop([i for i in range(3)]):
#     print i
#     print 'code'
