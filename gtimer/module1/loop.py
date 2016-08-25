
"""
Almost everything to do with loops (some visible to from user).
"""

from timeit import default_timer as timer
import data_glob as g
import times_glob
import timer_mgmt


class Loop(object):
    """Hold info for name checking and assigning."""
    def __init__(self, name=None):
        self.name = name
        self.stamps = list()
        self.itr_stamp_used = dict()
        self.loop = None


class TimedLoop(object):
    """Base class, not to be used."""
    def __init__(self, name=None):
        self.name = name
        self.started = False
        self.exited = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if g.loop_broken:
            g.loop_broken = False
        elif not self.exited:
            if self.started:
                loop_end()
            exit_loop()


#
# Exposed to user.
#


class timed_for(TimedLoop):
    """ Exposed to user."""
    def __init__(self, iterable, name=None):
        self.iterable = iterable
        super(timed_for, self).__init__(name)

    def __iter__(self):
        enter_loop(self.name)
        for i in self.iterable:
            loop_start()
            self.started = True
            yield i
            loop_end()
            self.started = False
        exit_loop()
        self.exited = True


class timed_while(TimedLoop):
    """ Exposed to user."""
    def __init__(self, name=None):
        self.first = True
        super(timed_while, self).__init__(name)

    def next(self):
        if self.first:
            enter_loop()
            self.first = False
            self.started = True
        else:
            loop_end()
        loop_start()


def break_for():
    loop_end()
    exit_loop()
    g.loop_broken = True


#
# Hidden from user.
#


def enter_loop(name=None):
    print "in enter loop"
    t = timer()
    g.tf.last_t = t
    if g.tf.stopped:
        raise RuntimeError("Timer already stopped.")
    if g.tf.paused:
        raise RuntimeError("Timer paused.")
    if g.rf.children_awaiting:
        times_glob.l_assign_children(g.UNASGN)
    g.loop_broken = False
    if name is not None:  # Entering a named loop.
        if g.tf.loop_depth < 1 or name not in g.lf.stamps:
            if name in g.rf.stamps:
                raise ValueError("Duplicate name given to loop: {}".format(repr(name)))
            g.rf.stamps[name] = 0.
            g.rf.stamps_itrs[name] = []
        if g.tf.loop_depth > 0 and name not in g.lf.stamps:
            g.lf.stamps.append(name)
        t = timer()
        g.tf.self_t += t - g.tf.last_t
        g.entering_named_loop = True  # make False immediately after open_next_timer()!
        timer_mgmt.open_next_timer(name)
        g.entering_named_loop = False
    else:  # Entering an anonymous loop.
        g.tf.loop_depth += 1
    g.create_next_loop(name)
    g.tf.self_t += timer() - t


def loop_start():
    print "in loop start"
    if g.tf.stopped:
        raise RuntimeError("Timer already stopped.")
    if g.tf.paused:
        raise RuntimeError("Timer paused.")
    for k in g.lf.itr_stamp_used:
        g.lf.itr_stamp_used[k] = False


def loop_end():
    print "in loop end"
    t = timer()
    if g.tf.stopped:
        raise RuntimeError("Timer already stopped.")
    if g.tf.paused:
        raise RuntimeError("Timer paused.")
    g.tf.last_t = t
    if g.rf.children_awaiting:
        times_glob.l_assign_children(g.UNASGN)
    if g.lf.name is not None:
        # Reach back and stamp in the parent timer.
        g.focus_backward_timer()
        elapsed = t - g.tf.last_t
        g.rf.stamps[g.lf.name] += elapsed
        g.rf.stamps_itrs[g.lf.name].append(elapsed)
        g.tf.last_t = t
        g.focus_forward_timer()
    g.tf.self_t += timer() - t


def exit_loop():
    print "in exit loop"
    t = timer()
    if g.tf.stopped:
        raise RuntimeError("Timer already stopped.")
    if g.tf.paused:
        raise RuntimeError("Timer paused.")
    if g.lf.name is not None:
        timer_mgmt.close_last_timer()
        if g.rf.children_awaiting:
            times_glob.l_assign_children(g.lf.name)
    else:
        g.tf.loop_depth -= 1
        if g.rf.children_awaiting:
            times_glob.l_assign_children(g.UNASGN)
    g.remove_last_loop()
    g.tf.self_t += timer() - t
