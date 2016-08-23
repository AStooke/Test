#
# Timer functions.
#
from timeit import default_timer as timer
import globalholder as g
import timesfuncs

UNASSIGNED = 'UNASSIGNED'


def start():
    if g.tf.stamps:
        raise RuntimeError("Already have stamps, can't start again.")
    if g.rf.children_awaiting:
        raise RuntimeError("Already have lower level timers, can't start again.")
    t = timer()
    g.tf.start_t = t
    g.tf.last_t = t


def stamp(name):
    t = timer()
    if name in g.rf.stamps:
        raise ValueError("Duplicate name: {}".format(repr(name)))
    if g.rf.children_awaiting:
        timesfuncs.assign_children(name)
    g.rf.stamps[name] = t - g.tf.last_t
    g.tf.last_t = t
    return t


def stop(name=None):
    if name is not None:
        t = stamp(name)
    else:
        t = timer()
        if g.rf.children_awaiting:
            timesfuncs.l_assign_children(UNASSIGNED)
    g.rf.total = t - g.tf.start_t
    timesfuncs.dump_times()


def l_stamp(name):
    t = timer()
    # Still need to check whether current timer is in loop.
    if name not in g.lf.stamps:
        if name in g.rf.stamps:
            raise ValueError("Duplicate name: {}".format(repr(name)))
        g.lf.stamps.append(name)
        g.lf.itr_stamp_used[name] = False
        g.tf.stamps[name] = 0.
        g.tf.stamps_itrs[name] = []
    if g.lf.itr_stamp_used[name]:
        raise RuntimeError("Loop stamp name twice in one itr: {}".format(repr(name)))
    if g.rf.children_awaiting:
        timesfuncs.l_assign_children(name)
    g.lf.itr_stamp_used[name] = True
    elapsed = t - g.tf.last_t
    g.rf.stamps[name] += elapsed
    g.rf.stamps_itrs[name].append(elapsed)
    g.tf.last_t = t
    return t


def _enter_loop(name=None):
    g.tf.last_t = timer()
    if g.rf.children_awaiting:
        timesfuncs.l_assign_children(UNASSIGNED)
    g.create_next_loop(name)
    if name is not None:
        if name in g.rf.stamps:
            raise ValueError("Duplicate name: {}".format(repr(name)))
        g.rf.stamps[name] = 0.
        g.rf.stamps_itrs[name] = []
        g.create_next_timer(name)


def _loop_start():
    for k in g.lf.itr_stamp_used:
        g.lf.itr_stamp_used[k] = False


def _loop_end():
    t = timer()
    g.tf.last_t = t
    if g.rf.children_awaiting:
        timesfuncs.l_assign_children(UNASSIGNED)
    if g.lf.name is not None:
        g.focus_backward_timer()
        g.rf.stamps[g.lf.name] += t - g.tf.last_t
        g.rf.last_t = t
        g.focus_forward_timer()


def _exit_loop():
    if g.lf.name is not None:
        # Then, currently in a timer made just for this loop, stop it.
        g.tf.stop()
        g.remove_last_timer()
    g.remove_last_loop()


def timed_for(loop_iterable, name=None):
    _enter_loop(name)
    for i in loop_iterable:
        _loop_start()
        yield i
        _loop_end()
    _exit_loop()


def timed_while(name=None):
    _enter_loop(name)
    i = -1  # throw in a loop counter for free!
    while g.lf.while_condition:
        _loop_start()
        i += 1
        yield i
        _loop_end()
    _exit_loop()


def set_while_false():
    g.lf.while_condition = False


def set_while_true():
    g.lf.while_condition = True


def break_for():
    raise NotImplementedError


def break_while():
    # Might be same as break_for()
    raise NotImplementedError


def continue_for():
    # Might not need anything here.
    raise NotImplementedError


def continue_while():
    # Might not need anything here.
    raise NotImplementedError
