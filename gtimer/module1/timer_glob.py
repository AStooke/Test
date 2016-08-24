
"""
Timer functions acting on global variables.
"""

from timeit import default_timer as timer
import data_glob as g
import times_glob
import timer_mgmt

UNASSIGNED = 'UNASSIGNED'

#
# Functions to expose to the user.
#


def start():
    if g.rf.stamps:
        raise RuntimeError("Already have stamps, can't start again.")
    if g.rf.children_awaiting:
        raise RuntimeError("Already have lower level timers, can't start again.")
    if g.tf.stopped:
        raise RuntimeError("Timer already stopped (must close and open new).")
    t = timer()
    g.tf.start_t = t
    g.tf.last_t = t


def stamp(name, unique=True):
    t = timer()
    elapsed = t - g.tf.last_t
    if g.tf.stopped:
        raise RuntimeError("Timer already stopped.")
    if g.tf.in_loop:
        if unique:
            _unique_loop_stamp(name, elapsed)
        else:
            _nonunique_loop_stamp(name, elapsed)
    else:
        if name not in g.rf.stamps:
            g.rf.stamps[name] = elapsed
        elif unique:
            raise ValueError("Duplicate stamp name: {}".format(repr(name)))
        else:
            g.rf.stamps[name] += elapsed
    if g.rf.children_awaiting:
        times_glob.l_assign_children(name)
    g.tf.last_t = t
    return t


def l_stamp(name, unique=True):
    t = timer()
    elapsed = t - g.tf.last_t
    if g.tf.stopped:
        raise RuntimeError("Timer already stopped.")
    if unique:
        if not g.tf.in_loop:
            raise RuntimeError("Should be in a timed loop to use unique l_stamp.")
        _unique_loop_stamp(name, elapsed)
    else:
        _nonunique_loop_stamp(name, elapsed)
    if g.rf.children_awaiting:
        times_glob.l_assign_children(name)
    g.tf.last_t = t
    return t


def stop(name=None, unique=True):
    if g.tf.stopped:
        raise RuntimeError("Timer already stopped.")
    if name is not None:
        t = stamp(name, unique)
    else:
        t = timer()
        if g.rf.children_awaiting:
            times_glob.l_assign_children(UNASSIGNED)
    g.rf.total = t - g.tf.start_t
    times_glob.dump_times()


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


def b_stamp(*args, **kwargs):
    if g.tf.stopped:
        raise RuntimeError("Timer already stopped.")
    t = timer()
    g.tf.last_t = t
    return t


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


#
# Private helper functions.
#


def _unique_loop_stamp(name, elapsed):
    if name not in g.lf.stamps:
        if name in g.rf.stamps:
            raise ValueError("Duplicate name: {} (might be anonymous inner loop)".format(repr(name)))
        g.lf.stamps.append(name)
        g.lf.itr_stamp_used[name] = False
        g.rf.stamps[name] = 0.
        g.rf.stamps_itrs[name] = []
    if g.lf.itr_stamp_used[name]:
        raise RuntimeError("Loop stamp name twice in one itr: {}".format(repr(name)))
    g.lf.itr_stamp_used[name] = True
    g.rf.stamps[name] += elapsed
    g.rf.stamps_itrs[name].append(elapsed)


def _nonunique_loop_stamp(name, elapsed):
    if name not in g.rf.stamps:
        g.rf.stamps[name] = elapsed
        g.rf.stamps_itrs[name] = [elapsed]
    else:
        g.rf.stmaps[name] += elapsed
        g.rf.stamps_itrs[name].append(elapsed)


def _enter_loop(name=None):
    g.tf.last_t = timer()
    if g.tf.stopped:
        raise RuntimeError("Timer already stopped.")
    if g.rf.children_awaiting:
        times_glob.l_assign_children(UNASSIGNED)
    if name is not None:  # entering a named loop
        if not g.tf.in_loop or name not in g.lf.stamps:
            if name in g.rf.stamps:
                raise ValueError("Duplicate name given to loop: {}".format(repr(name)))
            g.rf.stamps[name] = 0.
            g.rf.stamps_itrs[name] = []
        if g.tf.in_loop and name not in g.lf.stamps:
            g.lf.stamps.append(name)
        timer_mgmt.open_new_timer(name, in_loop=True)  # make a blank timer to be only inside the loop
    else:  # entering an anonymous loop
        g.tf.in_loop = True
        g.tf.loop_depth += 1
    g.create_next_loop(name)  # no matter what, make the next loop


def _loop_start():
    if g.tf.stopped:
        raise RuntimeError("Timer already stopped.")
    for k in g.lf.itr_stamp_used:
        g.lf.itr_stamp_used[k] = False


def _loop_end():
    t = timer()
    if g.tf.stopped:
        raise RuntimeError("Timer already stopped.")
    g.tf.last_t = t
    if g.rf.children_awaiting:
        times_glob.l_assign_children(UNASSIGNED)
    if g.lf.name is not None:
        # Reach back and stamp in the parent timer.
        g.focus_backward_timer()
        elapsed = t - g.tf.last_t
        g.rf.stamps[g.lf.name] += elapsed
        g.rf.stamps_itrs[g.lf.name].append(elapsed)
        g.tf.last_t = t
        g.focus_forward_timer()


def _exit_loop():
    if g.tf.stopped:
        raise RuntimeError("Timer already stopped.")
    if g.lf.name is not None:
        # Then, currently in a timer made just for this loop, stop it.
        timer_mgmt.close_last_timer()
        # The inner times (which was just removed) should be awaiting
        # assignment.  Might want to make this happend in _loop_end, so
        # it stays up to date, although waiting until here reduces the
        # amount of data movements.  On the other hand, this is the one
        # case where the pos_in_parent of the child timer is known ahead
        # of time, so maybe just short circuit the whole child_awaiting
        # procedure...although this seems to work for now.
        # 
        # Or maybe it's not working...deeply nested loops losing 
        # iterations currently...
        #
        # OK fixed that.  And now there is never a transfer in the 
        # dump to child_awaiting, there's always just the new
        # child which is the same object.  So there's no transfer 
        # there, just merging every time after the first loop exit,
        # during the children assignment.  So I guess that's as 
        # cheap as it gets!!  Just like having one tmp with one
        # master to dump to....since still can't know where the
        # parent is going to end up (all parents being named loops
        # is very special case).
        if g.rf.children_awaiting:
            times_glob.l_assign_children(g.lf.name)
    else:
        g.tf.loop_depth -= 1
        g.tf.in_loop = (g.tf.loop_depth > 0)
        if g.rf.children_awaiting:
            times_glob.l_assign_children(UNASSIGNED)
    g.remove_last_loop()
