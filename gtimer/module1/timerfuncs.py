#
# Timer functions.
#
from timeit import default_timer as timer
import globalholder as g


def stamp(name):
    t = timer()
    if name in g.tf.times.stamps:
        raise ValueError("Duplicate name: {}".format(repr(name)))
    g.tf.times.stamps[name] = t - g.tf.last_t
    g.tf.last_t = t
    return t


def stop(name=None):
    if name is not None:
        stamp(name)
    t = timer()
    g.tf.times.total = t - g.tf.start_t


def l_stamp(name):
    t = timer()
    if name not in g.lf.stamps:
        if name in g.tf.times.stamps:
            raise ValueError("Duplicate name: {}".format(repr(name)))
        g.lf.stamps.append(name)
        g.lf.itr_stamp_used[name] = False
        g.tf.stamps[name] = 0.
        g.tf.stamps_itrs[name] = []
    if g.lf.itr_stamp_used[name]:
        raise RuntimeError("Loop stamp name twice in one itr: {}".format(repr(name)))
    g.lf.itr_stamp_used[name] = True
    elapsed = t - g.tf.last_t
    g.tf.times.stamps[name] += elapsed
    g.tf.times.stamps_itrs[name].append(elapsed)
    g.tf.last_t = t
    return t


def _enter_loop(name=None):
    g.tf.last_t = timer()
    g.create_next_loop(name)
    if name is not None:
        if name in g.tf.stamps:
            raise ValueError("Duplicate name: {}".format(repr(name)))
        g.tf.times.stamps[name] = 0.
        g.tf.times.stamps_itrs[name] = []
        g.create_next_timer(name)


def _loop_start():
    for k in g.lf.itr_stamp_used:
        g.lf.itr_stamp_used[k] = False


def _loop_end():
    t = timer()
    g.tf.last_t = t
    if g.lf.name is not None:
        g.focus_backward_timer()
        g.tf.times.stamps[g.lf.name] += t - g.tf.last_t
        g.tf.last_t = t
        g.focus_forward_timer()


def _exit_loop():
    if g.lf.name is not None:
        g.tf.stop()
        # Need to get the data from this timer before removing it.
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
