from timeit import default_timer as timer
import copy

"""
Allows one level of loop, but no detail into inner loops. Branching within the
loop is supported (doesn't need to hit same stamps every iteration).

Currently intended for each timer object to hold only one times object, but
this could change if it becomes helpful to hold more.

"""

#
# Data structures for holding the final timing results.
#


def Timer(name='', check=True, save_itrs=True):
    if check:
        return TimerChecked(name=name, save_itrs=save_itrs)
    else:
        return TimerUnchecked(name=name, save_itrs=save_itrs)


class Times(object):

    def __init__(self, name=''):
        self.__dict__.update(name=name,
                             stamps=dict(),
                             intervals=dict(),
                             loops=dict(),
                             total=0.,
                             self_=0.,
                             self_agg=0.,
                             parent=None,
                             children=dict(),
                             )

    def __deepcopy__(self, memo):
        new = type(self)()
        new.__dict__.update(self.__dict__)
        new.stamps = copy.deepcopy(self.stamps, memo)
        new.intervals = copy.deepcopy(self.intervals, memo)
        print "in my __deepcopy__"
        print self
        print new
        new.loops = copy.deepcopy(self.loops, memo)
        # Don't copy the (timer) objects inside children, leave as shallow copy.
        return new

    def clear(self):
        name = self.name
        self.__init__(name=name)

    def absorb(self, child_times, position_name):
        name_found = False
        possible_dicts = [self.stamps, self.intervals, self.loops]
        for k, time_obj in self.loops.iteritems():
            possible_dicts += [time_obj.stamps, time_obj.intervals]
        for d in possible_dicts:
            if position_name in d:
                name_found = True
                break
        if not name_found:
            raise ValueError("position_name must be an existing stamp, interval, loop, loop-stamp, loop-interval in parent")
        is_existing_child = False
        is_existing_parent = False
        if position_name in self.children:
            is_existing_parent = True
            for child_old in self.children[position_name]:
                if child_old.name == child_times.name:
                    is_existing_child = True
                    break
        if not is_existing_parent:
            self.children[position_name] = []
        if not is_existing_child:
            child_times.parent = self
            self.children[position_name] += [copy.deepcopy(child_times)]
        else:
            self._absorb_existing(child_old, child_times)
        self._absorb_self_agg(child_times.self_agg)

    def _absorb_existing(self, old, new):
        self._merge_struct_dicts(old, new, 'stamps')
        self._merge_struct_dicts(old, new, 'intervals')
        old.total += new.total
        old.self_ += new.self_
        old.self_agg += new.self_agg
        for k, loop in new.loops.iteritems():
            if k in old.loops:
                self._absorb_loop(old.loops[k], loop)
            else:
                old.loops[k] = loop

    def _merge_struct_dicts(self, old, new, attr):
        old_dict = getattr(old, attr)
        new_dict = getattr(new, attr)
        for k, v in new_dict.iteritems():
            if k in old_dict:
                old_dict[k] += v
            else:
                old_dict[k] = v

    def _absorb_loop(self, old_loop, new_loop):
        old_loop.total += new_loop.total
        old_loop.self_ += new_loop.self_
        self._merge_struct_dicts(old_loop, new_loop, 'stamps')
        self._merge_struct_dicts(old_loop, new_loop, 'intervals')
        if hasattr(new_loop, 'total_itrs') and hasattr(old_loop, 'total_itrs'):
            old_loop.total_itrs += new_loop.total_itrs
        for attr in ['stamps_itrs', 'intervals_itrs']:
            if hasattr(new_loop, attr) and hasattr(old_loop, attr):
                self._absorb_itrs(old_loop, new_loop, attr)

    def _absorb_itrs(self, old_loop, new_loop, attr):
        old = getattr(old_loop, attr)
        new = getattr(new_loop, attr)
        for k, l in new.iteritems():
            if k not in old:
                old[k] = [0.] * old_loop.n_itr
            old[k] += l
        for k, l in old.iteritems():
            if k not in new:
                l += [0.] * new_loop.n_itr

    def _absorb_self_agg(self, agg_time):
        self.self_agg += agg_time
        if self.parent is not None:
            self.parent._absorb_self_agg(agg_time)


class LoopTimes(object):

    def __init__(self, name=''):
        self.__dict__.update(name=name,
                             stamps=dict(),
                             intervals=dict(),
                             total=0.,
                             self_=0.,
                             stamps_itrs=dict(),
                             intervals_itrs=dict(),
                             total_itrs=list(),
                             n_itr=0
                             )

    def clear(self):
        name = self.name
        self.__init__(name=name)


#
# Main class for recording timing, either directly or as a context manager.
#


class TimerChecked(object):

    _inactive_error = "Can't use stopped timer, must restart it."
    _no_loop_error = "Must be in timed loop to use loop methods."

    def __init__(self, name='', save_itrs=True, new_times=True):
        self.name = name
        self.save_itrs = save_itrs
        self._open_intervals = {}
        self._loop_open_intervals = {}
        if new_times:
            self.times = Times(self.name)
        self.while_condition = True
        self._active = True
        self._stamp_names = []
        self._in_loop = False
        self._itr_stamp_names = []
        self._start = timer()
        self._last = self._start

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._stop()

    def clear(self):
        name = self.name
        save_itrs = self.save_itrs
        self.__init__(name=name, save_itrs=save_itrs, new_times=False)
        self.times.clear()

    def absorb(self, timer_obj):
        t = timer()
        self.times.absorb(timer_obj.times)
        self.times.self_ += timer() - t

    def restart(self):
        self._active = True
        self._start = timer()
        self._last = self._start

    def _check_duplicate(self, name):
        if name in self._stamp_names or name in self._itr_stamp_names:
            w = "Duplicate stamp name used: {}\n".format(repr(name))
            raise ValueError(w)

    def _check_unclosed(self):
        if self._open_intervals:
            w = "Unclosed interval(s) at timer object exit: "
            for k in self._open_intervals:
                w += "{}, ".format(repr(k))
            w = w[:-2] + "\n"
            raise RuntimeError(w)

    def _stop(self):
        t = timer()
        if not self._active:
            raise RuntimeError("Cannot stop a stopped timer.")
        if self._in_loop:
            raise RuntimeError("Cannot stop/exit timer without exiting loop.")
        self._check_unclosed()
        self.times.total = t - self._start
        self._active = False

    def stamp(self, name):
        """ Assigns the time since the previous stamp to this times key. """
        t = timer()
        if not self._active:
            raise RuntimeError(TimerChecked._inactive_error)
        self._check_duplicate(name)
        self._stamp_names.append(name)
        self.times.stamps[name] = t - self._last
        self._last = t
        self_ = timer() - t
        self.times.self_ += self_
        self.times.self_agg += self_
        return t

    def interval(self, name):
        """ Starts a named interval or adds the time since last starting this
        inteveral to its times key. (Can measure disjoint sections.) """
        t = timer()
        if not self._active:
            raise RuntimeError(TimerChecked._inactive_error)
        self._check_duplicate(name)
        if name in self._open_intervals:
            self.times.intervals[name] += t - self._open_intervals[name]
            del(self._open_intervals[name])
        else:
            self._open_intervals[name] = t
            if name not in self.times.intervals:
                self.times.intervals[name] = 0.
        self_ = timer() - t
        self.times.self_ += self_
        self.times.self_agg += self_
        return t

    def stop(self):
        """ If used stand-alone rather than in a 'with' statement. """
        self._stop()

    #
    # Loop methods.
    #

    def _check_unclosed_l(self):
        if self._loop_open_intervals:
            w = "Unclosed loop interval(s) at timer loop end: "
            for k in self._loop_open_intervals:
                w += "{}, ".format(repr(k))
            w = w[:-2] + "\n"
            raise RuntimeError(w)

    def l_stamp(self, name):
        """ Assigns the time since the previous stamp to this times key. """
        if not self._in_loop:
            raise RuntimeError(TimerChecked._no_loop_error)
        t = timer()
        self._check_duplicate(name)
        self._itr_stamp_names.append(name)
        if name not in self._loop_stamp_names:
            self._loop_stamp_names.append(name)
        elapsed = t - self._last
        if self.save_itrs:
            self._current_itr.stamps[name] = elapsed
        if name in self._current_loop.stamps:
            self._current_loop.stamps[name] += elapsed
        else:
            self._current_loop.stamps[name] = elapsed
        self._last = t
        self._current_loop.self_ += timer() - t
        return t

    def l_interval(self, name):
        """ Starts a named interval or adds the time since last starting this
        inteveral to its times key. (Can measure disjoint sections.) """
        t = timer()
        if not self._in_loop:
            raise RuntimeError(Timer._no_loop_error)
        self._check_duplicate(name)
        if name in self._loop_open_intervals:
            elapsed = t - self._loop_open_intervals[name]
            if self.save_itrs:
                self._current_itr.intervals[name] += elapsed
            self._current_loop.intervals[name] += elapsed
            del(self._loop_open_intervals[name])
        else:
            self._loop_open_intervals[name] = t
            if self.save_itrs:
                if name not in self._current_itr.intervals:
                    self._current_itr.intervals[name] = 0.
            if name not in self._current_loop.intervals:
                self._current_loop.intervals[name] = 0.
        self._current_loop.self_ += timer() - t
        return t

    def enter_loop(self, name):
        if not self._active:
            raise RuntimeError(TimerChecked._inactive_error)
        if self._in_loop:
            raise RuntimeError("Timer does not support nested loops.")
        self._in_loop = True
        if self.save_itrs:
            self.times.loops[name] = LoopTimes(name)
        else:
            self.times.loops[name] = Times(name)
        self._current_loop = self.times.loops[name]
        self._loop_stamp_names = []
        self._itr_ended = True

    def loop_start(self):
        self._loop_start = timer()
        if not self._active:
            raise RuntimeError(TimerChecked._inactive_error)
        if not self._in_loop:
            raise RuntimeError(TimerChecked._no_loop_error)
        if not self._itr_ended:
            raise RuntimeError("Must loop_end() before returning to start.")
        self._last = self._loop_start
        if self.save_itrs:
            self._current_itr = Times()
        self._itr_ended = False
        self._itr_stamp_names = []
        self._current_loop.self_ += timer() - self._loop_start

    def loop_end(self):
        t = timer()
        if not self._in_loop:
            raise RuntimeError(TimerChecked._no_loop_error)
        self._check_unclosed_l()
        elapsed = t - self._loop_start
        self._current_loop.total += elapsed
        if self.save_itrs:
            self._append_itrs(self._current_itr.stamps, self._current_loop.stamps_itrs)
            self._append_itrs(self._current_itr.intervals, self._current_loop.intervals_itrs)
            self._current_loop.total_itrs.append(elapsed)
            self._current_loop.n_itr += 1
        self._itr_ended = True
        self._current_loop.self_ += timer() - t

    def _append_itrs(self, new_itr, old):
        for k, t in new_itr.iteritems():
            if k not in old:
                old[k] = [0.] * self._current_loop.n_itr
            old[k].append(t)
        for k, l in old.iteritems():
            if k not in new_itr:
                l.append(0.)

    def exit_loop(self):
        if not self._in_loop:
            raise RuntimeError(TimerChecked._no_loop_error)
        if not self._itr_ended:
            raise RuntimeError("Timer loop not ended before exit_loop().")
        self.times.self_ += self._current_loop.self_
        self._in_loop = False
        self._current_loop = None
        self._current_itr = None
        self._stamp_names += self._loop_stamp_names
        self._loop_stamp_names = []

    def timed_for(self, name, loop_iterable):
        self.enter_loop(name)
        for i in loop_iterable:
            self.loop_start()
            yield i
            self.loop_end()
        self.exit_loop()

    def timed_while(self, name):
        self.enter_loop(name)
        while self.while_condition:
            self.loop_start()
            yield None
            self.loop_end()
        self.exit_loop()
        self.while_condition = True


class TimerUnchecked(object):

    def __init__(self):
        pass

