from timeit import default_timer as timer

"""
Allows one level of loop, but no detail into inner loops. Branching within the
loop is supported (doesn't need to hit same stamps every iteration).
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
                             self=0.,
                             self_agg=0.,
                             parent=None,
                             children=dict(),
                             )

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
        existing_child = False
        existing_parent = False
        if position_name in self.children:
            existing_parent = True
            for child_old in self.children[position_name]:
                if child_old.name == child_times.name:
                    existing_child = True
                    break
        if not existing_parent:
            self.children[position_name] = []
        if not existing_child:
            child_times.parent = self
            self.children[position_name] += [child_times]
        else:
            self._merge_struct_dicts(child_old, child_times, 'stamps')
            self._merge_struct_dicts(child_old, child_times, 'intervals')
            child_old.total += child_times.total
            child_old.self += child_times.self
            child_old.self_agg += child_times.self_agg
            for k, loop in child_times.loops.iteritems():
                if k in child_old.loops:
                    self._absorb_loop(child_old.loops[k], loop)
                else:
                    child_old.loops[k] = loop
        self._absorb_self_agg(child_times.self_agg)

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
        self._merge_struct_dicts(old_loop, new_loop, 'stamps')
        self._merge_struct_dicts(old_loop, new_loop, 'intervals')
        if hasattr(new_loop, 'total_itrs') and hasattr(old_loop, 'total_itrs'):
            old_loop.total_itrs += new_loop.total_itrs
        if hasattr(new_loop, 'stamps_itrs') and hasattr(old_loop, 'stamps_itrs'):
            old_loop.stamps_itrs += new_loop.stamps_itrs
        if hasattr(new_loop, 'intervals_itrs') and hasattr(old_loop, 'intervals_itrs'):
            old_loop.intervals_itrs += new_loop.intervals_itrs

    def _absorb_self_agg(self, agg_time):
        self.self_agg += agg_time
        if self.parent is not None:
            self.parent._absorb_self_agg(agg_time)






class LoopTimes(Times):

    def __init__(self, name=''):
        self.__dict__.update(name=name,
                             stamps=dict(),
                             intervals=dict(),
                             total=0.,
                             stamps_itrs=list(),
                             intervals_itrs=list(),
                             total_itrs=list(),
                             )


#
# Main class for recording timing, either directly or as a context manager.
#


class TimerChecked(object):

    _inactive_error = "Can't use stopped timer, must restart it."
    _no_loop_error = "Must be in timed loop to use loop methods."

    def __init__(self, name='', save_itrs=True):
        self.name = name
        self.save_itrs = save_itrs
        self._open_intervals = {}
        self._loop_open_intervals = {}
        self.times = Times(self.name)
        self._current_times = self.times
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
        self.times.self += timer() - t
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
        self.times.self += timer() - t
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
        self.times.self += timer() - t
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
        self.times.self += timer() - t
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
        self.times.self += timer() - self._loop_start

    def loop_end(self):
        t = timer()
        if not self._in_loop:
            raise RuntimeError(TimerChecked._no_loop_error)
        self._check_unclosed_l()
        elapsed = t - self._loop_start
        self._current_loop.total += elapsed
        if self.save_itrs:
            self._current_itr.total = elapsed
            self._current_loop.stamps_itrs.append(self._current_itr.stamps)
            self._current_loop.intervals_itrs.append(self._current_itr.intervals)
            self._current_loop.total_itrs.append(elapsed)
        self._itr_ended = True
        self.times.self += timer() - t

    def exit_loop(self):
        if not self._in_loop:
            raise RuntimeError(TimerChecked._no_loop_error)
        if not self._itr_ended:
            raise RuntimeError("Timer loop not ended before exit_loop().")
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

