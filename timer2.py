from timeit import default_timer as timer

# NOT A FUNCTIONING CODE.

# Trying to handle arbitrarily nested loops.

#
# Data structures for holding the final timing results.
#


class Times(object):

    def __init__(self):
        self.__dict__.update(stamps=dict(), intervals=dict(), loops=dict())


class LoopTimes(Times):

    def __init__(self):
        super(LoopTimes, self).__init__()
        self.__dict__.update(stamps_itr=list(), intervals_itr=list(), parent=None)


class _LoopTimesTmp(LoopTimes):

    def __init__(self):
        super(_LoopTimesTmp, self).__init__()
        self.__dict__.update(stamps_cum=dict(), intervals_cum=dict())


#
# Main class for recording timing, either directly or as a context manager.
#


class Timer(object):

    # inactive_error = "Can't call methods or manage contexts using stopped timer."

    def __init__(self, exit_stamp_name=None, verbose=False, check=True):
        self.verbose = bool(verbose)
        self.check = bool(check)
        self._reserved = ["total", "loop"]
        self._exit_stamp_name = exit_stamp_name
        self._intervals = {}
        self.times = Times()
        self._stamp_names = []
        self._interval_names = []
        self._start = timer()
        self._last = self._start
        self._current_times = self.times
        # self._active = True
        self._in_loop = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._stop(self._exit_stamp_name)

    def _check_duplicate(self, name):
        if name in self._stamp_names:
            w = "Duplicate stamp name used: {}\n".format(repr(name))
            raise ValueError(w)

    def _check_reserved(self, name):
        if name in self._reserved:
            w = "Reserved stamp name {} not allowed.\n".format(repr(name))
            raise ValueError(w)

    def _check_unclosed(self):
        if self._intervals:
            w = "Unclosed interval(s) at timer object exit or loop end: "
            for k in self._intervals:
                w += "{}, ".format(repr(k))
            w = w[:-2] + "\n"
            raise RuntimeError(w)

    def _stop(self, name=None):
        if self._in_loop:
            raise RuntimeError("Cannot stop/exit timer without exiting loop.")
        else:
            self._end = timer()
            self.times['total'] = self._end - self._start
            if name is not None:
                if self.check:
                    self._check_duplicate(name)
                    self._check_reserved(name)
                self.times[name] = self._end - self._last
            if self.check:
                self._check_unclosed()
            if self.verbose:
                print "Elapsed time: %f s" % self.times['total']

    def stamp(self, name):
        """ Assigns the time since the previous stamp to this times key. """
        t = timer()
        if self.check:
            self._check_duplicate(name)
            self._check_reserved(name)
        self._current_times.stamps[name] = t - self._last
        self._last = t
        self._stamp_names.append(name)

    def interval(self, name):
        """ Starts a named interval or adds the time since last starting this
        inteveral to its times key. (Can measure disjoint sections.) """
        t = timer()
        if self.check:
            self._check_duplicate(name)
            self._check_reserved(name)
        if name in self._intervals:
            if name in self.times:
                self._current_times.intervals[name] += t - self._intervals[name]
            else:
                self._current_times.intervals[name] = t - self._intervals[name]
            del(self._intervals[name])
        else:
            self._intervals[name] = t

    def stop(self, stamp_name=None):
        """ If used stand-alone rather than in a 'with' statement. """
        self._stop(stamp_name)

    #
    # Loop methods.
    #

    def enter_loop(self, name):
        if name not in self._current_times.loops:  # First time entering this loop.
            temp_loop_times = _LoopTimesTmp()
            # FIX LATER: disallow same loop names (think inner loops with different outer loops).
        self._parent_times = self._current_times
        self._current_times = self._current_times.loops[name]
        self._loop_level += 1

    def loop_start(self):
        if self._iter_ended:
            self._loop_start[self._loop_level] = timer()
            self._last = self._loop_start
            self._iter_ended = False
            self._loop_stamp_names = []
        else:
            raise RuntimeError("Timer loop not ended before returning to start.")

    def loop_end(self, stamp_name=None):
        t = timer()
        self._current_times.stamps['loop'] = t - self._loop_start[self._loop_level]
        if stamp_name is not None:
            if self.check:
                self._check_duplicate(stamp_name)
                self._check_reserved(stamp_name)
            self._current_times.stamps[stamp_name] = t - self._last
            # self._stamp_names.append(stamp_name)
        if self.check:
            self._check_unclosed()
        self._current_times.stamps_itr.append(self._current_times.stamps)
        self._current_Tiems.intervals_iter.append(self._current_times.intervals)
        for k, t in self._current_times.stamps:
            if k in self._current_times.stamps_cum:
                self._current_times.stamps_cum[k] += t
            else:
                self._current_times.stamps_cum[k] = t
        for k, t in self._current_times.intervals:
            if k in self._current_times.intervals_cum:
                self._current_times.intervals_cum[k] += t
            else:
                self._current_times.intervals_cum[k] = t
        self._iter_ended = True

    def exit_loop(self):
        if self._iter_ended:
            self._loop_level -= 1
            self.times = self._times_temp
            self.loop_time_sums = dict()
            for loop_times in self.loop_times:
                for k, t in loop_times.iteritems():
                    if k in self.loop_time_sums:
                        self.loop_time_sums[k] += t
                    else:
                        self.loop_time_sums[k] = t
        else:
            raise RuntimeError("Timer loop not ended before calling exit_loop().")
