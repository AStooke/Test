from timeit import default_timer as timer


class Timer(object):

    inactive_error = "Can't call methods or manage contexts using stopped timer."

    def __init__(self, last_name=None, verbose=False, check=True):
        self.verbose = bool(verbose)
        self.check = bool(check)
        self._reserved = ["total"]
        self._last_name = last_name
        self._intervals = {}
        self.times = {}
        self._stamp_names = []
        self._interval_names = []
        self._start = timer()
        self._last = self._start
        self._active = True

    def __enter__(self):
        if self._active:
            return self
        else:
            raise TypeError(Timer.inactive_error)

    def __exit__(self, *args):
        if self._active:
            self._stop(self._last_name)
        else:
            raise TypeError(Timer.inactive_error)

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
            w = "Unclosed interval(s) at timer object exit: "
            for k in self._intervals:
                w += "{}, ".format(repr(k))
            w = w[:-2] + "\n"
            raise RuntimeError(w)

    def _stop(self, name=None):
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
        self._active = False

    def stamp(self, name):
        """ Assigns the time since the previous stamp to this times key. """
        if self._active:
            t = timer()
            if self.check:
                self._check_duplicate(name)
                self._check_reserved(name)
            self.times[name] = t - self._last
            self._last = t
            self._stamp_names.append(name)
        else:
            raise TypeError(Timer.inactive_error)

    def interval(self, name):
        """ Starts a named interval or adds the time since last starting this
        inteveral to its times key. (Can measure disjoint sections.) """
        if self._active:
            t = timer()
            if self.check:
                self._check_duplicate(name)
                self._check_reserved(name)
            if name in self._intervals:
                if name in self.times:
                    self.times[name] += t - self._intervals[name]
                else:
                    self.times[name] = t - self._intervals[name]
                del(self._intervals[name])
            else:
                self._intervals[name] = t
        else:
            raise TypeError(Timer.inactive_error)

    def stop(self, name=None):
        """ If used stand-alone rather than in a 'with' statement. """
        if self._active:
            self._stop(name)
        else:
            raise TypeError(Timer.inactive_error)
