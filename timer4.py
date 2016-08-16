from timeit import default_timer as timer
import copy

"""
Timer object for conveniently, quickly sprinkling timers into code to measure
whole programs or sections of interest.  Includes support for loops (for &
while) and branching within loops.

Times object for holding, organizing, and displaying timing measurements.

"""

#
# Data structure for holding the final timing results.
#


class Times(object):

    def __init__(self, name=''):
        self.name = name
        self.stamps = dict()
        self.stamps_itrs = dict()
        self.total = 0.
        self.stamps_sum = 0.
        self.self_ = 0.
        self.self_agg = 0.
        self.parent = None
        self.pos_in_parent = None
        self.children = list()
        self.stopped = False
        self.stamps_order = list()

    def __deepcopy__(self, memo):
        # Only reason for this method is to handle parent attribute properly.
        new = type(self)()
        new.__dict__.update(self.__dict__)
        new.stamps = copy.deepcopy(self.stamps, memo)
        new.stamps_itrs = copy.deepcopy(self.stamps_itrs, memo)
        new.children = copy.deepcopy(self.children, memo)
        # Avoid deepcopy of parent, and update parent attribute.
        for child in new.children:
            child.parent = self
        return new

    def clear(self):
        name = self.name
        self.__init__(name=name)

    #
    # Methods for linking / combining results from separate timers.
    #

    def absorb(self, child_times, position_name, self_agg_up=True):
        if not child_times.stopped:
            raise RuntimeError("Cannot absorb running times object, child must be stopped.")
        if position_name not in self.stamps:
            raise ValueError("Position name must be an existing stamp in parent.")
        is_existing_child = False
        for child in self.children:
            if position_name == child.pos_in_parent:
                if child_times.name == child.name:
                    is_existing_child = True
                    self._absorb_existing(child, child_times)
        if not is_existing_child:
            child_copy = copy.deepcopy(child_times)
            child_copy.parent = self
            child_copy.pos_in_parent = position_name
            self.children.append(child_copy)
        if self_agg_up:
            self._absorb_self_agg(child_times.self_agg)

    def _absorb_existing(self, old, new):
        self._merge_struct_dicts(old, new, 'stamps')
        self._merge_struct_dicts(old, new, 'stamps_itrs')
        old.total += new.total
        old.self_ += new.self_
        old.self_agg += new.self_agg
        for new_child in new.children:
            old.absorb(new_child, new_child.pos_in_parent, self_agg_up=False)

    def _merge_struct_dicts(self, old, new, attr):
        old_dict = getattr(old, attr)
        new_dict = getattr(new, attr)
        for k, v in new_dict.iteritems():
            if k in old_dict:
                old_dict[k] += v
            else:
                old_dict[k] = v

    def _absorb_self_agg(self, agg_time):
        self.self_agg += agg_time
        if self.parent is not None:
            self.parent._absorb_self_agg(agg_time)

    def merge(self, partner_times, copy_self=True):
        if not isinstance(partner_times, Times):
            raise TypeError("Valid Times object not recognized for merge.")
        if not partner_times.stopped:
            raise RuntimeError("Cannot merge running times object, partner must be stopped.")
        if copy_self:
            new = copy.deepcopy(self)
        else:
            new = self
        new.total += partner_times.total
        new.stamps_sum += partner_times.stamps_sum
        new.self_ += partner_times.self_
        new.self_agg += partner_times.self_agg
        for k, v in partner_times.stamps.iteritems():
            if k not in new.stamps:
                new.stamps[k] = v
            else:
                raise ValueError("Cannot merge stamps by the same name.")
        new.stamps_order += partner_times.stamps_order
        for k, v in partner_times.stamps_itrs.iteritems():
            new.stamps_itrs[k] = v
        new.children += copy.deepcopy(partner_times.children)
        for child in new.children:
            child.parent = new
        if copy_self:
            return new

    # def _set_parents(self):
    #     for child in self.children:
    #         child.parent = self
    #         child._set_parents()

    #
    # Reporting methods.
    #

    def report(self, include_itrs=True, include_self=True):
        if not self.stopped:
            raise RuntimeError("Cannot report an active Times structure, must be stopped.")
        fmt_num, fmt_gen = self._header_formats()
        rep = "\n---Timer Report---"
        if self.name:
            rep += fmt_gen.format('Timer:', repr(self.name))
        rep += fmt_num.format('Total:', self.total)
        rep += fmt_num.format('Stamps Sum:', self.stamps_sum)
        if include_self:
            rep += fmt_num.format('Self:', self.self_)
            rep += fmt_num.format('Self Agg.:', self.self_agg)
        rep += "\n\nStamps\n------"
        rep += self._report_stamps()
        if include_itrs:
            rep_itrs = ''
            rep_itrs += self._report_itrs()
            if rep_itrs:
                rep += "\n\nLoop Iterations\n---------------"
                rep += rep_itrs
        rep += "\n---End Report---\n"
        return rep

    def _report_stamps(self, indent=0, prec=3):
        s_rep = ''
        fmt = "\n{0}{{:<{1}}}\t{0}{{:.{2}g}}".format(' ' * indent, 16 - indent, prec)
        for stamp in self.stamps_order:
            s_rep += fmt.format(stamp, self.stamps[stamp])
            for child in self.children:
                if child.pos_in_parent == stamp:
                    s_rep += child._report_stamps(indent=indent + 2)
        return s_rep

    def _report_itrs(self):
        rep_itrs = ''
        fmt_num, fmt_gen = self._header_formats()
        if self.name:
            rep_itrs += fmt_gen.format('Timer:', repr(self.name))
        if self.parent is not None:
            rep_itrs += fmt_gen.format('Parent:', repr(self.parent.name))
            rep_itrs += fmt_gen.format('Pos in Parent:', repr(self.pos_in_parent))
        rep_itrs += "\n\nIter."
        stamps_itrs_order = []
        is_key_active = []
        for k in self.stamps_order:
            if k in self.stamps_itrs:
                stamps_itrs_order += [k]
                is_key_active += [True]
        print is_key_active
        for k in stamps_itrs_order:
            rep_itrs += "\t{:<12}".format(k)
        rep_itrs += "\n-----"
        for k in stamps_itrs_order:
            rep_itrs += "\t------\t"
        itr = 0
        while any(is_key_active):
            next_line = '\n{:<5}'.format(itr)
            for i, k in enumerate(stamps_itrs_order):
                if is_key_active[i]:
                    try:
                        next_line += "\t{:.3g}\t".format(self.stamps_itrs[k][itr])
                    except IndexError:
                        next_line += "\t\t"
                        is_key_active[i] = False
                else:
                    next_line += "\t\t"
            if any(is_key_active):
                rep_itrs += next_line
            itr += 1
        rep_itrs += "\n"
        for child in self.children:
            rep_itrs += child._report_itrs()
        return rep_itrs

    def _header_formats(self, width=12, prec=5):
        fmt_name = "\n{{:<{}}}\t".format(width)
        fmt_num = fmt_name + "{{:.{}g}}".format(prec)
        fmt_gen = fmt_name + "{}"
        return fmt_num, fmt_gen

    def print_report(self, include_self=True):
        print self.report(include_self=include_self)

    def write_structure(self):
        struct_str = '\n---Times Object Tree Structure---\n'
        struct_str += self._write_structure()
        struct_str += "\n\n"
        return struct_str

    def _write_structure(self, indent=0):
        if self.name:
            name_str = repr(self.name)
        else:
            name_str = '[Unnamed]'
        if self.parent:
            struct_str = "\n{}{} ({})".format(' ' * indent, name_str, repr(self.pos_in_parent))
        else:
            struct_str = "\n{}{}".format(' ' * indent, name_str)
        for child in self.children:
            struct_str += child._write_structure(indent=indent + 4)
        return struct_str

    def print_structure(self):
        print self.write_structure()


#
# Main class for recording timing, either directly or as a context manager.
#


class Timer(object):

    _inactive_error = "Can't use stopped timer (*maybe* can clear() to reset)."
    _no_loop_error = "Must be in timed loop to use loop methods."

    def __init__(self, name='', save_itrs=True, new_times=True):
        self.name = name
        self.save_itrs = save_itrs
        if new_times:
            self.times = Times(self.name)
        self.while_condition = True
        self._in_loop = False
        self._active = True
        self._stamp_names = []
        self._itr_stamp_used = dict()
        self._start = timer()
        self._last = self._start

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._stop()

    #
    # Methods operating on the Times data structure.
    #

    def report(self):
        if self._active:
            raise RuntimeError("Can't report an active timer, must stop it first.")
        return self.times.report()

    def print_report(self):
        print self.report()

    def clear(self):
        name = self.name
        save_itrs = self.save_itrs
        self.__init__(name=name, save_itrs=save_itrs, new_times=False)
        self.times.clear()

    def absorb(self, timing_obj, position_name):
        t = timer()
        if isinstance(timing_obj, Timer):
            self.times.absorb(timing_obj.times, position_name)
        elif isinstance(timing_obj, Times):
            self.times.aborb(timing_obj, position_name)
        if self._active:
            self.times.self_ += timer() - t

    def merge(self, timing_obj):
        t = timer()
        if isinstance(timing_obj, Timer):
            partner_times = timing_obj.times
        elif isinstance(timing_obj, Times):
            partner_times = timing_obj
        else:
            raise TypeError("Valid Timer or Times object not recognized for merge.")
        self.times.merge(partner_times, copy_self=False)
        if self._active:
            self.times.self_ += timer() - t

    #
    # Timing methods.
    #

    def _check_duplicate(self, name):
        if name in self._stamp_names or (name in self._itr_stamp_used and self._itr_stamp_used[name]):
            w = "Duplicate stamp name used: {}\n".format(repr(name))
            raise ValueError(w)

    def _stop(self):
        t = timer()
        if not self._active:
            raise RuntimeError("Timer already stopped.")
        if self._in_loop:
            raise RuntimeError("Cannot stop timer without exiting loop.")
        self.times.total += t - self._start - self.times.self_
        self.times.self_agg += self.times.self_
        for k, v in self.times.stamps.iteritems():
            self.times.stamps_sum += v
        self._active = False
        self.times.stamps_order = self._stamp_names
        self.times.stopped = True

    def stamp(self, name):
        """ Assigns the time since the previous stamp to this times key. """
        t = timer()
        if not self._active:
            raise RuntimeError(Timer._inactive_error)
        self._check_duplicate(name)
        self._stamp_names.append(name)
        self.times.stamps[name] = t - self._last
        self._last = timer()
        self.times.self_ += self._last - t
        return t

    def stop(self):
        """ If used stand-alone rather than in a 'with' statement. """
        self._stop()

    #
    # Loop methods.
    #

    def l_stamp(self, name):
        """ Assigns the time since the previous stamp to this times key. """
        t = timer()
        if not self._active:
            raise RuntimeError(Timer._inactive_error)
        if not self._in_loop:
            raise RuntimeError(Timer._no_loop_error)
        if name not in self._current_l_stamps:
            raise ValueError("Loop stamp name not registered at loop entrance.")
        self._check_duplicate(name)
        self._itr_stamp_used[name] = True
        elapsed = t - self._last
        if self.save_itrs:
            self.times.stamps_itrs[name].append(elapsed)
        self.times.stamps[name] += elapsed
        self._last = timer()
        self.times.self_ += self._last - t
        return t

    def _enter_loop(self, l_stamps_list):
        # if not self._active:
        #     raise RuntimeError(Timer._inactive_error)
        # if self._in_loop:
        #     raise RuntimeError("Timer does not support nested loops.")
        t = timer()
        self._in_loop = True
        self._itr_stamp_used.clear()
        for name in l_stamps_list:
            self._itr_stamp_used[name] = False
            self.times.stamps[name] = 0.
            if self.save_itrs:
                self.times.stamps_itrs[name] = []
        self._current_l_stamps = l_stamps_list
        self.times.self_ += timer() - t
        # self._itr_ended = True

    def _loop_start(self):
        t = timer()
        # if not self._active:
        #     raise RuntimeError(Timer._inactive_error)
        # if not self._in_loop:
        #     raise RuntimeError(Timer._no_loop_error)
        # if not self._itr_ended:
        #     raise RuntimeError("Must loop_end() before returning to start.")
        # self._itr_ended = False
        for k in self._itr_stamp_used:
            self._itr_stamp_used[k] = False
        self._last = timer()
        self.times.self_ += self._last - t

    def _loop_end(self):
        t = timer()
        # if not self._in_loop:
        #     raise RuntimeError(Timer._no_loop_error)
        if self.save_itrs:
            for k, v in self._itr_stamp_used.iteritems():
                if not v:
                    self.times.stamps_itrs[k].append(0.)
        for k in self._itr_stamp_used:
            self._itr_stamp_used[k] = False
        # self._itr_ended = True
        self.times.self_ += timer() - t

    def _exit_loop(self):
        t = timer()
        # if not self._in_loop:
        #     raise RuntimeError(Timer._no_loop_error)
        # if not self._itr_ended:
        #     raise RuntimeError("Timer loop not ended before exit_loop().")
        self._in_loop = False
        self._stamp_names += self._current_l_stamps
        self._current_l_stamps[:] = []
        self.times.self_ += timer() - t

    def timed_for(self, loop_iterable, l_stamps_list):
        if not self._active:
            raise RuntimeError(Timer._inactive_error)
        self._enter_loop(l_stamps_list)
        for i in loop_iterable:
            self._loop_start()
            yield i
            self._loop_end()
        self._exit_loop()

    def timed_while(self, l_stamps_list):
        if not self._active:
            raise RuntimeError(Timer._inactive_error)
        self._enter_loop(l_stamps_list)
        while self.while_condition:
            self._loop_start()
            yield None
            self._loop_end()
        self._exit_loop()
        self.while_condition = True
