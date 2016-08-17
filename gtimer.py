from timeit import default_timer as timer
import copy
from operator import attrgetter

"""
Timer object to sprinkle timers into code and easily add/remove/alter them to
measure whole programs or sections of interest.  Supports loops (both for &
while) including branching within loops.

Times object for holding, organizing, and displaying timing measurements.
Hierachical combinations of times objects provide drill-down capability for
measuring arbitrary sub-components.  Formatted reporting included.

Provides optional global timers held by the module so that timers from
separate files and subfunctions can be referenced without explicitly passing
through function returns statements.

"""

#
# Data structure for holding the final timing results.
#


class Times(object):

    _grabs_accum_keys = ['total', 'stamps_sum', 'self_', 'self_agg', 'calls',
                         'calls_agg', 'grabs_agg']

    def __init__(self, name=''):
        self._name = name
        self._stamps = dict()
        self._stamps_itrs = dict()
        self._total = 0.
        self._stamps_sum = 0.
        self._self = 0.
        self._self_agg = 0.
        self._calls = 0
        self._calls_agg = 0
        self._grabs_agg = 0
        self._parent = None
        self._pos_in_parent = None
        self._children = list()
        self._num_descendants = 0
        self._stopped = False
        self._stamps_ordered = list()

    name = property(attrgetter("_name"))
    stamps = property(attrgetter("_stamps"))
    stamps_itrs = property(attrgetter("_stamps_itrs"))
    total = property(attrgetter("_total"))
    stamps_sum = property(attrgetter("_stamps_sum"))
    self = property(attrgetter("_self"))
    self_agg = property(attrgetter("_self_agg"))
    calls = property(attrgetter("_calls"))
    calls_agg = property(attrgetter("_calls_agg"))
    grabs_agg = property(attrgetter("_grabs_agg"))
    parent = property(attrgetter("_parent"))
    children = property(attrgetter("_children"))
    num_descendants = property(attrgetter("_num_descendants"))
    stopped = property(attrgetter("_stopped"))
    stamps_ordered = property(attrgetter("_stamps_ordered"))

    def __deepcopy__(self, memo):
        # Only reason for this method is to handle parent attribute properly.
        new = type(self)()
        new.__dict__.update(self.__dict__)
        new._stamps = copy.deepcopy(self._stamps, memo)
        new._stamps_itrs = copy.deepcopy(self._stamps_itrs, memo)
        new._children = copy.deepcopy(self._children, memo)
        # Avoid deepcopy of parent, and update parent attribute.
        for child in new._children:
            child._parent = self
        return new

    def clear(self):
        name = self._name
        self.__init__(name=name)

    #
    # Methods for linking / combining results from separate timers.
    #

    def graft(self, child_times, position_name, aggregate_up=True):
        if not child_times._stopped:
            raise RuntimeError("Cannot graft running times object, child must be stopped.")
        if position_name not in self._stamps and self._stopped:
            raise ValueError("Position name must be an existing stamp in stopped parent.")
        is_existing_child = False
        for old_child in self._children:
            if position_name == old_child._pos_in_parent:
                if child_times._name == old_child._name:
                    is_existing_child = True
                    self._graft_existing(old_child, child_times)
        if not is_existing_child:
            child_copy = copy.deepcopy(child_times)
            child_copy._parent = self
            child_copy._pos_in_parent = position_name
            self._children.append(child_copy)
            self._num_descendants += child_copy._num_descendants + 1
        self._grabs_agg += child_times._num_descendants + 1
        if aggregate_up:
            self._aggregate_up(child_times)

    def _graft_existing(self, old_times, new_times):
        self._absorb_dict(old_times, new_times, '_stamps')
        self._absorb_dict(old_times, new_times, '_stamps_itrs')
        for k in self._grabs_accum_keys:
            old_times.__dict__[k] += new_times.__dict__[k]
        for new_child in new_times._children:
            old_times.graft(new_child, new_child._pos_in_parent, aggregate_up=False)

    def _absorb_dict(self, old_times, new_times, dict_name):
        old_dict = getattr(old_times, dict_name)
        new_dict = getattr(new_times, dict_name)
        for k, v in new_dict.iteritems():
            if k in old_dict:
                old_dict[k] += v
            else:
                old_dict[k] = v

    def _aggregate_up(self, new_times):
        self._self_agg += new_times._self_agg
        self._calls_agg += new_times._calls_agg
        self._grabs_agg += new_times._grabs_agg
        if self._parent is not None:
            self._parent._aggregate_up(new_times)

    def absorb(self, partner_times, copy_self=True):
        if not isinstance(partner_times, Times):
            raise TypeError("Valid Times object not recognized for absorb.")
        if not partner_times._stopped:
            raise RuntimeError("Cannot absorb running times object, partner must be stopped.")
        if copy_self:
            new = copy.deepcopy(self)
        else:
            new = self
        for k in self._grabs_accum_keys:
            new.__dict__[k] += partner_times.__dict__[k]
        for k, v in partner_times._stamps.iteritems():
            if k not in new._stamps:
                new._stamps[k] = v
            else:
                raise ValueError("Cannot absorb stamps by the same name.")
        new._stamps_ordered += partner_times._stamps_ordered
        for k, v in partner_times._stamps_itrs.iteritems():
            new._stamps_itrs[k] = v
        new._children += copy.deepcopy(partner_times._children)
        for child in new._children:
            child._parent = new
        new._num_descendants += partner_times._num_descendants
        self._grabs_agg += partner_times._num_descendants + 1
        if copy_self:
            return new

    #
    # Reporting methods.
    #

    def report(self, include_itrs=True, include_diagnostics=True):
        if not self._stopped:
            raise RuntimeError("Cannot report an active Times structure, must be stopped.")
        fmt_flt, fmt_gen = self._header_formats()
        rep = "\n---Timer Report---"
        if self._name:
            rep += fmt_gen.format('Timer:', repr(self._name))
        rep += fmt_flt.format('Total:', self._total)
        rep += fmt_flt.format('Stamps Sum:', self._stamps_sum)
        if include_diagnostics:
            rep += fmt_flt.format('Self:', self._self)
            rep += fmt_flt.format('Self Agg.:', self._self_agg)
            rep += fmt_gen.format('Calls:', self._calls)
            rep += fmt_gen.format('Calls Agg.:', self._calls_agg)
            rep += fmt_gen.format('Grabs Agg.:', self._grabs_agg)
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
        for stamp in self._stamps_ordered:
            s_rep += fmt.format(stamp, self._stamps[stamp])
            for child in self._children:
                if child._pos_in_parent == stamp:
                    s_rep += child._report_stamps(indent=indent + 2)
        return s_rep

    def _report_itrs(self):
        rep_itrs = ''
        if self._stamps_itrs:
            fmt_flt, fmt_gen = self._header_formats()
            if self._name:
                rep_itrs += fmt_gen.format('Timer:', repr(self._name))
            if self._parent is not None:
                rep_itrs += fmt_gen.format('Parent:', repr(self._parent._name))
                rep_itrs += fmt_gen.format('Pos in Parent:', repr(self._pos_in_parent))
            rep_itrs += "\n\nIter."
            stamps_itrs_order = []
            is_key_active = []
            for k in self._stamps_ordered:
                if k in self._stamps_itrs:
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
        for child in self._children:
            rep_itrs += child._report_itrs()
        return rep_itrs

    def _header_formats(self, width=12, prec=5):
        fmt_name = "\n{{:<{}}}\t".format(width)
        fmt_flt = fmt_name + "{{:.{}g}}".format(prec)
        fmt_gen = fmt_name + "{}"
        return fmt_flt, fmt_gen

    def print_report(self, include_diagnostics=True):
        print self.report(include_diagnostics=include_diagnostics)

    def write_structure(self):
        struct_str = '\n---Times Object Tree Structure---\n'
        struct_str += self._write_structure()
        struct_str += "\n\n"
        return struct_str

    def _write_structure(self, indent=0):
        if self._name:
            name_str = repr(self._name)
        else:
            name_str = '[Unnamed]'
        if self._parent:
            struct_str = "\n{}{} ({})".format(' ' * indent, name_str, repr(self._pos_in_parent))
        else:
            struct_str = "\n{}{}".format(' ' * indent, name_str)
        for child in self._children:
            struct_str += child._write_structure(indent=indent + 4)
        return struct_str

    def print_structure(self):
        print self.write_structure()


#
# Main class for recording timing, either directly or as a context manager.
#


class Timer(object):

    _inactive_error = "Can't use stopped timer (can clear() to reset)."
    _no_loop_error = "Must be in timed loop to use loop methods."

    def __init__(self, name='', save_itrs=True):
        self.save_itrs = save_itrs
        self.while_condition = True

        self._name = name
        if hasattr(self, "_times"):
            self._times.clear()
        else:
            self._times = Times(self._name)
        self._is_global = False
        self._global_context = None
        self._in_loop = False
        self._active = True
        self._start = timer()
        self._last = self._start

        self._itr_stamp_used = dict()
        self._pos_used = []

    name = property(getattr("_name"))
    times = property(getattr("_times"))
    is_global = property(getattr("_is_global"))
    global_context = property(getattr("_global_context"))
    in_loop = property(getattr("_in_loop"))
    start = property(getattr("_start"))
    last = property(getattr("_last"))

    def clear(self):
        name = self._name
        save_itrs = self.save_itrs
        is_global = self._is_global
        self.__init__(name=name, save_itrs=save_itrs)
        self._is_global = is_global

    #
    # Methods operating on the Times data structure.
    #

    def report(self, **kwargs):
        if self._active:
            raise RuntimeError("Can't report an active timer, must stop it first.")
        return self._times.report(**kwargs)

    def print_report(self, **kwargs):
        self._times.print_report(**kwargs)

    def write_structure(self):
        return self._times.write_structure()

    def print_structure(self):
        self._times.print_structure()

    def _prep_timer_obj_arg(self, timer_arg):
        if self._is_global:
            if isinstance(timer_arg, Timer):
                if not timer_arg._is_global:
                    raise ValueError("Global timer can only graft other global g_timers.")
                if timer_arg._name not in g_timers[self._g_context]:
                    raise ValueError("Cannot graft: timer is global but not found in same context.")
            elif timer_arg in g_timers[self._g_context]:
                timer_arg = g_timers[self._g_context][timer_arg]
            else:
                raise ValueError("Invalid timer object or name not found in same context.")
        else:
            if not isinstance(timer_arg, Timer):
                raise TypeError("Valid timer object not recognized for graft.")
        if timer_arg._active:
            timer_arg.stop()
        return timer_arg

    def _times_data_methods(self, times_method, child_timer, **kwargs):
        t = timer()
        child_timer = self._prep_timer_obj_arg(child_timer)
        times_method(child_timer._times, **kwargs)
        if self._active:
            self._times._self += timer() - t

    def absorb(self, partner_timer):
        self._times_data_methods(self._times.absorb, partner_timer, copy_self=False)
        self._stamp_names += partner_timer._stamp_names

    def graft(self, child_timer, position_name):
        if position_name not in self._pos_used:
            self._pos_used += [position_name]
        self._times_data_methods(self._times.graft, child_timer, position_name=position_name)

    #
    # Timing methods.
    #

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    def _check_duplicate(self, name):
        if name in self._times._stamps_ordered:
            w = "Duplicate stamp name used: {}\n".format(repr(name))
            raise ValueError(w)
        self._times._calls += 1

    def stamp(self, name):
        """ Assigns the time since the previous stamp to the <name> key. """
        t = timer()
        if not self._active:
            raise RuntimeError(Timer._inactive_error)
        self._check_duplicate(name)
        self._times._stamps_ordered.append(name)
        self._times._stamps[name] = t - self._last
        self._times._calls += 1
        self._last = timer()
        self._times._self += self._last - t
        return t

    def stop(self):
        t = timer()
        if not self._active:
            raise RuntimeError("Timer already stopped.")
        if self._in_loop:
            raise RuntimeError("Cannot stop timer without exiting loop.")
        for name in self._pos_used:
            if name in self._stamp_names:
                self._pos_used.remove(name)
        if self._pos_used:
            raise RuntimeError("Children awaiting non-existent graft positions (stamps): {}".format(self._pos_used))
        self._times._total += t - self._start - self._times._self
        self._times._self_agg += self._times._self
        for k, v in self._times._stamps.iteritems():
            self._times._stamps_sum += v
        self._times._calls += 1
        self._times._calls_agg += self._times._calls
        self._active = False
        self._times._stopped = True

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
        if self._itr_stamp_used[name]:
            raise RuntimeError("Loop stamp name used more than once within one iteration.")
        self._itr_stamp_used[name] = True
        elapsed = t - self._last
        if self.save_itrs:
            self._times._stamps_itrs[name].append(elapsed)
        self._times._stamps[name] += elapsed
        self._times._calls += 1
        self._last = timer()
        self._times._self += self._last - t
        return t

    def _enter_loop(self, l_stamps_list):
        t = timer()
        if not self._active:
            raise RuntimeError(Timer._inactive_error)
        if self._in_loop:
            raise RuntimeError("Single timer does not support nested loops, use another timer.")
        if not isinstance(l_stamps_list, (list, tuple)):
            raise TypeError("Expected list or tuple types for arg 'l_stamps_list'.")
        self._in_loop = True
        self._itr_stamp_used.clear()
        self._current_l_stamps = l_stamps_list
        for name in l_stamps_list:
            self._check_duplicate(name)
            self._times._stamps_ordered += [name]
            self._itr_stamp_used[name] = False
            self._times._stamps[name] = 0.
            if self.save_itrs:
                self._times._stamps_itrs[name] = []
        self._times._calls += 1
        self._times._self += timer() - t

    def _loop_start(self):
        t = timer()
        for k in self._itr_stamp_used:
            self._itr_stamp_used[k] = False
        self._times._calls += 1
        self._last = timer()
        self._times._self += self._last - t

    def _loop_end(self):
        t = timer()
        if self.save_itrs:
            for k, v in self._itr_stamp_used.iteritems():
                if not v:
                    self._times._stamps_itrs[k].append(0.)
        self._times._calls += 1
        self._times._self += timer() - t

    # def _exit_loop(self):
    #     t = timer()
    #     self._in_loop = False
    #     self._times._calls += 1
    #     self._times._self += timer() - t

    def timed_for(self, loop_iterable, l_stamps_list):
        self._enter_loop(l_stamps_list)
        for i in loop_iterable:
            self._loop_start()
            yield i
            self._loop_end()
        self._in_loop = False
        self._times._calls += 1

    def timed_while(self, l_stamps_list):
        self._enter_loop(l_stamps_list)
        while self.while_condition:
            self._loop_start()
            yield None
            self._loop_end()
        self._in_loop = False
        self.while_condition = True
        self._times._calls += 1

    def set_while_false(self):
        self.while_condition = False

    def set_while_true(self):
        self.while_condition = True


#
# Module provides globalized container for timers.
#


g_timers = dict()


def G_Timer(names, save_itrs=True, context='default_context'):
    if not isinstance(names, (list, tuple)):
        return _make_g_timer(names, save_itrs, context)
    if len(names) == 1:
        return _make_g_timer(names[0], save_itrs, context)
    else:
        ret = ()
        for name in names:
            ret += (_make_g_timer(name, save_itrs, context), )
        return ret


def _make_g_timer(name, save_itrs, context):
    if context not in g_timers:
        g_timers[context] = dict()
    if name in g_timers[context]:
        raise ValueError("Global timers must have unique names within context.")
    new_timer = Timer(name=name, save_itrs=save_itrs)
    new_timer.is_global = True
    new_timer.global_context = context
    g_timers[context][name] = new_timer
    return new_timer


def clear_g_context(context=None):
    if context is not None:
        if context in g_timers:
            g_timers[context].clear()
    else:
        g_timers.clear()


def get_g_timer(name, context='default_context'):
    try:
        return g_timers[context][name]
    except KeyError:
        print "WARNING: Timer name ('{}') and/or context ('{}') not found.".format(name, context)
        pass
