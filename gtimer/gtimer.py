from timeit import default_timer as timer
import copy
from operator import attrgetter
from overrides import overrides

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

    _grabs_accum_keys = ['_total',
                         '_stamps_sum',
                         '_self',
                         '_self_agg',
                         '_calls',
                         '_calls_agg',
                         '_grabs_agg'
                         ]

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
        for old_child in self._children:
            if position_name == old_child._pos_in_parent:
                if child_times._name == old_child._name:
                    old_child._parent = self
                    self._graft_existing(old_child, child_times)
                    break
        else:
            child_copy = copy.deepcopy(child_times)
            child_copy._parent = self
            child_copy._pos_in_parent = position_name
            self._children.append(child_copy)
            self._num_descendants += child_copy._num_descendants + 1
        self._grabs_agg += child_times._num_descendants + 1
        if aggregate_up:
            self._aggregate_up(child_times)

    def _graft_existing(self, old_child, new_child):
        self._absorb_dict(old_child, new_child, '_stamps')
        self._absorb_dict(old_child, new_child, '_stamps_itrs')
        for k in self._grabs_accum_keys:
            old_child.__dict__[k] += new_child.__dict__[k]
        for grandchild in new_child._children:
            old_child.graft(grandchild, grandchild._pos_in_parent, aggregate_up=False)

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
        # if not self._stopped:
        #     raise RuntimeError("Cannot report an active Times structure, must be stopped.")
        fmt_flt, fmt_gen, fmt_int = self._header_formats()
        rep = "\n---Timer Report---"
        if self._name:
            rep += fmt_gen.format('Timer:', repr(self._name))
        rep += fmt_flt.format('Total:', self._total)
        rep += fmt_flt.format('Stamps Sum:', self._stamps_sum)
        if include_diagnostics:
            rep += fmt_flt.format('Self:', self._self)
            rep += fmt_flt.format('Self Agg.:', self._self_agg)
            rep += fmt_int.format('Calls:', self._calls)
            rep += fmt_int.format('Calls Agg.:', self._calls_agg)
            rep += fmt_int.format('Grabs Agg.:', self._grabs_agg)
        rep += "\n\nIntervals\n---------"
        rep += self._report_stamps()
        if include_itrs:
            rep_itrs = ''
            rep_itrs += self._report_itrs()
            if rep_itrs:
                rep += "\n\nLoop Iterations\n---------------"
                rep += rep_itrs
        rep += "\n---End Report---\n"
        return rep

    def _report_stamps(self, indent=0, prec=4):
        s_rep = ''
        fmt = "\n{}{{:.<24}} {{:.{}g}}".format(' ' * indent, prec)
        for stamp in self._stamps_ordered:
            s_rep += fmt.format("{} ".format(stamp), self._stamps[stamp])
            for child in self._children:
                if child._pos_in_parent == stamp:
                    s_rep += child._report_stamps(indent=indent + 2)
        return s_rep

    def _report_itrs(self):
        rep_itrs = ''
        if self._stamps_itrs:
            fmt_flt, fmt_gen, _ = self._header_formats()
            if self._name:
                rep_itrs += fmt_gen.format('Timer:', repr(self._name))
            if self._parent is not None:
                rep_itrs += fmt_gen.format('Parent Timer:', repr(self._parent._name))
                lin_str = self._fmt_lineage(self._get_lineage())
                rep_itrs += fmt_gen.format('Stamp Lineage:', lin_str)
            rep_itrs += "\n\nIter."
            stamps_itrs_order = []
            is_key_active = []
            for k in self._stamps_ordered:
                if k in self._stamps_itrs:
                    stamps_itrs_order += [k]
                    is_key_active += [True]
            for k in stamps_itrs_order:
                rep_itrs += "\t{:<12}".format(k)
            rep_itrs += "\n-----"
            for k in stamps_itrs_order:
                rep_itrs += "\t------\t"
            itr = 0
            while any(is_key_active):
                next_line = '\n{:<5,}'.format(itr)
                for i, k in enumerate(stamps_itrs_order):
                    if is_key_active[i]:
                        try:
                            val = self._stamps_itrs[k][itr]
                            if val < 0.001:
                                prec = 2
                            else:
                                prec = 3
                            next_line += "\t{{:.{}g}}\t".format(prec).format(val)
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
        fmt_int = fmt_name + "{:,}"
        return fmt_flt, fmt_gen, fmt_int

    def _get_lineage(self):
        if self._pos_in_parent is not None:
            return self._parent._get_lineage() + (repr(self._pos_in_parent), )
        else:
            return tuple()

    def _fmt_lineage(self, lineage):
        lin_str = ''
        for link in lineage:
            lin_str += "({})-->".format(link)
        try:
            return lin_str[:-3]
        except IndexError:
            pass

    def print_report(self, include_diagnostics=True):
        print self.report(include_diagnostics=include_diagnostics)

    def write_structure(self):
        struct_str = '\n---Times Data Tree---\n'
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
# Placeholder class with same signature as main Timer class, for disabled timers.
#


class EmptyTimer(object):

    def __init__(self, *args, **kwargs):
        self.while_condition = True
        self._disabled = True

    disabled = property(attrgetter("_disabled"))
    name = property(lambda _: None)
    save_itrs = property(lambda _: None)
    times = property(lambda _: None)
    is_global = property(lambda _: None)
    g_context = property(lambda _: None)
    in_loop = property(lambda _: None)
    start = property(lambda _: None)
    last = property(lambda _: None)

    def clear(self):
        self.__init__()

    def report(self, *args, **kwargs):
        pass

    def print_report(self, *args, **kwargs):
        pass

    def write_structure(self):
        pass

    def print_structure(self):
        pass

    def absorb(self, *args, **kwargs):
        pass

    def graft(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self):
        pass

    def stamp(self, *args, **kwargs):
        pass

    def stop(self):
        pass

    def pause(self):
        pass

    def resume(self):
        pass

    def b_stamp(self, *args, **kwargs):
        pass

    def l_stamp(self, *args, **kwargs):
        pass

    def timed_for(self, loop_iterable, *args, **kwargs):
        for i in loop_iterable:
            yield i

    def timed_while(self, *args, **kwargs):
        while self.while_condtion:
            yield None
        self.while_condition = True

    def break_for(self):
        pass

    def set_while_false(self):
        self.while_condition = False

    def set_while_true(self):
        self.while_condition = True


#
# Main class for timing.
#


class Timer(EmptyTimer):

    _error_msgs = {'inactive': "Can't use stopped or paused timer (can clear() to reset or resume() from pause).",
                   'no_loop': "Must be in timed loop to use loop methods."
                   }

    def __init__(self, name='', save_itrs=True):
        self.while_condition = True

        self._disabled = False
        self._save_itrs_orig = save_itrs
        self._save_itrs = save_itrs
        self._name = name
        if hasattr(self, "_times"):
            self._times.clear()
        else:
            self._times = Times(self._name)
        self._is_global = False
        self._g_context = None
        self._in_loop = False
        self._active = True
        self._tmp_self = 0.  # Finish using this.
        self._tmp_calls = 0
        self._start = timer()
        self._last = self._start

        self._itr_stamp_used = dict()
        self._pos_used = []

    # all overrides
    name = property(attrgetter("_name"))
    save_itrs = property(attrgetter("_save_itrs_orig"))
    times = property(attrgetter("_times"))
    is_global = property(attrgetter("_is_global"))
    g_context = property(attrgetter("_g_context"))
    in_loop = property(attrgetter("_in_loop"))
    start = property(attrgetter("_start"))
    last = property(attrgetter("_last"))

    @overrides
    def clear(self):
        name = self._name
        save_itrs = self.save_itrs
        is_global = self._is_global
        g_context = self._g_context
        self.__init__(name=name, save_itrs=save_itrs)
        self._is_global = is_global
        self._g_context = g_context

    #
    # Methods operating on the Times data structure.
    #

    @overrides
    def report(self, **kwargs):
        t = timer()
        if self._active:
            self._tmp_calls += 1
            self._dump_tmp_times(t)
            rep = self._times.report(**kwargs)
            elapsed = timer() - t
            self._tmp_self += elapsed
            self._start += elapsed
            return rep
        else:
            return self._times.report(**kwargs)

    @overrides
    def print_report(self, **kwargs):
        t = timer()
        if self._active:
            self._tmp_calls += 1
            self._dump_tmp_times(t)
            self._times.print_report(**kwargs)
            elapsed = timer() - t
            self._tmp_self += elapsed
            self._start += elapsed
        else:
            self._times.print_report(**kwargs)

    @overrides
    def write_structure(self):
        return self._times.write_structure()

    @overrides
    def print_structure(self):
        self._times.print_structure()

    def _prep_timer_obj_arg(self, timer_arg):
        if self._is_global:
            if isinstance(timer_arg, Timer):
                if timer_arg._disabled:
                    return None
                if not timer_arg._is_global:
                    raise ValueError("Global timer can only graft other global g_timers.")
                if timer_arg._name not in g_timers[self._g_context]:
                    raise ValueError("Cannot graft: timer is global but not found in same context.")
            elif timer_arg in g_timers[self._g_context]:
                timer_arg = g_timers[self._g_context][timer_arg]
                if timer_arg._disabled:
                    return None
            else:
                for context in g_timers:
                    if timer_arg in context:
                        if timer_arg._disabled:
                            return None
                else:
                    raise ValueError("Invalid timer object or name not found in same context.")
        else:
            if not isinstance(timer_arg, EmptyTimer):
                raise TypeError("Valid timer object not recognized for graft.")
            if timer_arg._disabled:
                return None
        if timer_arg._active:
            timer_arg.stop()
        return timer_arg

    def _times_data_methods(self, times_method, timer_arg, **kwargs):
        t = timer()
        target_timer = self._prep_timer_obj_arg(timer_arg)
        if target_timer is not None:
            target_disabled = False
            times_method(target_timer._times, **kwargs)
            if self._active:
                self._times._self += timer() - t
        else:
            target_disabled = True
        return target_disabled

    @overrides
    def absorb(self, partner_timer):
        partner_disabled = self._times_data_methods(self._times.absorb, partner_timer, copy_self=False)
        if not partner_disabled:
            self._stamp_names += partner_timer._stamp_names

    @overrides
    def graft(self, child_timer, position_name):
        child_disabled = self._times_data_methods(self._times.graft, child_timer, position_name=position_name)
        if not child_disabled:
            if position_name not in self._pos_used:
                self._pos_used += [position_name]

    #
    # Timing methods.
    #

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    def _check_duplicate(self, name):
        if name in self._times._stamps_ordered:
            raise ValueError("Duplicate stamp name used: {}\n".format(repr(name)))
        self._times._calls += 1

    @overrides
    def register_stamps(self, stamp_list):
        if not isinstance(stamp_list, (list, tuple)):
            raise TypeError("Expect a list or tuple for 'stamp_list' argument.")
        stamp_list = list(set(stamp_list))
        self._reg_stamps += stamp_list
        self._tmp_calls += 1

    @overrides
    def stamp(self, name):
        """ Assigns the time since the previous stamp to the <name> key. """
        t = timer()
        if not self._active:
            raise RuntimeError(Timer._error_msgs['inactive'])
        self._check_duplicate(name)
        self._times._stamps_ordered.append(name)
        elapsed = t - self._last
        self._times._stamps[name] = elapsed
        self._times._stamps_sum += elapsed
        self._tmp_calls += 1
        self._last = timer()
        self._tmp_self += self._last - t
        return t

    @overrides
    def d_stamp(self, name):
        t = timer()
        if not self._active:
            raise RuntimeError(Timer._error_msgs['inactive'])
        elapsed = t - self._last
        if name not in self._times._stamps_ordered:
            self._times._stamps_ordered.append(name)
            self._times._stamps[name] = elapsed
        else:
            self._times._stamps[name] += elapsed
        self._times._stamps_sum += elapsed
        self._tmp_calls += 1
        self._last = timer()
        self._tmp_self += self._last - t
        return t

    def _dump_tmp_times(self, total_mark):
        t = timer()
        self._times._total += total_mark - self._start - self._tmp_self
        self._times._self += self._tmp_self
        self._times._self_agg += self._tmp_self
        self._times._calls += self._tmp_calls + 1
        self._times._calls_agg += self._tmp_calls + 1
        self._tmp_calls = 0
        self._start = timer()
        self._tmp_self = self._start - t

    @overrides
    def stop(self):
        t = timer()
        if not self._active:
            raise RuntimeError("Timer already stopped or paused.")
        if self._in_loop:
            raise RuntimeError("Cannot stop timer without exiting loop.")
        for name in copy.deepcopy(self._pos_used):
            if name in self._times._stamps_ordered:
                self._pos_used.remove(name)
        if self._pos_used:
            raise RuntimeError("Children awaiting non-existent graft positions (stamps): {}".format(self._pos_used))
        self._tmp_calls += 1
        self._dump_tmp_times(t)
        for name in self._reg_stamps:
            if name not in self._times._stamps_ordered:
                self._times._stamps_ordered.append(name)
                self._times._stamps[name] = 0.
        self._active = False
        self._times._stopped = True
        return t

    @overrides
    def pause(self):
        t = timer()
        self._times._total += t - self._last
        self._active = False
        self._tmp_calls += 1
        self._tmp_self += timer() - t
        return t

    @overrides
    def resume(self):
        t = timer()
        self._active = True
        self._tmp_calls += 1
        self._start = t
        self._last = t
        return t

    @overrides
    def b_stamp(self, *args, **kwargs):
        self._times._calls += 1
        self._last = timer()
        return self._last

    #
    # Loop methods.
    #

    @overrides
    def l_stamp(self, name):
        """ Assigns the time since the previous stamp to this times key. """
        t = timer()
        if not self._active:
            raise RuntimeError(Timer._error_msgs['inactive'])
        if not self._in_loop:
            raise RuntimeError(Timer._error_msgs['no_loop'])
        if name not in self._l_stamps:
            self._check_duplicate(name)
            self._l_stamps.append(name)
            self._times._stamps_ordered.append(name)
            self._itr_stamp_used[name] = False
            self._times._stamps[name] = 0.
            if self.save_itrs:
                self._times._stamps_itrs[name] = []
        if self._itr_stamp_used[name]:
            raise RuntimeError("Loop stamp name used more than once within one iteration.")
        elapsed = t - self._last
        self._times._stamps[name] += elapsed
        self._times._stamps_sum += elapsed
        if self.save_itrs:
            self._times._stamps_itrs[name].append(elapsed)
        self._itr_stamp_used[name] = True
        self._tmp_calls += 1
        self._last = timer()
        self._tmp_self += self._last - t
        return t

    def _enter_loop(self, loop_name=None, registered_l_stamps=None, save_itrs=None):
        t = timer()
        if not self._active:
            raise RuntimeError(Timer._error_msgs['inactive'])
        if self._in_loop:
            raise RuntimeError("Single timer does not support nested loops, use another timer.")
        if save_itrs is not None:
            self._save_itrs = bool(save_itrs)
        else:
            self._save_itrs = self._save_itrs_orig
        self._in_loop = True
        self._itr_stamp_used.clear()
        if registered_l_stamps is not None:
            if not isinstance(registered_l_stamps, (list, tuple)):
                raise TypeError("Expected list or tuple types for arg 'registered_l_stamps'.")
            for name in registered_l_stamps:
                self._check_duplicate(name)
                self._times._stamps_ordered += [name]
                self._itr_stamp_used[name] = False
                self._times._stamps[name] = 0.
                if self.save_itrs:
                    self._times._stamps_itrs[name] = []
            self._reg_stamps += registered_l_stamps
            self._l_reg_stamps = registered_l_stamps
            self._l_stamps = registered_l_stamps
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
            for name in self._l_reg_stamps:
                if not self._itr_stamp_used[name]:
                    self._times._stamps_itrs[name].append(0.)
        self._times._calls += 1
        self._times._self += timer() - t

    # def _exit_loop(self):
    #     t = timer()
    #     self._in_loop = False
    #     self._times._calls += 1
    #     self._times._self += timer() - t

    @overrides
    def timed_for(self, loop_iterable, loop_name=None, l_stamps_list=None, save_itrs=None):
        self._enter_loop(loop_name, l_stamps_list, save_itrs)
        for i in loop_iterable:
            self._loop_start()
            yield i
            self._loop_end()
        self._in_loop = False
        self._times._calls += 1

    @overrides
    def timed_while(self, loop_name=None, l_stamps_list=None, save_itrs=None):
        self._enter_loop(loop_name, l_stamps_list, save_itrs)
        while self.while_condition:
            self._loop_start()
            yield None
            self._loop_end()
        self._in_loop = False
        self.while_condition = True
        self._times._calls += 1

    @overrides
    def break_for(self):
        self._loop_end()
        self._in_loop = False
        self._times._calls += 1


#
# Module provides globalized container for timers.
#


g_timers = dict()
timers_disabled = False


def _make_g_timer(name, save_itrs, context):
    if context not in g_timers:
        g_timers[context] = dict()
    if name in g_timers[context]:
        raise ValueError("Global timers must have unique names within context.")
    new_timer = Timer(name=name, save_itrs=save_itrs)
    new_timer._is_global = True
    new_timer._g_context = context
    g_timers[context][name] = new_timer
    return new_timer


def _make_empty_timer(name, save_itrs, context):
    new_timer = EmptyTimer()
    g_timers[context][name] = new_timer
    return new_timer


def G_Timer(names, save_itrs=True, context='default_context', disable=None):
    if disable is not None:
        disable = bool(disable)
    else:
        disable = bool(timers_disabled)
    if disable:
        timer_make_func = _make_empty_timer
    else:
        timer_make_func = _make_g_timer
    if not isinstance(names, (list, tuple)):
        return timer_make_func(names, save_itrs, context)
    if len(names) == 1:
        return timer_make_func(names[0], save_itrs, context)
    else:
        ret = ()
        for name in names:
            ret += (timer_make_func(name, save_itrs, context), )
        return ret


def clear_context(context=None):
    if context is not None:
        if context in g_timers:
            g_timers[context].clear()
    else:
        g_timers.clear()


def clear_timers(context=None):
    if context is not None:
        if context in g_timers:
            _clear_in_context(context)
    else:
        for context in g_timers:
            _clear_in_context(context)


def _clear_in_context(context):
    for k, t in g_timers[context].iteritems():
        t.clear()


def get_timer(name, context='default_context'):
    try:
        return g_timers[context][name]
    except KeyError:
        print "WARNING: Timer name ({}) and/or context ({}) not found.\n".format(repr(name), repr(context))


def get_context(context='default_context'):
    try:
        return g_timers[context]
    except KeyError:
        print "WARNING: Timer context {} not found.\n".format(repr(context))
