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
        self._parent = None
        self._pos_in_parent = None
        self._children = dict()
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

    def graft(self, child_times, position_name='Unassigned'):
        """ New graft doesn't actually move any data"""
        if not isinstance(child_times, Times):
            raise TypeError("Times.graft() expects Times object for child_times.")
        self._children[position_name].append(child_times)
        child_times._parent = self
        child_times._pos_in_parent = position_name

    def _recv_tmp_dict(self, tmp_times, dict_name):
        self_dict = getattr(self, dict_name)
        tmp_dict = getattr(tmp_times, dict_name)
        for k, v in tmp_dict.iteritems():
            if k in self_dict:
                self_dict[k] += v
            else:
                self_dict[k] = v

    def _recv_tmp_dump(self, tmp_times, first_dump=True):
        for k in tmp_times._stamps:
            if k not in self._stamps_ordered:
                self._stamps_ordered.append(k)
        self._recv_tmp_dict(tmp_times, '_stamps')
        self._recv_tmp_dict(tmp_times, '_stamps_itrs')
        # Maybe make a separate save_self_itrs option.
        if not first_dump and self._save_itrs:
            for k, v in tmp_times._stamps.iteritems():
                if k not in tmp_times._stamps_itrs:
                    if k not in self._stamps_itrs:
                        self._stamps_itrs[k] = []
                    self._stamps_itrs[k].append(v)
        for k in self._grabs_accum_keys:
            self.__dict__[k] += tmp_times.__dict__[k]

    def _aggregate_up(self, new_times):
        self._self_agg += new_times._self_agg
        self._calls_agg += new_times._calls_agg
        if self._parent is not None:
            self._parent._aggregate_up(new_times)

    def combine_data(self, partner_times):
        if not isinstance(partner_times, Times):
            raise TypeError("Valid Times object not recognized for combining data.")
        if not partner_times._stopped:
            raise RuntimeError("Cannot combine data from running times object, partner must be stopped.")
        for k in self._grabs_accum_keys:
            self.__dict__[k] += partner_times.__dict__[k]
        for k, v in partner_times._stamps.iteritems():
            if k not in self._stamps:
                self._stamps[k] = v
            else:
                raise ValueError("Cannot combine stamps by the same name.")
        self._stamps_ordered += partner_times._stamps_ordered
        for k, v in partner_times._stamps_itrs.iteritems():
            self._stamps_itrs[k] = v
        for pos, child in partner_times._children:
            if pos not in self._children:
                self._children[pos] = []
            child_copy = copy.deepcopy(child)
            child_copy._parent = self
            self._children[pos] += [child_copy]
        self._num_descendants += partner_times._num_descendants

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
    in_loop = property(lambda _: None)
    start = property(lambda _: None)
    last = property(lambda _: None)
    is_global = property(lambda _: None)
    g_context = property(lambda _: None)

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

    def combine(self, *args, **kwargs):
        pass

    def graft(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self):
        pass

    def register_stamps(self, *args, **kwargs):
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

    _error_msgs = {'inactive': "Cannot use stopped or paused timer.",
                   'no_loop': "Must be in timed loop to use loop methods."
                   }

    def __init__(self, name='', save_itrs=True):
        self._disabled = False
        self._name = name
        self._save_itrs_orig = save_itrs
        if hasattr(self, '_times'):
            self._times.clear()
            self._tmp_times.clear()
        else:
            self._times = Times(name=name)
            self._tmp_times = Times()
        if hasattr(self, '_l_sub_timer') and self._l_sub_timer is not None:
            self._l_sub_timer.clear()
        else:
            self._l_sub_timer = None
        self._first_tmp_times = True
        self._reg_stamps = []
        self._itr_stamp_used = dict()
        self._reset_self_data()

        self._is_global = False
        self._g_context = None

    def _reset_self_data(self):
        self.while_condition = True
        self._save_itrs = self._save_itrs_orig
        self._in_loop = False
        self._in_named_loop = False
        self._active = True
        self._paused = False
        self._tmp_self_t = 0.
        self._tmp_calls = 0
        self._itr_stamp_used.clear()
        self._start_t = timer()
        self._last_t = self._start_t

    def _dump_tmp_times(self):
        self._times._recv_tmp_dump(self._tmp_times, first_dump=self._first_tmp_times)
        self._first_tmp_times = False

    # all overrides
    name = property(attrgetter("_name"))
    save_itrs = property(attrgetter("_save_itrs_orig"))
    times = property(attrgetter("_times"))
    in_loop = property(attrgetter("_in_loop"))
    start = property(attrgetter("_start_t"))
    last = property(attrgetter("_last_t"))
    is_global = property(attrgetter("_is_global"))
    g_context = property(attrgetter("_g_context"))

    @overrides
    def clear(self):
        name = self._name
        save_itrs = self._save_itrs_orig
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
            self._dump_tmp_self_data(t)
            # need to sort out what to do about printing tmp_times, also need to combine this function with print_report
            rep = self._times.report(**kwargs)
            elapsed = timer() - t
            self._tmp_self_t += elapsed
            self._start_t += elapsed
            return rep
        else:
            return self._times.report(**kwargs)

    @overrides
    def print_report(self, **kwargs):
        t = timer()
        if self._active:
            self._tmp_calls += 1
            self._dump_tmp_self_data(t)
            # Need to sort out what to do about printing tmp_times
            self._times.print_report(**kwargs)
            elapsed = timer() - t
            self._tmp_self_t += elapsed
            self._start_t += elapsed
        else:
            self._times.print_report(**kwargs)

    @overrides
    def write_structure(self):
        return self._times.write_structure()

    @overrides
    def print_structure(self):
        self._times.print_structure()

    def _get_timer_arg(self, timer_arg):
        if self._is_global:
            if isinstance(timer_arg, EmptyTimer):
                if not timer_arg._is_global:
                    raise ValueError("Global timer can only graft other global g_timers.")
                if timer_arg._g_context != self._g_context:  # This is weaker than looking in the context, but needed because automatically spawned timers won't be listed in the context.
                    raise ValueError("Cannot graft: timer is global but not in same context.")
            elif timer_arg in g_timers[self._g_context]:
                timer_arg = g_timers[self._g_context][timer_arg]
            else:
                raise ValueError("Invalid timer object or name not found in same context.")
        else:
            if not isinstance(timer_arg, EmptyTimer):
                raise TypeError("Valid timer object not recognized for graft.")
        return timer_arg

    @overrides
    def combine_data(self, timer_arg):
        partner_timer = self._get_timer_arg(timer_arg)
        if not self._disabled and not partner_timer._disabled:
            if partner_timer._active:
                partner_timer.stop()
            self._times.combine(partner_timer)

    @overrides
    def graft(self, timer_arg, position_name='Unassigned'):
        child_timer = self._get_timer_arg(timer_arg)
        self._times.graft(child_timer._times, position_name=position_name)

    #
    # Timing methods.
    #

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    def start_run(self):
        self._tmp_times.clear()
        self._reset_self_data()

    def _check_duplicate(self, name):
        if name in self._tmp_times._stamps_ordered:
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
    def stamp(self, name, allow_disjoint=False):
        """ Assigns the time since the previous stamp to the <name> key. """
        t = timer()
        if not self._active:
            raise RuntimeError(Timer._error_msgs['inactive'])
        elapsed = t - self._last_t
        if not allow_disjoint:
            self._check_duplicate(name)
        if name not in self._tmp_times._stamps_ordered:
            self._tmp_times._stamps_ordered.append(name)
            self._tmp_times._stamps[name] = elapsed
        else:
            self._tmp_times._stamps[name] += elapsed
        self._tmp_times._stamps_sum += elapsed
        self._tmp_calls += 1
        self._last_t = timer()
        self._tmp_self_t += self._last_t - t
        return t

    def _dump_tmp_self_data(self, total_mark):
        t = timer()
        self._tmp_times._total += total_mark - self._start_t - self._tmp_self_t
        self._tmp_times._self += self._tmp_self_t
        self._tmp_times._self_agg += self._tmp_self_t
        self._tmp_times._calls += self._tmp_calls + 1
        self._tmp_times._calls_agg += self._tmp_calls + 1
        self._tmp_calls = 0
        self._start_t = timer()
        self._tmp_self_t = self._start_t - t

    @overrides
    def stop(self, name=None, d_stamp=False):
        if name is not None:
            if d_stamp:
                self.d_stamp(name)
            else:
                self.stamp(name)
        t = timer()
        if not self._active and not self._paused:
            raise RuntimeError("Timer already stopped.")
        if self._in_loop:
            raise RuntimeError("Cannot stop timer without exiting loop.")
        self._tmp_calls += 1
        self._dump_tmp_self_data(t)
        for name in self._reg_stamps:
            if name not in self._tmp_times._stamps_ordered:
                self._tmp_times._stamps_ordered.append(name)
                self._tmp_times._stamps[name] = 0.
        self._tmp_times._stopped = True  # What do I use this for again??
        self._dump_tmp_times()
        self._active = False
        return t

    @overrides
    def pause(self):
        t = timer()
        if not self._active:
            raise RuntimeError("Cannot pause a stopped timer.")
        self._tmp_times._total += t - self._last_t
        self._active = False
        self._paused = True
        self._tmp_calls += 1
        self._tmp_self_t += timer() - t
        return t

    @overrides
    def resume(self):
        t = timer()
        if not self._paused:
            raise RuntimeError("Can only resume() a paused timer.")
        self._active = True
        self._paused = False
        self._tmp_calls += 1
        self._start_t = t
        self._last_t = t
        return t

    @overrides
    def b_stamp(self, *args, **kwargs):
        if not self._active:
            raise RuntimeError(Timer._error_msgs['inactive'])
        self._tmp_calls += 1
        self._last_t = timer()
        return self._last_t

    #
    # Loop methods.
    #

    @overrides
    def l_stamp(self, name, allow_disjoint=False):
        """ Assigns the time since the previous stamp to this times key. """
        t = timer()
        if not self._active:
            raise RuntimeError(Timer._error_msgs['inactive'])
        if not self._in_loop:
            raise RuntimeError(Timer._error_msgs['no_loop'])
        if self._in_named_loop:
            if self._l_sub_timer is None:
                self._spawn_l_sub_timer(backdate=True)
            self._l_sub_timer.l_stamp(name)
        else:
            if name not in self._l_stamps:
                self._init_l_stamp(name)
            if not allow_disjoint and self._itr_stamp_used[name]:
                raise RuntimeError("Loop stamp name used more than once within one iteration.")
            elapsed = t - self._last_t
            self._tmp_times._stamps[name] += elapsed
            self._tmp_times._stamps_sum += elapsed
            if self._save_itrs:
                if allow_disjoint and self._itr_stamp_used[name]:
                    self._tmp_times._stamps_itrs[name][-1] += elapsed
                else:
                    self._tmp_times._stamps_itrs[name].append(elapsed)
            self._itr_stamp_used[name] = True
            self._last_t = timer()
        self._tmp_calls += 1
        self._tmp_self_t += self._last_t - t
        return t

    def _named_loop_stamp(self):
        t = timer()
        elapsed = t - self._last_t
        self._tmp_times._stamps[self._loop_name] += elapsed
        self._tmp_times._stamps_sum += elapsed
        if self._save_itrs:
            self._tmp_times._stamps_itrs[self._loop_name].append(elapsed)
        self._tmp_calls += 1
        self._last_t = timer()
        self._tmp_self_t += self._last_t - t

    def _init_l_stamp(self, name):
        self._check_duplicate(name)
        self._l_stamps.append(name)
        self._tmp_times._stamps_ordered.append(name)
        self._itr_stamp_used[name] = False
        self._tmp_times._stamps[name] = 0.
        if self._save_itrs:
            self._tmp_times.stamps_itrs[name] = []
        self._tmp_calls += 1

    def _spawn_l_sub_timer(self, rgstr_l_stamps=None, backdate=False):
        self._l_sub_timer = Timer(name=self._loop_name, save_itrs=self._save_itrs)
        self._l_sub_timer._is_global = self._is_global
        self._l_sub_timer._g_context = self._g_context
        # Don't actually expose it in the g_context though, that would be confusing
        self.graft(self._l_sub_timer, self._loop_name)
        self._l_sub_timer._enter_loop(rgstr_l_stamps=rgstr_l_stamps)
        if backdate:
            # self._l_sub_timer._loop_start()  # currently not needed
            self._l_sub_timer._start_t = self._loop_start_t  # beginning of looping
            self._l_sub_timer._loop_start_t = self._loop_start_t
            self._l_sub_timer._last = self._last  # beginning of this loop
        self._tmp_calls += 1

    def _enter_loop(self, loop_name=None, rgstr_l_stamps=None, save_itrs=None):
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
        self._l_start_t = t
        self._itr_stamp_used.clear()
        self._l_stamps = []
        self._l_reg_stamps = []
        if loop_name is not None:
            self._init_l_stamp(loop_name)
            self._in_named_loop = True
        if rgstr_l_stamps is not None:
            if self._in_named_loop:
                self._spawn_l_sub_timer(rgstr_l_stamps=rgstr_l_stamps)
            else:
                if not isinstance(rgstr_l_stamps, (list, tuple)):
                    raise TypeError("Expected list or tuple types for arg 'rgstr_l_stamps'.")
                for name in rgstr_l_stamps:
                    self._init_l_stamp(name)
                # self._reg_stamps += rgstr_l_stamps  # Not needed, handled in init_l_stamps()
                self._l_reg_stamps += rgstr_l_stamps
                self._l_stamps += rgstr_l_stamps
        self._tmp_calls += 1
        self._tmp_self_t += timer() - t

    def _loop_start(self):
        t = timer()
        for k in self._itr_stamp_used:
            self._itr_stamp_used[k] = False
        self._tmp_calls += 1
        if self._l_sub_timer is not None:
            self._l_sub_timer._loop_start()
        self._last_t = timer()
        self._tmp_self_t += self._last_t - t

    def _loop_end(self):
        t = timer()
        if self._in_named_loop:
            if self._l_sub_timer is not None:
                self._l_sub_timer._loop_end()
            self._named_loop_stamp()
            t = timer()
        elif self._save_itrs:
            for name in self._l_reg_stamps:
                if not self._itr_stamp_used[name]:
                    self._tmp_times._stamps_itrs[name].append(0.)
        self._tmp_calls += 1
        self._tmp_self_t += timer() - t

    def _exit_loop(self):
        t = timer()
        if self._l_sub_timer is not None:
            self._l_sub_timer._exit_loop()
            self._l_sub_timer.stop()
            self._l_sub_timer = None  # maybe later have it keep the timer object and just clear it.
        self._in_named_loop = False
        self._in_loop = False
        self._tmp_calls += 1
        self._tmp_self_t += timer() - t

    @overrides
    def timed_for(self, loop_iterable, loop_name=None, l_stamps_list=None, save_itrs=None):
        self._enter_loop(loop_name, l_stamps_list, save_itrs)
        for i in loop_iterable:
            self._loop_start()
            yield i
            self._loop_end()
        self._exit_loop()

    @overrides
    def timed_while(self, loop_name=None, l_stamps_list=None, save_itrs=None):
        self._enter_loop(loop_name, l_stamps_list, save_itrs)
        while self.while_condition:
            self._loop_start()
            yield None
            self._loop_end()
        self._exit_loop()
        self.while_condition = True

    @overrides
    def break_for(self):
        self._loop_end()
        self._in_loop = False
        self._tmp_calls += 1


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
