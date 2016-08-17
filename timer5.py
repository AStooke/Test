from timeit import default_timer as timer
import copy

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

    grabs_accum_keys = ['total', 'stamps_sum', 'self_', 'self_agg', 'calls', 'calls_agg', 'grabs_agg']

    def __init__(self, name=''):
        self.name = name
        self.stamps = dict()
        self.stamps_itrs = dict()
        self.total = 0.
        self.stamps_sum = 0.
        self.self_ = 0.
        self.self_agg = 0.
        self.calls = 0
        self.calls_agg = 0
        self.grabs_agg = 0
        self.parent = None
        self.pos_in_parent = None
        self.children = list()
        self._child_pos_awaiting = list()
        self.num_descendents = 0
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

    def graft(self, child_times, position_name, aggregate_up=True):
        if not child_times.stopped:
            raise RuntimeError("Cannot graft running times object, child must be stopped.")
        if position_name not in self.stamps:
            if self.stopped:
                raise ValueError("Position name must be an existing stamp in stopped parent.")
            elif position_name not in self._child_pos_awaiting:
                self._child_pos_awaiting += [position_name]
        is_existing_child = False
        for old_child in self.children:
            if position_name == old_child.pos_in_parent:
                if child_times.name == old_child.name:
                    is_existing_child = True
                    self._graft_existing(old_child, child_times)
        if not is_existing_child:
            child_copy = copy.deepcopy(child_times)
            child_copy.parent = self
            child_copy.pos_in_parent = position_name
            self.children.append(child_copy)
            self.num_descendents += child_copy.num_descendents + 1
        self.grabs_agg += child_times.num_descendents + 1
        if aggregate_up:
            self._aggregate_up(child_times)

    def _graft_existing(self, old_times, new_times):
        self._absorb_dict(old_times, new_times, 'stamps')
        self._absorb_dict(old_times, new_times, 'stamps_itrs')
        for k in Times.grabs_accum_keys:
            old_times.__dict__[k] += new_times.__dict__[k]
        for new_child in new_times.children:
            old_times.graft(new_child, new_child.pos_in_parent, aggregate_up=False)

    def _absorb_dict(self, old_times, new_times, dict_name):
        old_dict = getattr(old_times, dict_name)
        new_dict = getattr(new_times, dict_name)
        for k, v in new_dict.iteritems():
            if k in old_dict:
                old_dict[k] += v
            else:
                old_dict[k] = v

    def _aggregate_up(self, new_times):
        self.self_agg += new_times.self_agg
        self.calls_agg += new_times.calls_agg
        self.grabs_agg += new_times.grabs_agg
        if self.parent is not None:
            self.parent._aggregate_up(new_times)

    def absorb(self, partner_times, copy_self=True):
        if not isinstance(partner_times, Times):
            raise TypeError("Valid Times object not recognized for absorb.")
        if not partner_times.stopped:
            raise RuntimeError("Cannot absorb running times object, partner must be stopped.")
        if copy_self:
            new = copy.deepcopy(self)
        else:
            new = self
        for k in Times.grabs_accum_keys:
            new.__dict__[k] += partner_times.__dict__[k]
        for k, v in partner_times.stamps.iteritems():
            if k not in new.stamps:
                new.stamps[k] = v
            else:
                raise ValueError("Cannot absorb stamps by the same name.")
        new.stamps_order += partner_times.stamps_order
        for k, v in partner_times.stamps_itrs.iteritems():
            new.stamps_itrs[k] = v
        new.children += copy.deepcopy(partner_times.children)
        for child in new.children:
            child.parent = new
        new.num_descendents += partner_times.num_descendents
        self.grabs_agg += partner_times.num_descendents + 1
        if copy_self:
            return new

    #
    # Reporting methods.
    #

    def report(self, include_itrs=True, include_diagnostics=True):
        if not self.stopped:
            raise RuntimeError("Cannot report an active Times structure, must be stopped.")
        fmt_flt, fmt_gen = self._header_formats()
        rep = "\n---Timer Report---"
        if self.name:
            rep += fmt_gen.format('Timer:', repr(self.name))
        rep += fmt_flt.format('Total:', self.total)
        rep += fmt_flt.format('Stamps Sum:', self.stamps_sum)
        if include_diagnostics:
            rep += fmt_flt.format('Self:', self.self_)
            rep += fmt_flt.format('Self Agg.:', self.self_agg)
            rep += fmt_gen.format('Calls:', self.calls)
            rep += fmt_gen.format('Calls Agg.:', self.calls_agg)
            rep += fmt_gen.format('Grabs Agg.:', self.grabs_agg)
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
        if self.stamps_itrs:
            fmt_flt, fmt_gen = self._header_formats()
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

    def __init__(self, name='', save_itrs=True):
        self.name = name
        self.save_itrs = save_itrs
        if hasattr(self, 'times'):
            self.times.clear()
        else:
            self.times = Times(self.name)
        self.while_condition = True
        self.is_global = False
        self._in_loop = False
        self._active = True
        self._stamp_names = []
        self._itr_stamp_used = dict()
        self._start = timer()
        self._last = self._start

    def clear(self):
        name = self.name
        save_itrs = self.save_itrs
        is_global = self.is_global
        self.__init__(name=name, save_itrs=save_itrs)
        self.is_global = is_global

    #
    # Methods operating on the Times data structure.
    #

    def report(self, **kwargs):
        if self._active:
            raise RuntimeError("Can't report an active timer, must stop it first.")
        return self.times.report(**kwargs)

    def print_report(self, **kwargs):
        self.times.print_report(**kwargs)

    def write_structure(self):
        return self.times.write_structure()

    def print_structure(self):
        self.times.print_structure()

    def _check_timer_obj_arg(self, timer_arg):
        if self.is_global:
            if isinstance(timer_arg, Timer):
                if not timer_arg.is_global:
                    raise ValueError("Global timer can only graft other global g_timers.")
                if timer_arg.name not in g_timers:
                    raise ValueError("Cannot graft: timer is global but not found in record.")
            elif timer_arg in g_timers:
                timer_arg = g_timers[timer_arg]
            else:
                raise ValueError("Invalid timer object or name not found in global record.")
        else:
            if not isinstance(timer_arg, Timer):
                raise TypeError("Valid timer object not recognized for graft.")
        if timer_arg._active:
            timer_arg.stop()
        return timer_arg

    def _times_data_methods(self, times_method, child_timer, **kwargs):
        t = timer()
        child_timer = self._check_timer_obj_arg(child_timer)
        times_method(child_timer.times, **kwargs)
        if self._active:
            self.times.self_ += timer() - t

    def absorb(self, child_timer):
        self._times_data_methods(self.times.absorb, child_timer, copy_self=False)

    def graft(self, child_timer, position_name):
        self._times_data_methods(self.times.graft, child_timer, position_name=position_name)

    #
    # Timing methods.
    #

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop()

    def _check_duplicate(self, name):
        if name in self._stamp_names:
            w = "Duplicate stamp name used: {}\n".format(repr(name))
            raise ValueError(w)
        self.times.calls += 1

    def stamp(self, name):
        """ Assigns the time since the previous stamp to the <name> key. """
        t = timer()
        if not self._active:
            raise RuntimeError(Timer._inactive_error)
        self._check_duplicate(name)
        self._stamp_names.append(name)
        self.times.stamps[name] = t - self._last
        self.times.calls += 1
        self._last = timer()
        self.times.self_ += self._last - t
        return t

    def stop(self):
        t = timer()
        if not self._active:
            raise RuntimeError("Timer already stopped.")
        if self._in_loop:
            raise RuntimeError("Cannot stop timer without exiting loop.")
        await = self.times._child_pos_awaiting
        for name in await:
            if name in self._stamp_names:
                await.remove(name)
        if await:
            raise RuntimeError("Children awaiting non-existent graft positions (stamps): {}".format(await))
        self.times.total += t - self._start - self.times.self_
        self.times.self_agg += self.times.self_
        for k, v in self.times.stamps.iteritems():
            self.times.stamps_sum += v
        self.times.calls += 1
        self.times.calls_agg += self.times.calls
        self._active = False
        self.times.stamps_order = self._stamp_names
        self.times.stopped = True

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
            self.times.stamps_itrs[name].append(elapsed)
        self.times.stamps[name] += elapsed
        self.times.calls += 1
        self._last = timer()
        self.times.self_ += self._last - t
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
            self._stamp_names += [name]
            self._itr_stamp_used[name] = False
            self.times.stamps[name] = 0.
            if self.save_itrs:
                self.times.stamps_itrs[name] = []
        self.times.calls += 1
        self.times.self_ += timer() - t

    def _loop_start(self):
        t = timer()
        for k in self._itr_stamp_used:
            self._itr_stamp_used[k] = False
        self.times.calls += 1
        self._last = timer()
        self.times.self_ += self._last - t

    def _loop_end(self):
        t = timer()
        if self.save_itrs:
            for k, v in self._itr_stamp_used.iteritems():
                if not v:
                    self.times.stamps_itrs[k].append(0.)
        self.times.calls += 1
        self.times.self_ += timer() - t

    # def _exit_loop(self):
    #     t = timer()
    #     self._in_loop = False
    #     self.times.calls += 1
    #     self.times.self_ += timer() - t

    def timed_for(self, loop_iterable, l_stamps_list):
        self._enter_loop(l_stamps_list)
        for i in loop_iterable:
            self._loop_start()
            yield i
            self._loop_end()
        self._in_loop = False
        self.times.calls += 1

    def timed_while(self, l_stamps_list):
        self._enter_loop(l_stamps_list)
        while self.while_condition:
            self._loop_start()
            yield None
            self._loop_end()
        self._in_loop = False
        self.while_condition = True
        self.times.calls += 1


#
# Module provides globalized container for timers.
#


g_timers = dict()


def G_Timer(names, save_itrs=True):
    if not isinstance(names, (list, tuple)):
        return _g_timer(names, save_itrs)
    else:
        ret = ()
        for name in names:
            ret += (_g_timer(name, save_itrs), )
        return ret


def _g_timer(name, save_itrs):
    if name in g_timers:
        raise ValueError("Timer name already used, global timers must have unique names.")
    new_timer = Timer(name=name, save_itrs=save_itrs)
    new_timer.is_global = True
    g_timers[name] = new_timer
    return new_timer


def clear_g_timers():
    g_timers.clear()
