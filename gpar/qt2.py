# import functools
# BROADCAST = 7

# def gpu_comm_procedure(comm_code, **kwargs):
#     def gpu_comm_wrap(comm_func):
#         # @functools.wraps(comm_func)
#         def gpu_comm_fcn(functions=None, shared_names=None, **kwargs):
#             print("doing the before stuff.  comm_code: ", comm_code)
#             print("functions: ", functions)
#             print("shared_names: ", shared_names)
#             for idx in shared_names:
#                 print("in the loop, idx: ", idx)
#                 comm_func(idx, **kwargs)
#             print("doing the after stuff.")
#         return gpu_comm_fcn
#     return gpu_comm_wrap


# @gpu_comm_procedure(BROADCAST, root=None)
# def broadcast(idx, root=None):
#     print("In the comm_func. idx: ", idx, " root: ", root)





# def outermost_wrapper(comm_code, **kwargs):
#     def outermost_wrapped(func):
#         print("comm_code: ", comm_code)
#         idx = 7
#         func(idx, **kwargs)
#     return outermost_wrapped


# def inner_wrapper(**kwargs):
#     def inner_inner_wrapper(func):
#         def function(**kwargs):
#             func(**kwargs)


# def make_func(parameters):
#     exec("def f_make_fun(default=None, {}): pass".format(', '.join(parameters)))
#     return locals()['f_make_fun']

# LIST = [10, 11, 12, 13]


# def gpu_comm_helper(comm_code, functions, shared_names):
#     print("comm_helper got the code: ", comm_code)
#     shared_codes = [1, 2, 3]
#     print("comm_helper found the shared codes: ", shared_codes)
#     return shared_codes

# def broadcast(functions=None, shared_names=None, root=None):
#     shared_codes = gpu_comm_helper(BROADCAST, functions, shared_names)
#     print("barrier in")
#     for idx in shared_codes:
#         print("working on idx: ", idx, " and root: ", root)
#     print("barrier out")

# def gpu_comm_looper(comm_func):
#     def gpu_comm_loop(shared_codes, *args, **kwargs):
#         for idx in shared_codes:
#             comm_func(LIST[idx], *args, **kwargs)  # .data
#     return gpu_comm_loop

# @gpu_comm_looper
# def broadcast_loop(var, root=None):
#     print("broadcasting shared var: ", var, " and root: ", root)


# def broadcast(functions=None, shared_names=None, root=None):
#     shared_codes = gpu_comm_helper(BROADCAST, functions, shared_names)
#     print("barrier in")
#     broadcast_loop(shared_codes, root)
#     print("barrier out")


# @gpu_comm_procedure
# @gpu_comm_looper
# def broadcast(var, root=None):
#     print("broadcasting shared var: ", var, " and root: ", root)

# def outer(comm_code, loop_func):
#     def gpu_comm_procedure(func):
#         def inner(comm_code, loop_func):
#             def exposed_function(functions=None, shared_names=None, **kwargs):
#                 shared_codes = gpu_comm_helper(comm_code, functions, shared_names)
#                 print("barrier in ")
#                 looper_func(shared_codes, **kwargs)
#                 print("barrier out")

# def inner_template(comm_code, looper_func, functions=None, shared_names=None, **kwargs):
#     shared_codes = gpu_comm_helper(comm_code, functions, shared_names)
#     print("barrier in")
#     looper_func(shared_codes, **kwargs)
#     print("barrier out")

##### RIGHT HERE BABY  vvvvvvvvv  ########


# LIST = [10, 11, 12, 13]


# def gpu_comm_helper(comm_code, functions, shared_names):
#     print("comm_helper got the code: ", comm_code)
#     shared_codes = [1, 2, 3]
#     print("comm_helper found the shared codes: ", shared_codes)
#     return shared_codes

# def broadcast_fcn(var, root=None):
#     print("broadcasting shared var: ", var, " and root: ", root)


# def inner_template(comm_code, comm_func, functions=None, shared_names=None, **kwargs):
#     shared_codes = gpu_comm_helper(comm_code, functions, shared_names)
#     print("barrier in")
#     for idx in shared_codes:
#         comm_func(LIST[idx], **kwargs)
#     print("barrier out")


# def gpu_comm_outer(comm_code, comm_func):
#     def gpu_comm_procedure(f):
#         @functools.wraps(f)
#         def procedure(functions=None, shared_names=None, **kwargs):
#             inner_template(comm_code, comm_func, functions=None, shared_names=None, **kwargs)
#         return procedure
#     return gpu_comm_procedure


# @gpu_comm_outer(BROADCAST, broadcast_fcn)
# def broadcast(functions=None, shared_names=None, root=None):
#     """ broadcast docstring """
#     print('in broadcast')


######################  YEAH THIS WORKS ^^^^ #######

#### OK here's one example...  ####
# import ipdb
import functools

def wrapper(inner_func, outer_arg, outer_kwarg=None, has_extra_kwarg=False):
    def wrapped_func(f):
        @functools.wraps(f)
        def template(common_exposed_arg, *other_args, common_exposed_kwarg=None, **other_kwargs):
            print("\nstart of template.")
            print("outer_arg: ", outer_arg, " outer_kwarg: ", outer_kwarg)
            print("common_exposed_kwarg: ", common_exposed_kwarg)
            inner_arg = outer_arg * 10 + common_exposed_arg
            print("other_kwargs: ", other_kwargs)
            if has_extra_kwarg:
                other_args = (*other_args, other_kwargs.pop("extra_kwarg", "default"))
            print("other_args: ", other_args)
            # ipdb.set_trace()
            inner_func(inner_arg, *other_args, common_exposed_kwarg=common_exposed_kwarg, **other_kwargs)
            print("template done")
        return template
    return wrapped_func

# Build two examples.
def inner_fcn_1(hidden_arg, exposed_arg, common_exposed_kwarg=None):
    print("inner_fcn, hidden_arg: ", hidden_arg, ", exposed_arg: ", exposed_arg, ", common_exposed_kwarg: ", common_exposed_kwarg)

def inner_fcn_2(hidden_arg, extra_arg, common_exposed_kwarg=None, other_exposed_kwarg=None):
    print("inner_fcn_2, hidden_arg: ", hidden_arg, ", common_exposed_kwarg: ", common_exposed_kwarg, ", other_exposed_kwarg: ", other_exposed_kwarg)
    print("extra_arg: ", extra_arg)


@wrapper(inner_fcn_1, 1)
def exposed_function_1(common_exposed_arg, other_exposed_arg, common_exposed_kwarg=None):
    """exposed_function_1 docstring: this dummy function exposes the right signature """
    print("this won't get printed")

@wrapper(inner_fcn_2, 2, outer_kwarg="outer", has_extra_kwarg=True)
def exposed_function_2(common_exposed_arg, common_exposed_kwarg=None, other_exposed_kwarg=None, extra_kwarg=None):
    """ exposed_2 doc """
    pass


print("\nAll functions defined.")

exposed_function_1(1, -1, common_exposed_kwarg="common_option_1")
exposed_function_2(2, common_exposed_kwarg="common_option_2", other_exposed_kwarg="special_option", extra_kwarg=17)
exposed_function_2(2, common_exposed_kwarg="common_option_2", other_exposed_kwarg="special_option")

print("\n", exposed_function_1.__name__)
print(exposed_function_1.__doc__)

################################################

# BUT it can be made even simpler....remove the whole inner function business?

# import functools

# def wrapper(f):
#     @functools.wraps(f)
#     def template(common_exposed_arg, *other_args, common_exposed_kwarg=None, **other_kwargs):
#         print("\ninside template.")
#         print("common_exposed_arg: ", common_exposed_arg, ", common_exposed_kwarg: ", common_exposed_kwarg)
#         print("other_args: ", other_args, ",  other_kwargs: ", other_kwargs)
#     return template

# @wrapper
# def exposed_func_1(common_exposed_arg, other_exposed_arg, common_exposed_kwarg=None):
#     """exposed_func_1 docstring: this dummy function exposes the right signature"""
#     print("this won't get printed")

# @wrapper
# def exposed_func_2(common_exposed_arg, common_exposed_kwarg=None, other_exposed_kwarg=None):
#     """exposed_func_2 docstring"""
#     pass

# exposed_func_1(10, -1, common_exposed_kwarg='one')
# exposed_func_2(20, common_exposed_kwarg='two', other_exposed_kwarg='done')
# print("\n" + exposed_func_1.__name__)
# print(exposed_func_1.__doc__)


# RESULT:

    # >> inside template.
    # >> common_exposed_arg:  10 , common_exposed_kwarg:  one
    # >> other_args:  (-1,) ,  other_kwargs:  {}
    # >>
    # >> inside template.
    # >> common_exposed_arg:  20 , common_exposed_kwarg:  two
    # >> other_args:  () ,  other_kwargs:  {'other_exposed_kwarg': 'done'}
    # >>
    # >> exposed_func_1
    # >> exposed_func_1 docstring: this dummy function exposes the right signature







# # def broadcast(functions=None, shared_names=None, root=None):
# #     shared_codes = gpu_comm_helper(BROADCAST, functions, shared_names)
# #     broadcast_fcn(shared_codes, root)
# #     g.sync.barriers.exec_out.wait()

# def gpu_comm_procedure(comm_fcn):
#     def gpu_comm_thing(*args, functions=None, shared_names=None, **kwargs):
#         shared_codes = gpu_comm_helper(BROADCAST, functions, shared_names)
#         print("gpu_comm_thing got the shared_codes: ", shared_codes)
#         # g.sync.barriers.exec_in.wait()
#         comm_fcn(shared_codes, *args, **kwargs)
#         # g.sync.barriers.exec_out.wait()
#     return gpu_comm_thing


# @gpu_comm_procedure
# def broadcast(functions=None, shared_names=None, root=None):
#     broadcast_fcn(idx, root=None)
