def reduce_func(g_shareds, gpu_comm, shared_ID, op, in_place=True, dest=None):
    src = g_shareds.gpuarrays[shared_ID]
    if in_place:
        gpu_comm.reduce(src, op=op, dest=src)
        if avg:
            g_shareds.avg_functions[shared_ID]()  # TODO: put in separate loop.
    else:
        if avg:
            raise ValueError("Cannot use 'average' reduce op if not in-place.")
        return gpu_comm.reduce(src, op=op, dest=dest)  # makes a new gpuarray


def all_reduce_func(g_shareds, gpu_comm, shared_ID, op):
    # workers can't get new arrays; everyone (including master) overwrites src
    src = g_shareds.gpuarrays[shared_ID]
    gpu_comm.all_reduce(src, op=op, dest=src)


def broadcast_func(g_shareds, gpu_comm, shared_ID):
    src = g_shareds.gpuarrays[shared_ID]
    gpu_comm.broadcast(src)


def gather_func(g_shareds, gpu_comm, shared_ID, dest=None, nd_up=1):
    src = g_shareds.gpuarrays[shared_ID]
    return gpu_comm.all_gather(src, dest=dest, nd_up=nd_up)


def gpu_comm_function(gpu_comm_func, comm_ID, has_op=False):
    def build_comm_procedure(f):
        @functools.wraps(f)  # (preserves signature and docstring of wrapped)
        def gpu_comm_procedure(functions=None, shared_vars=None, op=None,
                               **kwargs):
            shared_IDs = gpu_comm_prep(comm_ID, functions, shared_vars)
            if has_op:
                op_ID = check_op(op)
                avg = op in AVG_ALIASES
                kwargs["op"] = "sum" if avg else op
                g.sync.comm_op.value = op_ID
            g.sync.barriers.exec_in.wait()
            results = list()
            for shared_ID in shared_IDs:
                r = gpu_comm_func(g.shareds, g.gpu_comm, shared_ID, **kwargs)
                if r is not None:
                    results.append(r)
            if has_op and avg:
                for shared_ID in shared_IDs:
                    g.shareds.avg_funcs[shared_ID]()
            results = None if len(results) == 0 else results
            g.sync.barriers.exec_out.wait()
            return results
        return gpu_comm_procedure
    return build_comm_procedure
