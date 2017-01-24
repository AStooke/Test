"""
Helper functions for setting up variables and functions.
"""

from Input import Input


# def inputs_handling(inputs, global_inputs):
#     """
#     Ensure that each input was previously registered with gpar by name matching.
#     Associate these names with the function so that it knows which
#     multiprocessing shared variables to use.
#     """
#     mp_inputs = list()
#     worker_indeces = None
#     for inpt in inputs:
#         if inpt.name is None:
#             raise ValueError("Naming of theano input variables is required.")
#         for g_inpt in global_inputs:
#             if g_inpt.name == inpt.name:
#                 mp_inputs.append(g_inpt.mp_array)
#                 if worker_indeces is None:
#                     worker_indeces = g_inpt.worker_indeces
#                 elif worker_indeces != g_inpt.worker_indeces:
#                     raise ValueError("Different worker indeces for different inputs to same function.")
#                 break
#         else:
#             raise ValueError("Theano input variable had no recognized gpar input variable (matching by name).")
#     return tuple(mp_inputs), worker_indeces


def gpu_outputs(outputs):
    """
    Change all outputs to remain on GPU, if not already.  Record which were
    requested to return to CPU so they can be transfered after reducing.
    """
    if outputs is None:
        return None, None
    else:
        from theano.gpuarray.type import GpuArrayVariable
        if isinstance(outputs, (list, tuple)):
            outputs_to_cpu = list()
            for idx, otpt in enumerate(outputs):
                if isinstance(otpt, GpuArrayVariable):
                    outputs_to_cpu.append(False)
                else:
                    outputs_to_cpu.append(True)
                    outputs[idx] = otpt.transfer(None)
            return tuple(outputs), tuple(outputs_to_cpu)
        else:
            if isinstance(outputs, GpuArrayVariable):
                outputs_to_cpu = False
            else:
                outputs_to_cpu = True
                outputs = outputs.transfer(None)
            return outputs, outputs_to_cpu


def register_inputs(inputs, global_inputs, global_named_inputs):
    fcn_input_codes = list()
    for inpt in inputs:
        if inpt.name is None:
            new_code = len(global_inputs)
            global_inputs.append(Input(new_code))
            fcn_input_codes.append(new_code)
            raise RuntimeWarning("Gpar encountered un-named input: shared memory management is improved if inputs used in multiple functions are named.")
        else:
            if inpt.name in global_named_inputs:
                fcn_input_codes.append(global_named_inputs[inpt.name])
            else:
                new_code = len(global_inputs)
                global_inputs.append(Input(new_code, inpt.name))
                fcn_input_codes.append(new_code)
                global_named_inputs[inpt.name] = new_code
    return tuple(fcn_input_codes)


def register_shareds(theano_function, global_shareds, global_named_shareds):
    """
    Find any shared variables associated with the function, and keep a
    centralized list of all such shareds (only one entry per distinct variable).
    Explicitly associate shared variables with this function by their index
    (code) in the overall list.

    NOTE: Modifies inputs "global" by side-effect.
    """
    fcn_shared_codes = list()
    for store in theano_function.input_storage:
        if store.data is not None:  # (then it is a shared variable)
            for idx, shared in enumerate(global_shareds):
                if store.data is shared.data:
                    fcn_shared_codes.append(idx)  # a previously used shared
                    break
            else:  # (does not match any previously used)
                next_code = len(global_shareds)
                global_shareds.append(store)
                fcn_shared_codes.append(next_code)
                if store.name is not None:
                    global_named_shareds[store.name] = next_code
    return tuple(fcn_shared_codes)


def unpack_functions(theano_functions, inputs, rank):
    """
    Worker will recover shared variables in the same order as the master
    committed them, so they will have the same code (index).
    """
    from Function import WorkerFunction

    functions = list()
    shareds = list()
    named_shareds = dict()
    for fcn in theano_functions:
        worker_indeces = None
        fcn_mp_inputs = list()
        fcn_shared_codes = register_shareds(fcn, shareds, named_shareds)
        for store in fcn.input_storage:
            # Register inputs.
            if store.data is None:  # (it's an explicit input, not a shared)
                assert store.name is not None, "Worker encountered function input with no name."
                for inpt in inputs:
                    if inpt.name == store.name:
                        fcn_mp_inputs.append(inpt.mp_array)
                        if worker_indeces is None:
                            worker_indeces = inpt.worker_indeces
                        elif worker_indeces != inpt.worker_indeces:
                            raise ValueError("Different inputs to same function have different worker index boundaries.")
                    break
                else:
                    raise ValueError("Worker could not match function input name to multiprocessing shared variable name.")

        functions.append(WorkerFunction(name=fcn.name,
                                        theano_function=fcn,
                                        mp_inputs=tuple(fcn_mp_inputs),
                                        shared_codes=fcn_shared_codes,
                                        mp_indeces=worker_indeces[rank],
                                        )
        )

    return functions, shareds, named_shareds
