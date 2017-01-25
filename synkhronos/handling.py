"""
Helper functions for setting up variables and functions.
"""

from util import Inputs, Shareds


def gpu_outputs(outputs):
    """
    Change all outputs to remain on GPU, if not already.  Record which were
    requested to return to CPU so they can be transfered after collecting.
    """
    if outputs is None:
        return None, None
    else:
        from theano.gpuarray.type import GpuArrayVariable
        if not isinstance(outputs, (list, tuple)):
            outputs = (outputs,)
        outputs = list(outputs)
        outputs_to_cpu = list()
        for idx, otpt in enumerate(outputs):
            if isinstance(otpt, GpuArrayVariable):
                outputs_to_cpu.append(False)
            else:
                outputs_to_cpu.append(True)
                outputs[idx] = otpt.transfer(None)
        return tuple(outputs), tuple(outputs_to_cpu)


def register_inputs(theano_function, inputs_global, shareds_global):
    input_codes = list()
    shared_codes = list()
    for store in theano_function.input_storage:
        if store.implicit:  # (a shared variable)
            for idx, gpuarray in enumerate(shareds_global.gpuarrays):
                if store.data is gpuarray:  # (a previously registered shared)
                    shared_codes.append(idx)
                    break
            else:  # (does not match any previously registered)
                sh_code = shareds_global.append(store)
                shared_codes.append(sh_code)
        else:  # (an explicit input)
            if store.name is None or store.name not in inputs_global.names:
                inpt_code = inputs_global.append(store)
                input_codes.append(inpt_code)
            else:
                input_codes.append(inputs_global.names.index(store.name))
    return tuple(input_codes), tuple(shared_codes)


def unpack_functions(theano_functions):
    """
    Worker will recover shared variables in the same order as the master
    committed them, so they will have the same code (index).
    """
    from worker import Function

    synk_functions = list()
    inputs = Inputs()
    shareds = Shareds()
    for idx, fcn in enumerate(theano_functions):
        input_codes, shared_codes = register_inputs(fcn, inputs, shareds)
        synk_functions.append(Function(name=fcn.name,
                                       code=idx,
                                       theano_function=fcn,
                                       input_codes=input_codes,
                                       shared_codes=shared_codes)
                              )
    return synk_functions, inputs, shareds
