
from theano.gof.graph import is_same_graph
from theano.tensor.var import TensorVariable
from theano.compile.sharedvalue import SharedVariable
VAR = (TensorVariable, SharedVariable)


def is_parent(output, maybe_parent, check_graph=True):
    immediate_parents = output.get_parents()
    if check_graph:
        for p in immediate_parents:
            if isinstance(p, VAR) and is_same_graph(p, maybe_parent):
                return True
    else:
        for p in immediate_parents:
            if p is maybe_parent:
                return True
    for p in immediate_parents:
        if is_parent(p, maybe_parent, check_graph):
            return True
    return False
