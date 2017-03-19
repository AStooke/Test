import theano


class ChangeFlags(object):
    """
    Use this as a decorator or context manager to change the value of
    Theano config variables.
    Useful during tests.
    """
    def __init__(self, args=(), **kwargs):
        confs = dict()
        args = dict(args)
        args.update(kwargs)
        for k in args:
            l = [v for v in theano.configparser._config_var_list
                 if v.fullname == k]
            assert len(l) == 1
            confs[k] = l[0]
        self.confs = confs
        self.new_vals = args

    def __call__(self, f):
        @wraps(f)
        def res(*args, **kwargs):
            with self:
                return f(*args, **kwargs)
        return res

    def __enter__(self):
        self.old_vals = {}
        for k, v in self.confs.items():
            self.old_vals[k] = v.__get__(True, None)
        try:
            for k, v in self.confs.items():
                v.__set__(None, self.new_vals[k])
        except:
            self.__exit__()
            raise

    def __exit__(self, *args):
        for k, v in self.confs.items():
            v.__set__(None, self.old_vals[k])
