

def build_list(yeah, whoa, kk, *args):
    args_list = list()
    for y, *a in zip(yeah, *args):
        for w in whoa:
            for k in range(kk):
                args = (y, w, k, *a)
                args_list.append(args)
    return args_list


yeah = ['yeah1', 'yeah2']
whoa = ['whoa1', 'whoa2']
kk = 2
arg1 = ['arg1_1', 'arg1_2']
arg2 = ['arg2_1', 'arg2_2']
extra_args = (arg1, arg2)
extra_args = ()

args_list = build_list(yeah, whoa, kk, *extra_args)
