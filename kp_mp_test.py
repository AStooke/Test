from line_profiler import LineProfiler
import line_profiler as lp
import numpy as np
import multiprocessing as mp

def func_1(num):
	num += 100
	print num


def main():
	n_proc = 2
	num = 100
	proc = [mp.Process(target=func_1,
						args=(num,))
				for proc in range(n_proc) ]
	for p in proc: p.start()
	for p in proc: p.join()


if __name__ == '__main__':
    profile = LineProfiler()
    new_func = profile.__call__(main)
    profile.runcall(new_func)
    profile.print_stats()
