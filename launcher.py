
import subprocess

procs = list()
for i in range(3):
    p = subprocess.Popen(["taskset", "-c", "0-2", "python", "to_be_launched.py", "--ID", str(i)])
    procs.append(p)
print("master done launching")

for p in procs:
    p.wait()
print("master done joining")
