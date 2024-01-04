import psutil
import os


def test_psutil_vs_os():
    proc = psutil.Process()
    print("psutil proc affinity", proc.cpu_affinity())
    print("                  =>", len(proc.cpu_affinity()))
    print("CPU count with hyperthreads:", psutil.cpu_count(logical=True))
    print("CPU count ignoring hyperthreads:", psutil.cpu_count(logical=False))
    print("os.sched_affinity:", os.sched_getaffinity(0))
    print("                =>", len(os.sched_getaffinity(0)))
    print("cpuinfo")
    os.system("cat /proc/cpuinfo")
    print("lscpu")
    os.system("lscpu")
    print("lscpu extended")
    os.system("lscpu --extended")
    print("lscpu extended offline")
    os.system("lscpu --extended --offline")
    raise RuntimeError()
