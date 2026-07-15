import os
import psutil


def get_cpu_mem_usage_by_gb():
    pid = os.getpid()
    process = psutil.Process(pid)
    mem_info = process.memory_info()
    mem_usage = mem_info.rss / 1024.0 / 1024.0 / 1024.0  # Convert bytes to GB

    return mem_usage