import time
import psutil
from contextlib import contextmanager

@contextmanager
def timed_section(section_name: str, logger: dict):
    start = time.time()
    yield
    end=time.time()
    duration = end - start
    logger[section_name] = logger.get(section_name, 0.0) + duration
    
def get_memory_usage():
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024 / 1024
    return round(mem_mb, 2)
