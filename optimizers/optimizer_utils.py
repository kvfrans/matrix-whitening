from typing import Callable, NamedTuple
from contextlib import contextmanager
import time

class Optimizer(NamedTuple):
    init: Callable
    update: Callable
    get_grad_params: Callable
    get_eval_params: Callable
    update_slow: Callable = None
    log_stats: Callable = None


@contextmanager
def timer(name):
    """Context manager for timing code blocks"""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"{name}: {elapsed:.4f} seconds")