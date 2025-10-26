from __future__ import annotations
import time
from typing import Callable, Type, Iterable

def retry(fn: Callable, exceptions: Iterable[Type[BaseException]], attempts: int = 3, base: float = 0.8, cap: float = 3.0):
    for i in range(attempts):
        try:
            return fn()
        except tuple(exceptions) as e:
            if i == attempts - 1:
                raise
            sleep = min(base * (2 ** i), cap)
            time.sleep(sleep)
