"""Shared utilities for NLU model engines (inference budget, device helpers)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Callable, TypeVar

T = TypeVar("T")


def run_with_budget_ms(budget_ms: int, fn: Callable[[], T], *, on_timeout: T) -> T:
    """Run ``fn`` in a worker thread; return ``on_timeout`` if it does not finish in time.

    Best-effort wall-clock bound (CUDA work may continue after timeout in the worker).
    ``budget_ms <= 0`` means no threading timeout — run ``fn`` inline.
    """
    if budget_ms <= 0:
        return fn()

    timeout_s = budget_ms / 1000.0
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(fn)
        try:
            return fut.result(timeout=timeout_s)
        except FuturesTimeoutError:
            return on_timeout
