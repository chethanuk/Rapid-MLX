# SPDX-License-Identifier: Apache-2.0
"""Atomic rollback admission for composite generation caches."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def can_trim(cache: Any, n: int) -> bool:
    """Return whether ``cache.trim(n)`` can commit without partial mutation."""
    if n < 0:
        return False
    children = getattr(cache, "caches", None)
    if children is not None:
        return all(can_trim(child, n) for child in children)
    amount_check = getattr(cache, "can_trim", None)
    if callable(amount_check):
        return bool(amount_check(n))
    can_undo = getattr(cache, "_can_undo", None)
    if callable(can_undo) and can_undo(n):
        return True
    check = getattr(cache, "is_trimmable", None)
    return bool(callable(check) and check())


def trim_all(caches: Iterable[Any], n: int) -> bool:
    """Preflight every cache before mutating any member of the collection."""
    if n <= 0:
        return n == 0
    cache_list = list(caches)
    if not all(can_trim(cache, n) for cache in cache_list):
        return False
    return all(cache.trim(n) == n for cache in cache_list)
