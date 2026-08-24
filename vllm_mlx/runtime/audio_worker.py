"""Server-owned MLX worker dispatch and lifecycle state for audio lanes."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

_T = TypeVar("_T")


class ModelWorker(Protocol):
    """Minimal execution surface exported by an inference engine."""

    async def execute_on_model_worker(
        self, func: Callable[..., _T], *args: Any, **kwargs: Any
    ) -> _T: ...

    def execute_on_model_worker_sync(
        self, func: Callable[..., _T], *args: Any, **kwargs: Any
    ) -> _T: ...


@dataclass
class AudioLaneState:
    model: str | None = None
    state: str = "registered"
    active_requests: int = 0
    loaded_at: float | None = None
    last_used_at: float | None = None
    last_error: str | None = None


class AudioWorkerDispatcher:
    """Route audio MLX work through the server's model-owning worker."""

    def __init__(self) -> None:
        self._worker: ModelWorker | None = None
        self._lock = threading.Lock()
        self._lanes: dict[str, AudioLaneState] = {}

    def bind(self, worker: ModelWorker | None) -> None:
        if worker is not None and (
            not callable(getattr(worker, "execute_on_model_worker", None))
            or not callable(getattr(worker, "execute_on_model_worker_sync", None))
        ):
            raise TypeError("engine does not expose the model-worker contract")
        with self._lock:
            self._worker = worker

    def _bound_worker(self) -> ModelWorker | None:
        with self._lock:
            return self._worker

    def _begin(self, lane: str, model: str, operation: str) -> None:
        now = time.monotonic()
        with self._lock:
            state = self._lanes.setdefault(lane, AudioLaneState())
            state.model = model
            state.active_requests += 1
            state.state = "loading" if operation == "load" else "busy"
            state.last_used_at = now
            state.last_error = None

    def _finish(
        self, lane: str, model: str, operation: str, error: BaseException | None
    ) -> None:
        now = time.monotonic()
        with self._lock:
            state = self._lanes.setdefault(lane, AudioLaneState())
            state.model = model
            state.active_requests = max(0, state.active_requests - 1)
            state.last_used_at = now
            if error is None:
                state.state = "registered" if operation == "unload" else "resident"
                state.last_error = None
                if operation == "load":
                    state.loaded_at = now
                elif operation == "unload":
                    state.model = None
                    state.loaded_at = None
            else:
                state.state = "failed"
                state.last_error = type(error).__name__

    def snapshot(self) -> list[dict[str, object]]:
        """Return stable, secret-free lane state for the residency API."""

        now = time.monotonic()
        with self._lock:
            return [
                {
                    "lane": lane,
                    "model": state.model,
                    "state": state.state,
                    "active_requests": state.active_requests,
                    "loaded_at": state.loaded_at,
                    "idle_seconds": (
                        max(0.0, now - state.last_used_at)
                        if state.last_used_at is not None and state.active_requests == 0
                        else 0.0
                    ),
                    "last_error": state.last_error,
                }
                for lane, state in sorted(self._lanes.items())
            ]

    async def execute(
        self,
        lane: str,
        model: str,
        operation: str,
        func: Callable[..., _T],
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        self._begin(lane, model, operation)
        error: BaseException | None = None
        try:
            worker = self._bound_worker()
            if worker is None:
                return func(*args, **kwargs)
            return await worker.execute_on_model_worker(func, *args, **kwargs)
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._finish(lane, model, operation, error)

    def execute_sync(
        self,
        lane: str,
        model: str,
        operation: str,
        func: Callable[..., _T],
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        self._begin(lane, model, operation)
        error: BaseException | None = None
        try:
            worker = self._bound_worker()
            if worker is None:
                return func(*args, **kwargs)
            return worker.execute_on_model_worker_sync(func, *args, **kwargs)
        except BaseException as exc:
            error = exc
            raise
        finally:
            self._finish(lane, model, operation, error)


audio_worker = AudioWorkerDispatcher()


def bind_audio_worker(worker: ModelWorker | None) -> None:
    audio_worker.bind(worker)


async def run_audio_mlx(
    lane: str,
    model: str,
    operation: str,
    func: Callable[..., _T],
    *args: Any,
    **kwargs: Any,
) -> _T:
    return await audio_worker.execute(lane, model, operation, func, *args, **kwargs)


def run_audio_mlx_sync(
    lane: str,
    model: str,
    operation: str,
    func: Callable[..., _T],
    *args: Any,
    **kwargs: Any,
) -> _T:
    return audio_worker.execute_sync(lane, model, operation, func, *args, **kwargs)
