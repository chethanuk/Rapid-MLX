"""Contracts for server-owned auxiliary audio execution."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading

import pytest

from vllm_mlx.engine.batched import BatchedEngine
from vllm_mlx.runtime.audio_worker import AudioWorkerDispatcher


class _RecordingWorker:
    def __init__(self) -> None:
        self.async_calls: list[tuple[object, tuple, dict]] = []
        self.sync_calls: list[tuple[object, tuple, dict]] = []

    async def execute_on_model_worker(self, func, *args, **kwargs):
        self.async_calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    def execute_on_model_worker_sync(self, func, *args, **kwargs):
        self.sync_calls.append((func, args, kwargs))
        return func(*args, **kwargs)


@pytest.mark.asyncio
async def test_audio_only_dispatch_uses_dedicated_worker_without_primary():
    dispatcher = AudioWorkerDispatcher()
    caller_thread = threading.get_ident()

    assert (
        await dispatcher.execute("stt", "whisper", "infer", threading.get_ident)
        != caller_thread
    )
    assert (
        dispatcher.execute_sync("tts", "kokoro", "infer", threading.get_ident)
        != caller_thread
    )
    dispatcher.bind(None)


@pytest.mark.asyncio
async def test_bound_dispatch_uses_public_worker_contract():
    dispatcher = AudioWorkerDispatcher()
    worker = _RecordingWorker()
    dispatcher.bind(worker)

    assert (
        await dispatcher.execute("stt", "whisper", "infer", lambda value: value + 1, 4)
        == 5
    )
    assert (
        dispatcher.execute_sync(
            "tts", "kokoro", "infer", lambda *, value: value + 1, value=6
        )
        == 7
    )
    assert len(worker.async_calls) == 1
    assert len(worker.sync_calls) == 1


def test_bind_rejects_engine_without_complete_worker_contract():
    dispatcher = AudioWorkerDispatcher()

    with pytest.raises(TypeError, match="model-worker contract"):
        dispatcher.bind(object())


def test_server_selects_isolated_fallback_for_non_batched_engine():
    from vllm_mlx import server

    assert server._bind_audio_worker_for_engine(object()) is False


@pytest.mark.asyncio
async def test_unbind_restores_dedicated_audio_only_worker():
    dispatcher = AudioWorkerDispatcher()
    dispatcher.bind(_RecordingWorker())
    dispatcher.bind(None)

    caller_thread = threading.get_ident()
    assert (
        await dispatcher.execute("stt", "whisper", "infer", threading.get_ident)
        != caller_thread
    )
    dispatcher.bind(None)


@pytest.mark.asyncio
async def test_batched_engine_async_dispatch_uses_owning_executor_thread():
    owner = object.__new__(BatchedEngine)
    caller_thread = threading.get_ident()
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="test-model-worker"
    )
    owner._model_load_executor = executor
    try:
        worker_thread = await owner.execute_on_model_worker(threading.get_ident)
    finally:
        executor.shutdown(wait=True)

    assert worker_thread != caller_thread


def test_batched_engine_sync_dispatch_uses_owning_executor_thread():
    owner = object.__new__(BatchedEngine)
    caller_thread = threading.get_ident()
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="test-model-worker"
    )
    owner._model_load_executor = executor
    try:
        worker_thread = owner.execute_on_model_worker_sync(threading.get_ident)
    finally:
        executor.shutdown(wait=True)

    assert worker_thread != caller_thread


@pytest.mark.asyncio
async def test_batched_engine_rejects_async_dispatch_when_stopped():
    owner = object.__new__(BatchedEngine)
    owner._model_load_executor = None

    with pytest.raises(RuntimeError, match="model worker is not running"):
        await owner.execute_on_model_worker(lambda: None)


def test_batched_engine_rejects_sync_dispatch_when_stopped():
    owner = object.__new__(BatchedEngine)
    owner._model_load_executor = None

    with pytest.raises(RuntimeError, match="model worker is not running"):
        owner.execute_on_model_worker_sync(lambda: None)


@pytest.mark.asyncio
async def test_lane_snapshot_reports_resident_model_after_successful_load():
    dispatcher = AudioWorkerDispatcher()

    await dispatcher.execute("stt", "whisper-small", "load", lambda: None)

    assert dispatcher.snapshot() == [
        {
            "lane": "stt",
            "model": "whisper-small",
            "state": "resident",
            "active_requests": 0,
            "loaded_at": dispatcher.snapshot()[0]["loaded_at"],
            "idle_seconds": pytest.approx(0.0, abs=0.1),
            "last_error": None,
        }
    ]
    dispatcher.bind(None)


def test_lane_snapshot_records_failure_without_leaking_message():
    dispatcher = AudioWorkerDispatcher()

    with pytest.raises(ValueError, match="secret detail"):
        dispatcher.execute_sync(
            "tts",
            "kokoro",
            "load",
            lambda: (_ for _ in ()).throw(ValueError("secret detail")),
        )

    lane = dispatcher.snapshot()[0]
    assert lane["state"] == "failed"
    assert lane["active_requests"] == 0
    assert lane["last_error"] == "ValueError"
    assert "secret detail" not in repr(lane)
    dispatcher.bind(None)


@pytest.mark.asyncio
async def test_cancellation_drains_worker_before_releasing_lane_lease():
    dispatcher = AudioWorkerDispatcher()
    started = threading.Event()
    release = threading.Event()

    def blocking_transcription() -> str:
        started.set()
        assert release.wait(timeout=5)
        return "done"

    task = asyncio.create_task(
        dispatcher.execute("stt", "whisper", "infer", blocking_transcription)
    )
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    await asyncio.sleep(0.05)

    assert not task.done()
    assert dispatcher.snapshot()[0]["active_requests"] == 1

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert dispatcher.snapshot()[0]["active_requests"] == 0
    dispatcher.bind(None)


@pytest.mark.asyncio
async def test_server_shutdown_unloads_cached_audio_engines(monkeypatch):
    from vllm_mlx.routes import audio as audio_route

    class _Cached:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name
            self.unloaded = False

        def unload(self) -> None:
            self.unloaded = True

    stt = _Cached("whisper")
    aligner = _Cached("aligner")
    tts = _Cached("kokoro")
    monkeypatch.setattr(audio_route, "_stt_engine", stt)
    monkeypatch.setattr(audio_route, "_aligner_engine", aligner)
    monkeypatch.setattr(audio_route, "_tts_engine", tts)
    monkeypatch.setattr(audio_route, "_music_engine", object())

    await audio_route.shutdown_audio_lanes()

    assert stt.unloaded and aligner.unloaded and tts.unloaded
    assert audio_route._stt_engine is None
    assert audio_route._aligner_engine is None
    assert audio_route._tts_engine is None
    assert audio_route._music_engine is None
