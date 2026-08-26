# SPDX-License-Identifier: Apache-2.0
"""Task C (instrumentation-release) — END-TO-END caller attribution.

These tests close the gap codex flagged on the first pass of #2436: the
earlier telemetry tests called ``emit.request`` / ``normalize_caller_agent``
directly, so they stayed green even if the route wiring was deleted. These
drive the ACTUAL routes (``/v1/messages`` and ``/v1/completions``) through
``TestClient`` with telemetry opted-in + stubbed, and assert the emitted
``request`` event — proving the User-Agent reaches the payload bucketed to
the correct caller label, end to end.

This is the attribution fix the instrument-release drop depends on for the
agent-strategy decision: claude-code traffic over the Anthropic surface must
land in ``claude-code`` (not ``other``), and openai-python ride-through must
land in ``openai-python``.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ------------------------------------------------------------------ fixtures
# Mirror the opted_in + stub_queue pattern from test_telemetry_emit.py so
# telemetry is on (consent + sampling=1) and every emitted payload is
# captured in-memory instead of hitting the real queue sink.


@pytest.fixture
def telemetry_on(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("RAPID_MLX_TELEMETRY", raising=False)

    import vllm_mlx.telemetry.emit as emit
    import vllm_mlx.telemetry.state as state

    importlib.reload(state)
    importlib.reload(emit)
    emit._reset_for_tests()

    from vllm_mlx.telemetry.state import record_consent

    record_consent(True, rapid_mlx_version="0.0.0+test")
    monkeypatch.setenv("RAPID_MLX_TELEMETRY_REQUEST_SAMPLE", "1")
    return emit


@pytest.fixture
def captured(monkeypatch):
    """Capture every ``emit.request`` payload into a list."""
    from vllm_mlx.telemetry import emit

    captured: list[dict] = []

    class _StubQueue:
        def enqueue(self, payload):
            captured.append(payload)

    monkeypatch.setattr(emit, "get_queue", lambda: _StubQueue())
    return captured


def _request_events(captured) -> list[dict]:
    """Pull ``request``-type payloads ('request' is the schema descriminator
    key inside each captured envelope)."""
    events = []
    for payload in captured:
        req = payload.get("request")
        if req is not None:
            events.append(req)
    return events


async def _sleep(seconds: float) -> None:
    """Async sleep helper so fake engines can inject a real post-first-token
    gap without blocking the event loop (mirrors the fake engine in
    ``test_telemetry_streaming_request_wiring.py``)."""
    import asyncio

    await asyncio.sleep(seconds)


def _capture_request(monkeypatch, captured):
    """Capture every ``emit.request`` KWARG (not the enqueued payload) so a
    test can assert on raw values (e.g. exact ``ttft_ms``) that are otherwise
    bucketed in the wire payload. Mirrors the emit.request-capture pattern in
    ``test_telemetry_streaming_request_wiring.py``."""
    from vllm_mlx.telemetry import emit

    def _wrapper(**kw):
        captured.append(kw)

    monkeypatch.setattr(emit, "request", _wrapper)
    return captured


# ---------------------------------------------------------------- anthropic


class _AnthropicEngine:
    """Combined non-stream ``chat`` + stream ``stream_chat`` fake engine.

    Mirrors the ``_StreamingEngine`` from ``test_anthropic_route_streaming``
    and adds a non-streaming ``chat`` that returns a token-bearing output so
    the same harness drives both the ``/v1/messages`` branches.
    """

    preserve_native_tool_format = False
    tokenizer = SimpleNamespace(
        chat_template=None,
        apply_chat_template=lambda *a, **k: "templated",
        decode=lambda *a, **k: "",
        encode=lambda *a, **k: [1, 2, 3],
    )

    def __init__(self):
        self.nonstream_calls = 0
        self.stream_calls = 0

    async def chat(self, messages, **kwargs):
        self.nonstream_calls += 1
        return SimpleNamespace(
            text="hello there",
            raw_text="hello there",
            prompt_tokens=9,
            completion_tokens=7,
            finish_reason="stop",
            tool_calls=None,
            matched_stop=None,
            reasoning_text=None,
            model="test-model",
        )

    async def stream_chat(self, messages, **kwargs):
        self.stream_calls += 1
        for i, text in enumerate(["Hello ", "world"], start=1):
            if i == len(["Hello ", "world"]):
                # Post-first-token gap so total latency dwarfs true TTFT
                # (mirrors test_telemetry_streaming_request_wiring).
                await _sleep(0.2)
            yield SimpleNamespace(
                new_text=text,
                prompt_tokens=9,
                completion_tokens=i,
            )


def _anthropic_client(engine: _AnthropicEngine) -> TestClient:
    from vllm_mlx.config import reset_config
    from vllm_mlx.routes.anthropic import router

    cfg = reset_config()
    cfg.engine = engine
    cfg.model_name = "test-model"
    cfg.model_registry = None

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_anthropic_messages_nonstream_emits_claude_code_attribution(
    telemetry_on, captured
):
    """A claude-code User-Agent on a completed non-streaming ``/v1/messages``
    request must surface a bucketed ``request`` event with
    ``caller_agent == "claude-code"``, the correct endpoint, stream=False,
    token counts and a positive TTFT. This is the structural proof codex
    asked for: it would FAIL if the route's ``emit.request`` were deleted or
    the User-Agent were not threaded through."""
    engine = _AnthropicEngine()
    client = _anthropic_client(engine)

    resp = client.post(
        "/v1/messages",
        headers={"user-agent": "claude-code/1.0.3"},
        json={
            "model": "test-model",
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert resp.status_code == 200, resp.text
    events = _request_events(captured)
    assert engine.nonstream_calls == 1
    assert len(events) >= 1, f"no request telemetry emitted: {captured!r}"
    ev = events[-1]
    assert ev["endpoint"] == "/v1/messages"
    assert ev["stream"] is False
    assert ev["caller_agent"] == "claude-code"
    # Token counts are bucketed to a fixed allowlist (red-line: raw counts
    # are a soft fingerprint) — the 9/7 token counts land in "0-256".
    assert ev["prompt_tokens_bucket"] == "0-256"
    assert ev["completion_tokens_bucket"] == "0-256"
    assert ev["completion_empty"] is False
    # TTFT is bucketed (never a raw ms value) — just assert a valid label.
    assert ev["ttft_ms_bucket"] in (
        "<100ms",
        "100-500ms",
        "500-1500ms",
        "1.5-5s",
        ">5s",
    )
    assert ev["status"] == 200


def test_anthropic_messages_stream_emits_claude_code_attribution(
    telemetry_on, monkeypatch
):
    """Same contract on the streaming ``/v1/messages`` branch: the emit fires
    after the stream drains, caller_agent is the bucketed claude-code label,
    stream=True, and TTFT is TRUE first-token latency.

    Captures raw ``emit.request`` kwargs (before bucketing) while driving the
    route end-to-end; the fake engine injects a real 0.2s gap after the first
    token so total stream latency dwarfs true TTFT. Asserting ``ttft_ms`` is a
    small fraction of total proves ``_first_token_ts`` is latched and used —
    a regression to "total latency" would blow past ``0.5 * total``.
    """
    import time

    calls: list[dict] = []
    _capture_request(monkeypatch, calls)

    engine = _AnthropicEngine()
    client = _anthropic_client(engine)

    t0 = time.perf_counter()
    resp = client.post(
        "/v1/messages",
        headers={"user-agent": "Claude-Code/2.0 (macOS) Anthropic/API"},
        json={
            "model": "test-model",
            "max_tokens": 2048,
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    total_ms = (time.perf_counter() - t0) * 1000.0
    assert resp.status_code == 200, resp.text
    assert "text/event-stream" in resp.headers["content-type"]
    assert engine.stream_calls == 1
    assert len(calls) == 1, f"expected one request emit, got {calls!r}"
    kw = calls[0]
    assert kw["endpoint"] == "/v1/messages"
    assert kw["stream"] is True
    assert kw["status"] == 200
    assert kw["caller_agent"] == "Claude-Code/2.0 (macOS) Anthropic/API"
    assert kw["prompt_tokens"] == 9
    assert kw["completion_tokens"] == 2
    # TTFT is measured at the FIRST token, so it is a small fraction of total
    # latency (which includes the 0.2s post-first-token gap) — NOT total.
    assert kw["ttft_ms"] > 0.0
    assert kw["ttft_ms"] < 0.5 * total_ms, (
        f"ttft_ms={kw['ttft_ms']:.1f} should be well under half of total "
        f"latency {total_ms:.1f}ms — it must reflect first-token timing, not total"
    )


# ---------------------------------------------------------------- completions


def _completions_client(monkeypatch):
    """Drive ``/v1/completions`` end-to-end with a fake engine, mirroring the
    proven harness from ``test_completions_log_redaction`` (engine + admission
    + context-length shims so the route completes without loading a model)."""
    from vllm_mlx.routes import completions as completions_mod

    fake_engine = MagicMock()

    async def _finish(*_a, **_k):
        return SimpleNamespace(
            text="done",
            finish_reason="stop",
            completion_tokens=5,
            prompt_tokens=3,
            cached_tokens=0,
        )

    fake_engine.generate = AsyncMock(side_effect=_finish)

    async def _stream_finish(*_a, **_k):
        # First chunk: non-empty content, not finished.
        yield SimpleNamespace(
            new_text="done",
            finished=False,
            finish_reason=None,
            completion_tokens=0,
            prompt_tokens=3,
        )
        # Post-first-token gap so total latency dwarfs true TTFT.
        await _sleep(0.2)
        # Final chunk: finished, carries the engine's final usage.
        yield SimpleNamespace(
            new_text="",
            finished=True,
            finish_reason="stop",
            completion_tokens=5,
            prompt_tokens=3,
        )

    fake_engine.stream_generate = _stream_finish

    monkeypatch.setattr(completions_mod, "get_engine", lambda _name: fake_engine)
    monkeypatch.setattr(completions_mod, "_check_admission_or_503", lambda _e: None)
    monkeypatch.setattr(
        completions_mod, "_release_admission_unless_committed", lambda *a, **k: None
    )
    monkeypatch.setattr(
        completions_mod, "enforce_context_length_for_prompt", lambda *a, **k: None
    )
    monkeypatch.setattr(completions_mod, "_validate_model_name", lambda _m: None)
    monkeypatch.setattr(completions_mod, "_resolve_model_name", lambda m: m)
    monkeypatch.setattr(completions_mod, "_resolve_max_tokens", lambda m: m or 16)
    monkeypatch.setattr(completions_mod, "_resolve_temperature", lambda t: t)
    monkeypatch.setattr(completions_mod, "_resolve_top_p", lambda p: p)
    monkeypatch.setattr(
        completions_mod, "build_extended_sampling_kwargs", lambda _r: {}
    )

    async def _passthrough(coro, *_a, **_k):
        return await coro

    monkeypatch.setattr(completions_mod, "_wait_with_disconnect", _passthrough)

    with (
        patch("vllm_mlx.middleware.auth.verify_api_key", new=lambda *a, **k: None),
        patch("vllm_mlx.middleware.auth.check_rate_limit", new=lambda *a, **k: None),
    ):
        app = FastAPI()
        app.include_router(completions_mod.router)
        return TestClient(app)


def test_completions_nonstream_emits_openai_python_attribution(
    telemetry_on, captured, monkeypatch
):
    """A completed non-streaming ``/v1/completions`` with an openai-python
    User-Agent surfaces ``caller_agent == "openai-python"``, endpoint
    ``/v1/completions``, stream=False, and the engine's token counts."""
    client = _completions_client(monkeypatch)
    resp = client.post(
        "/v1/completions",
        headers={"user-agent": "OpenAI/Python 1.30.1"},
        json={"model": "test-model", "prompt": "hi", "max_tokens": 8},
    )
    assert resp.status_code == 200, resp.text
    events = _request_events(captured)
    assert len(events) >= 1, f"no request telemetry emitted: {captured!r}"
    ev = events[-1]
    assert ev["endpoint"] == "/v1/completions"
    assert ev["stream"] is False
    assert ev["caller_agent"] == "openai-python"
    # engine's 3 prompt + 5 completion tokens → "0-256" bucket
    assert ev["prompt_tokens_bucket"] == "0-256"
    assert ev["completion_tokens_bucket"] == "0-256"
    assert ev["ttft_ms_bucket"] in (
        "<100ms",
        "100-500ms",
        "500-1500ms",
        "1.5-5s",
        ">5s",
    )
    assert ev["status"] == 200


def test_completions_stream_emits_openai_python_attribution(telemetry_on, monkeypatch):
    """The streaming ``/v1/completions`` branch emits stream=True with TRUE
    first-token TTFT and final-usage token counts.

    Captures raw ``emit.request`` kwargs; the fake stream injects a real 0.2s
    gap after the first chunk so total latency dwarfs true TTFT. Asserting
    ``ttft_ms`` is a small fraction of total proves ``_first_token_ts`` is
    latched and used (a total-latency regression would blow past 0.5*total).
    """
    import time

    calls: list[dict] = []
    _capture_request(monkeypatch, calls)

    client = _completions_client(monkeypatch)
    t0 = time.perf_counter()
    resp = client.post(
        "/v1/completions",
        headers={"user-agent": "OpenAI/Python 1.30.1"},
        json={"model": "test-model", "prompt": "hi", "max_tokens": 8, "stream": True},
    )
    total_ms = (time.perf_counter() - t0) * 1000.0
    assert resp.status_code == 200, resp.text
    assert len(calls) == 1, f"expected one request emit, got {calls!r}"
    kw = calls[0]
    assert kw["endpoint"] == "/v1/completions"
    assert kw["stream"] is True
    assert kw["status"] == 200
    assert kw["caller_agent"] == "OpenAI/Python 1.30.1"
    assert kw["prompt_tokens"] == 3
    assert kw["completion_tokens"] == 5
    assert kw["ttft_ms"] > 0.0
    assert kw["ttft_ms"] < 0.5 * total_ms, (
        f"ttft_ms={kw['ttft_ms']:.1f} should be well under half of total "
        f"latency {total_ms:.1f}ms — it must reflect first-token timing, not total"
    )
