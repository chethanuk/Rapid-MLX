# SPDX-License-Identifier: Apache-2.0
"""HTTP-level contract tests for the #2358 / #2359 request-schema fixes.

Covers two behaviours on both ``/v1/chat/completions`` and
``/v1/completions``:

  * ``stop`` accepts a scalar string (OpenAI shape ``str | list[str]``)
    and is normalized once in the request schema — a bare ``"END"`` must
    parse instead of 4xx-ing as "not a list".
  * a non-positive / non-finite ``timeout`` is rejected with the unified
    ``invalid_request_error`` 400 naming the field — NOT an instant 504
    from an ``asyncio.wait_for``-style guard consuming the bad value.

The schema-level normalization / rejection logic itself is unit-tested in
``tests/test_api_models.py``; this file pins the wire behaviour through
the real routes (the 400-not-504 claim is only provable at HTTP level).

The engine is stubbed so we exercise only the Pydantic + route-layer
validators, never a real sampler.
"""

import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def patched_config():
    """Patch the global config singleton and restore on teardown."""
    from vllm_mlx.config import get_config

    cfg = get_config()
    saved: dict = {}

    def patch(**kwargs):
        for k, v in kwargs.items():
            saved.setdefault(k, getattr(cfg, k, None))
            setattr(cfg, k, v)

    yield patch

    for k, v in saved.items():
        setattr(cfg, k, v)


def _stub_engine_cfg(patch_cfg):
    engine = MagicMock()
    engine.is_mllm = False
    patch_cfg(
        engine=engine,
        model_name="stub-model",
        model_alias=None,
        model_path=None,
        model_registry=None,
        tool_call_parser=None,
        reasoning_parser=None,
        ready=True,
        api_key=None,
    )
    return engine


@pytest.fixture
def chat_client(patched_config, monkeypatch):
    from vllm_mlx.middleware.exception_handlers import install_exception_handlers
    from vllm_mlx.routes import chat as chat_route

    engine = _stub_engine_cfg(patched_config)
    monkeypatch.setattr(chat_route, "get_engine", lambda *_a, **_kw: engine)

    app = FastAPI()
    app.include_router(chat_route.router)
    install_exception_handlers(app)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def completion_client(patched_config, monkeypatch):
    from vllm_mlx.middleware.exception_handlers import install_exception_handlers
    from vllm_mlx.routes import completions as comp_route

    engine = _stub_engine_cfg(patched_config)
    monkeypatch.setattr(comp_route, "get_engine", lambda *_a, **_kw: engine)

    app = FastAPI()
    app.include_router(comp_route.router)
    install_exception_handlers(app)
    return TestClient(app, raise_server_exceptions=False)


def _chat_body(**kw) -> dict:
    body = {"model": "stub-model", "messages": [{"role": "user", "content": "hi"}]}
    body.update(kw)
    return body


def _completion_body(**kw) -> dict:
    body = {"model": "stub-model", "prompt": "Once upon a time"}
    body.update(kw)
    return body


# ---------------------------------------------------------------------------
# ``stop`` as a scalar string — must parse (no schema 4xx), not 422.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "client_name,url,body_fn",
    [("chat_client", "/v1/chat/completions", _chat_body),
     ("completion_client", "/v1/completions", _completion_body)],
)
def test_scalar_stop_accepted(request, client_name, url, body_fn):
    """A bare-string ``stop`` must be accepted by the schema on both
    OpenAI routes (pre-fix it 422'd as "not a list"). We assert it is
    NOT a client-4xx validation rejection — the route will fail later on
    the stubbed engine, but must get past request parsing."""
    client = request.getfixturevalue(client_name)
    r = client.post(url, json=body_fn(stop="END"))
    assert r.status_code != 422, f"scalar stop rejected at schema: {r.text[:200]}"
    # It must also not be a 400 (validation) — a 500/503 from the stubbed
    # engine is acceptable proof that parsing succeeded.
    assert r.status_code not in (400, 422), f"scalar stop 4xx'd: {r.text[:200]}"


@pytest.mark.parametrize(
    "client_name,url,body_fn",
    [("chat_client", "/v1/chat/completions", _chat_body),
     ("completion_client", "/v1/completions", _completion_body)],
)
def test_stop_list_still_accepted(request, client_name, url, body_fn):
    client = request.getfixturevalue(client_name)
    r = client.post(url, json=body_fn(stop=["END", "STOP"]))
    assert r.status_code not in (400, 422), f"list stop 4xx'd: {r.text[:200]}"


# ---------------------------------------------------------------------------
# non-positive / non-finite ``timeout`` → 400 (not 504)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "client_name,url,body_fn",
    [("chat_client", "/v1/chat/completions", _chat_body),
     ("completion_client", "/v1/completions", _completion_body)],
)
@pytest.mark.parametrize("bad", [0, 0.0, -1, -0.5, -1.0])
def test_nonpositive_timeout_400(request, client_name, url, body_fn, bad):
    client = request.getfixturevalue(client_name)
    r = client.post(url, json=body_fn(timeout=bad))
    assert r.status_code == 400, (
        f"expected 400 for timeout={bad}; got {r.status_code} body={r.text[:200]}"
    )
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "timeout" in body["error"]["message"]


@pytest.mark.parametrize(
    "client_name,url,body_fn",
    [("chat_client", "/v1/chat/completions", _chat_body),
     ("completion_client", "/v1/completions", _completion_body)],
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_timeout_400(request, client_name, url, body_fn, bad):
    """NaN / ±inf travel as raw JSON tokens (allow_nan), so we post the
    raw payload rather than httpx's json= channel."""
    client = request.getfixturevalue(client_name)
    payload = json.dumps(body_fn(timeout=bad))  # allow_nan=True
    r = client.post(
        url, content=payload, headers={"Content-Type": "application/json"}
    )
    assert r.status_code == 400, (
        f"expected 400 for timeout={bad!r}; got {r.status_code} body={r.text[:200]}"
    )
    body = r.json()
    assert body["error"]["type"] == "invalid_request_error"
    assert "timeout" in body["error"]["message"]


@pytest.mark.parametrize(
    "client_name,url,body_fn",
    [("chat_client", "/v1/chat/completions", _chat_body),
     ("completion_client", "/v1/completions", _completion_body)],
)
def test_positive_timeout_not_validation_4xx(request, client_name, url, body_fn):
    """A valid positive timeout must NOT 400/422 at the schema — failure
    (if any) comes from the stubbed engine, not request parsing."""
    client = request.getfixturevalue(client_name)
    r = client.post(url, json=body_fn(timeout=30.0))
    assert r.status_code not in (400, 422), f"valid timeout 4xx'd: {r.text[:200]}"
