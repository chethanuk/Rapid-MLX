# SPDX-License-Identifier: Apache-2.0
"""Live speculative configuration vs request-level fallback contract."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("method", [None, "", "none", " NONE ", 7])
def test_disabled_speculative_methods_have_no_live_policy(method):
    from vllm_mlx.speculative.request_policy import (
        resolve_speculative_request_policy,
    )

    assert resolve_speculative_request_policy(method) is None


def test_mtp_policy_reports_tools_as_safe_ordinary_decode_fallback():
    from vllm_mlx.speculative.request_policy import (
        resolve_speculative_request_policy,
    )

    policy = resolve_speculative_request_policy(" MTP ")
    assert policy is not None
    assert policy.method == "mtp"
    assert policy.request_fallback_features == ("tools",)


def test_other_speculative_methods_do_not_inherit_mtp_tool_policy():
    from vllm_mlx.speculative.request_policy import (
        resolve_speculative_request_policy,
    )

    policy = resolve_speculative_request_policy("suffix")
    assert policy is not None
    assert policy.method == "suffix"
    assert policy.request_fallback_features == ()


def test_model_profile_reads_policy_from_matching_live_scheduler(monkeypatch):
    from vllm_mlx.routes import models as models_route

    scheduler = SimpleNamespace(config=SimpleNamespace(spec_decode="mtp"))
    engine = object()
    monkeypatch.setattr(models_route, "_engine_for", lambda _model_id: engine)
    monkeypatch.setattr(models_route, "_scheduler_of", lambda candidate: scheduler)

    info = models_route._resolve_speculative_decoding("served-model")

    assert info is not None
    assert info.configured is True
    assert info.method == "mtp"
    assert info.request_fallback_features == ["tools"]
    assert info.model_dump() == {
        "configured": True,
        "method": "mtp",
        "request_fallback_features": ["tools"],
    }


def test_model_card_carries_live_speculative_policy(monkeypatch):
    from vllm_mlx.api.models import SpeculativeDecodingInfo
    from vllm_mlx.routes import models as models_route

    expected = SpeculativeDecodingInfo(
        configured=True,
        method="mtp",
        request_fallback_features=["tools"],
    )
    monkeypatch.setattr(
        models_route, "_resolve_speculative_decoding", lambda _model_id: expected
    )
    monkeypatch.setattr(models_route, "_resolve_context_window", lambda _model_id: None)
    monkeypatch.setattr(
        models_route,
        "_resolve_max_model_len",
        lambda _model_id, _native_context: None,
    )
    monkeypatch.setattr(models_route, "_audio_lane_snapshot", lambda: None)
    monkeypatch.setattr(
        models_route, "_served_lane_fields", lambda _model_id: (None, None)
    )
    monkeypatch.setattr(models_route, "_resolve_audio_entry", lambda _model_id: None)
    monkeypatch.setattr(models_route, "_locked_embedding_id", lambda: None)
    monkeypatch.setattr(
        models_route, "_reported_hybrid", lambda _model_id, static: static
    )
    monkeypatch.setattr(
        models_route,
        "effective_parsers_for",
        lambda _model_id, tool, reasoning: (tool, reasoning),
    )
    monkeypatch.setattr(
        models_route, "_detect_capabilities", lambda *_args, **_kwargs: ["text"]
    )
    monkeypatch.setattr(
        models_route,
        "_reported_modality",
        lambda _model_id, modality, _is_text_only=False: modality,
    )

    info = models_route._build_model_info("qwen3.8-27b-4bit")

    assert info.speculative_decoding == expected
    assert info.model_dump()["speculative_decoding"] == expected.model_dump()


@pytest.mark.parametrize("engine,scheduler", [(None, object()), (object(), None)])
def test_model_profile_never_advertises_unattached_runtime(
    monkeypatch, engine, scheduler
):
    from vllm_mlx.routes import models as models_route

    monkeypatch.setattr(models_route, "_engine_for", lambda _model_id: engine)
    monkeypatch.setattr(models_route, "_scheduler_of", lambda _engine: scheduler)

    assert models_route._resolve_speculative_decoding("unloaded-model") is None
