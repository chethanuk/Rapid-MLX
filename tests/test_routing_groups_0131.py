"""Routing-group fixes for 0.13.1 (fix/0131-routing-groups).

FIX 1 — ``qwen4_exp`` (Qwen3.8-Flash-Next) visual-config checkpoints must stay
on the vendored TEXT lane rather than the mlx-vlm MLLM lane, both when served
through a curated ``is_text_only`` alias and when served from an unaliased
local path (via ``_VENDORED_TEXT_FALLBACK_MODEL_TYPES``).

FIX 3 — a requested speculative decoder (``requested_spec_decode != none``) must
slide a vision-capable checkpoint back onto the text lane (reason
``text_lane_speculative_decode``) instead of silently being dropped by the MLLM
lane, which never consumes ``scheduler_config.spec_decode``.

These tests fake the installed mlx-vlm version to >= 0.6.16 (the Desktop sidecar
pin) and assert the resulting lane decision for the Qwen3.8-Flash-Next config
shape returned by the alias artifact's ``config.json`` (``model_type=qwen4_exp``,
``vision_config`` present, ``image_token_id`` present, ``language_model_only==
false``).
"""

from pathlib import Path

from vllm_mlx.api import utils as utils_mod
from vllm_mlx.api.utils import (
    _VENDORED_TEXT_FALLBACK_MODEL_TYPES,
    mllm_arch_unsupported_but_text_vendored,
    resolve_serving_lane_decision,
)
from vllm_mlx.model_metadata import ModelMetadata
from vllm_mlx.model_profile import ModelProfile


def _flash_next_config() -> dict:
    """The exact config shape the Qwen3.8-Flash-Next-4bit checkpoint ships."""
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "vision_config": {"hidden_size": 1024},
        "image_token_id": 248056,
        "language_model_only": False,
    }


def _fake_mlx_vlm_ge_016(monkeypatch):
    """Fake the installed mlx-vlm as a recent version that can drive the
    vision arch — i.e. the scenario where the alias would otherwise be routed
    into the mlx-vlm MLLM lane (Desktop sidecar pins mlx-vlm 0.6.16)."""
    import importlib.metadata as md

    _orig_version = md.version
    monkeypatch.setattr(
        md,
        "version",
        lambda name: "0.6.17" if name == "mlx-vlm" else _orig_version(name),
    )
    # Ensure the MLLM runtime is "supported" under the faked version so the
    # only thing keeping the model off the MLLM lane is our fix.
    monkeypatch.setattr(utils_mod, "mllm_hybrid_runtime_supported", lambda: True)


def _metadata(config: dict, snapshot_dir: Path) -> ModelMetadata:
    return ModelMetadata(
        config=config,
        chat_template=None,
        snapshot_dir=snapshot_dir,
        is_local=False,
    )


def test_fix1_unaliased_qwen4_exp_vendored_goes_text_lane(monkeypatch, tmp_path):
    """FIX 1: an UNALIASED local qwen4_exp checkpoint (no curated alias profile)
    whose vision config + real vision weights would otherwise route to MLLM must
    stay on the vendored text lane once ``qwen4_exp`` is in
    ``_VENDORED_TEXT_FALLBACK_MODEL_TYPES``."""
    _fake_mlx_vlm_ge_016(monkeypatch)
    assert "qwen4_exp" in _VENDORED_TEXT_FALLBACK_MODEL_TYPES

    # No curated alias for this raw path.
    monkeypatch.setattr(utils_mod, "resolve_profile", lambda name: None)
    monkeypatch.setattr(
        utils_mod,
        "read_model_metadata",
        lambda name: _metadata(_flash_next_config(), tmp_path),
    )
    # Positive multimodal weight evidence — the vendored-text refire is what
    # must pull it back to the text lane, not an inconclusive verdict.
    monkeypatch.setattr(
        utils_mod,
        "checkpoint_has_multimodal_weights",
        lambda snapshot, config: True,
    )

    # Without the fix the raw mllm model check is True and mlx-vlm has no
    # qwen4_exp package, so the vendored fallback must fire.
    assert mllm_arch_unsupported_but_text_vendored("/some/local/qwen4_exp") is True
    decision = resolve_serving_lane_decision("/some/local/qwen4_exp")
    assert decision.is_mllm is False
    assert decision.auto_text_fallback is True
    assert decision.reason == "vision_architecture_unavailable"


def test_fix1_curated_text_alias_goes_text_lane(monkeypatch, tmp_path):
    """FIX 1: the curated qwen3.8-flash-next-4bit alias pins ``is_text_only`` so
    it stays text even against real vision weights and a >= 0.6.16 mlx-vlm."""
    _fake_mlx_vlm_ge_016(monkeypatch)
    curated = ModelProfile(
        hf_path="mlx-community/Qwen3.8-Flash-Next-4bit",
        is_text_only=True,
    )
    monkeypatch.setattr(utils_mod, "resolve_profile", lambda name: curated)
    monkeypatch.setattr(
        utils_mod,
        "read_model_metadata",
        lambda name: _metadata(_flash_next_config(), tmp_path),
    )
    monkeypatch.setattr(
        utils_mod,
        "checkpoint_has_multimodal_weights",
        lambda snapshot, config: True,
    )
    decision = resolve_serving_lane_decision("qwen3.8-flash-next-4bit")
    assert decision.is_mllm is False
    assert decision.reason == "text_checkpoint"


def test_fix2_text_diffusion_resolves_to_assistant_replacement_group():
    """FIX 2: a text-diffusion profile (e.g. diffusion-gemma-26b-4bit) must
    resolve to the SAME replacement group as resident_models._replacement_group
    derives for the text engine ("assistant"), so loading it with a chat model
    resident does not trip the resolved_group != replace_group 409 guard."""
    from vllm_mlx.routes.residency import _resolved_group_for_profile
    from vllm_mlx.runtime.resident_models import ModelEntry, _replacement_group

    # Request-facing profile modality → group, matching the Fix 2 mapping.
    assert _resolved_group_for_profile("text-diffusion") == "assistant"
    assert _resolved_group_for_profile("text") == "assistant"
    assert _resolved_group_for_profile("vision") == "assistant"
    assert _resolved_group_for_profile("image-gen") == "image-gen"
    assert _resolved_group_for_profile("video-gen") == "video-gen"

    # Engine-derivation parity: a text-diffusion engine is a text engine.
    entry = ModelEntry(
        engine=_MockEngine("text"),
        model_name="diffusion-gemma-26b-4bit",
        model_path="diffusion-gemma-26b-4bit",
    )
    assert _replacement_group(entry) == "assistant"


class _MockEngine:
    """Minimal engine stub for resident_models._replacement_group."""

    def __init__(self, modality: str):
        self._modality = modality

    @property
    def is_image_gen(self) -> bool:
        return self._modality == "image-gen"

    @property
    def is_video_gen(self) -> bool:
        return self._modality == "video-gen"

    @property
    def is_mllm(self) -> bool:
        return self._modality == "mllm"


def test_fix3_spec_decode_requested_slides_vision_capable_to_text(
    monkeypatch,
    tmp_path,
):
    """FIX 3: requesting MTP spec-decode on a vision-capable (non-vendored)
    checkpoint must route to the TEXT lane (reason
    ``text_lane_speculative_decode``) so the decoder is honoured instead of
    silently dropped by the MLLM lane, which never consumes
    ``scheduler_config.spec_decode``."""
    _fake_mlx_vlm_ge_016(monkeypatch)
    # A genuinely MLLM-routable vision-capable checkpoint (NOT the qwen4_exp
    # vendored-fallback shape, which Fix 1 already pins to text): the Qwen3.5-4B
    # vision-config shape used by the existing vision-evidence tests.
    monkeypatch.setattr(utils_mod, "resolve_profile", lambda name: None)
    monkeypatch.setattr(
        utils_mod,
        "read_model_metadata",
        lambda name: _metadata(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "vision_config": {"hidden_size": 1024},
                "image_token_id": 248056,
            },
            tmp_path,
        ),
    )
    monkeypatch.setattr(
        utils_mod,
        "checkpoint_has_multimodal_weights",
        lambda snapshot, config: True,
    )

    # Without a spec-decode request the model is vision-capable -> MLLM lane.
    assert resolve_serving_lane_decision("/local/qwen35-vision").is_mllm is True
    # With a requested decoder -> forced text lane with the fix-3 reason.
    decision = resolve_serving_lane_decision(
        "/local/qwen35-vision", requested_spec_decode="mtp"
    )
    assert decision.is_mllm is False
    assert decision.reason == "text_lane_speculative_decode"
    assert decision.auto_text_fallback is True

    # Spec-decode wins over an explicit --mllm: the MLLM lane can never honour
    # the decoder, so forcing vision must not silently drop it.
    with_mllm = resolve_serving_lane_decision(
        "/local/qwen35-vision",
        force_mllm=True,
        requested_spec_decode="mtp",
    )
    assert with_mllm.is_mllm is False
    assert with_mllm.reason == "text_lane_speculative_decode"

    # An explicit --mllm WITHOUT a speculative request still forces vision.
    vision_only = resolve_serving_lane_decision("/local/qwen35-vision", force_mllm=True)
    assert vision_only.is_mllm is True
    assert vision_only.reason == "vision_lane_forced"
