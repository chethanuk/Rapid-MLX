# SPDX-License-Identifier: Apache-2.0
"""The base-wheel "needs mlx-vlm" install hint must recommend the validated
vision runtime.

Pre-fix, the vision-alias boot guard (and the DiffusionEngine import-error)
told users to install a different mlx-vlm release than the version validated
for the optional runtime and packaged app.

The fix keeps ``rapid-mlx[vision]`` as the primary suggestion and pins the
bare-mlx-vlm fallback to the same validated runtime.

These tests pin the message text at every user-facing site so a future
edit can't silently regress back to the conflict-producing hint.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


def test_vlm_extra_install_hint_is_pinned_and_conflict_free():
    """The shared vision install hint (printed by the serve boot guard and
    the engine-side ``_require_mlx_vlm``) recommends the extra first and a
    PINNED bare mlx-vlm — not the conflict-producing ``>=0.6.3``."""
    from vllm_mlx.models.mllm import VLM_EXTRA_INSTALL_HINT

    # Primary path stays the extra.
    assert "rapid-mlx[vision]" in VLM_EXTRA_INSTALL_HINT
    # Bare fallback is pinned to the transformers-compatible version.
    assert "mlx-vlm==0.6.16" in VLM_EXTRA_INSTALL_HINT
    # And the unpinned form that produces the transformers conflict is gone.
    assert "mlx-vlm>=0.6.3" not in VLM_EXTRA_INSTALL_HINT


def test_boot_guard_absent_hint_names_pinned_install(monkeypatch, capsys):
    """The ABSENT-path boot guard stderr carries the pinned hint so a user
    copy-pasting from the terminal lands in a conflict-free environment."""
    from vllm_mlx.models.mllm import VisionRuntimeStatus, require_mlx_vlm_or_exit

    monkeypatch.setattr(
        "vllm_mlx.models.mllm.vision_runtime_status",
        lambda: (VisionRuntimeStatus.ABSENT, "mlx_vlm"),
    )

    try:
        require_mlx_vlm_or_exit("gemma-4-e4b-it-4bit")
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover - guard must exit
        raise AssertionError("require_mlx_vlm_or_exit must sys.exit(2)")

    err = capsys.readouterr().err
    assert "rapid-mlx[vision]" in err
    assert "mlx-vlm==0.6.16" in err
    assert "mlx-vlm>=0.6.3" not in err


def test_gemma4_load_fallback_hint_is_pinned():
    """The Gemma-4-specific ``serve``/``chat`` load-fallback hint (printed
    when mlx-lm can't import the Gemma-4 architecture classes on a base
    wheel) must pin the bare mlx-vlm text-only install to ``==0.6.16`` too.

    Scan the CLI source so a future edit cannot silently regress this last
    user-facing hint to an unpinned or differently pinned runtime.
    """
    import pathlib

    import vllm_mlx.cli as cli_mod

    source = pathlib.Path(cli_mod.__file__).read_text()

    # The text-only footprint fallback must be pinned...
    assert "pip install --no-deps 'mlx-vlm==0.6.16'" in source, (
        "Gemma-4 load-fallback hint must pin mlx-vlm==0.6.16 to match "
        "VLM_EXTRA_INSTALL_HINT."
    )
    # ...and no CLI hint may use the conflict-producing unpinned lower bound.
    assert "mlx-vlm>=0.6.1" not in source, (
        "cli.py still recommends the unpinned 'mlx-vlm>=0.6.1' which pulls "
        "a runtime that was not validated with this rapid-mlx release."
    )


def test_gemma4_load_fallback_prints_validated_runtime(monkeypatch, capsys):
    """Execute the submit-load failure path that prints the recovery hint."""
    import concurrent.futures

    import vllm_mlx.cli as cli_mod

    class FailedLoad:
        def result(self):
            raise ValueError("Model type gemma4_unified not supported")

    class ImmediateExecutor:
        def __init__(self, **_kwargs):
            pass

        def submit(self, _function, *_args, **_kwargs):
            return FailedLoad()

        def shutdown(self, *, wait):
            assert wait is False

    monkeypatch.setattr(
        "vllm_mlx.community_bench.hardware.is_apple_silicon", lambda: True
    )
    monkeypatch.setattr(
        "vllm_mlx.model_aliases.resolve_profile",
        lambda _alias: SimpleNamespace(hf_path="org/gemma-4-test"),
    )
    monkeypatch.setattr(cli_mod, "_check_disk_space", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        cli_mod, "_check_memory_capacity", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        cli_mod, "_ensure_model_downloaded", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", ImmediateExecutor)

    args = SimpleNamespace(
        model="gemma-4-test",
        notes=None,
        force_disk_check=False,
        sampled=False,
        spec_decode="none",
        run_group=None,
        repo_root=None,
    )
    assert cli_mod._run_submit_flow(args) == 2
    out = capsys.readouterr().out
    assert "rapid-mlx[vision]" in out
    assert "pip install --no-deps 'mlx-vlm==0.6.16'" in out


def test_diffusion_lane_import_error_hint_is_pinned(monkeypatch):
    """DiffusionEngine's dependency-import failure (Gemma 4 DLM path) points
    at the extra + a PINNED mlx-vlm, dropping the old ``-U 'mlx-vlm>=0.6.3'``
    upgrade that would break the transformers pin."""
    from vllm_mlx.runtime.diffusion_lane import DiffusionEngine

    # Force ``from mlx_vlm.generate.diffusion import ...`` to fail so the
    # engine records its actionable ``_load_error`` — a None entry in
    # sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate.diffusion", None)

    eng = DiffusionEngine("mlx-community/some-diffusion-gemma")
    # ``_load_blocking`` records ``_load_error`` and then re-raises it via
    # ``_wait_until_ready`` — assert on the surfaced RuntimeError message.
    with pytest.raises(RuntimeError) as exc_info:
        eng._load_blocking()

    msg = str(exc_info.value)
    assert eng._load_error is not None
    assert "rapid-mlx[vision]" in msg
    assert "mlx-vlm==0.6.16" in msg
    assert "mlx-vlm>=0.6.3" not in msg
    # The conflict-producing forced-upgrade flag is gone.
    assert "-U 'mlx-vlm" not in msg
