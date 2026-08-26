from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "apps/rapid-mac/scripts/smoke-sidecar-vision.py"
_SPEC = importlib.util.spec_from_file_location("sidecar_vision_smoke", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_sensible_completion_accepts_known_fixture_descriptions() -> None:
    assert _MODULE._completion_is_sensible("A spotted cheetah cub runs forward.")
    assert _MODULE._completion_is_sensible("The image shows a playful feline mascot.")


def test_sensible_completion_rejects_empty_error_or_unrelated_output() -> None:
    assert not _MODULE._completion_is_sensible(None)
    assert not _MODULE._completion_is_sensible("")
    assert not _MODULE._completion_is_sensible("Internal server error")
    assert not _MODULE._completion_is_sensible("A blue square is visible.")
    assert not _MODULE._completion_is_sensible("I cannot identify the animal.")
    assert not _MODULE._completion_is_sensible("I cannot identify the cat.")
    assert not _MODULE._completion_is_sensible(
        "The animal might be a cat, but I am unsure."
    )


def test_release_workflow_runs_content_addressed_real_image_gate() -> None:
    workflow = (
        Path(__file__).parents[1] / ".github/workflows/auto-release.yml"
    ).read_text()
    assert "SIDECAR_VISION_SMOKE_MODEL: mlx-community/Qwen3.5-9B-4bit" in workflow
    assert (
        "SIDECAR_VISION_SMOKE_REVISION: 8b2b98c00a6b4d291155e4890773ca8f769aee53"
        in workflow
    )
    assert "SIDECAR_GEMMA_SMOKE_MODEL: mlx-community/gemma-4-e2b-it-8bit" in workflow
    assert (
        "SIDECAR_GEMMA_SMOKE_REVISION: 03dcf209f3f549b4075e7191e77cf69b3d48e1b2"
        in workflow
    )
    assert "HF_HUB_OFFLINE=1 bash apps/rapid-mac/scripts/build-sidecar.sh" in workflow
    assert '"$SIDE/python/bin/python3.12"' in workflow
    assert '--model "$SIDECAR_GEMMA_SMOKE_MODEL"' in workflow
    assert "apps/rapid-mac/scripts/build-sidecar.sh" in workflow


def test_repository_model_without_revision_fails_closed() -> None:
    with pytest.raises(SystemExit, match="requires --revision"):
        _MODULE._resolve_model("owner/model", None)
