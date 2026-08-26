from __future__ import annotations

import importlib.util
from pathlib import Path

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
