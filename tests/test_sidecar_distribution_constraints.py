from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from packaging.requirements import Requirement

SCRIPT = (
    Path(__file__).parents[1]
    / "apps"
    / "rapid-mac"
    / "scripts"
    / "check-sidecar-distributions.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("sidecar_constraints", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def constraints():
    return _load_module()


def test_exact_reduced_vision_runtime_is_allowed(constraints) -> None:
    assert constraints.is_validated_compatibility_exception(
        owner="mlx-vlm",
        owner_version="0.6.16",
        requirement=Requirement("transformers>=5.14.0"),
        actual="5.12.1",
    )


@pytest.mark.parametrize(
    ("owner", "owner_version", "requirement", "actual"),
    [
        ("other", "0.6.16", "transformers>=5.14.0", "5.12.1"),
        ("mlx-vlm", "0.6.15", "transformers>=5.14.0", "5.12.1"),
        ("mlx-vlm", "0.6.17", "transformers>=5.14.0", "5.12.1"),
        ("mlx-vlm", "0.6.16", "transformers>=5.13.0", "5.12.1"),
        ("mlx-vlm", "0.6.16", "other>=5.14.0", "5.12.1"),
        ("mlx-vlm", "0.6.16", "transformers>=5.14.0", "5.12.0"),
        ("mlx-vlm", "0.6.16", "transformers>=5.14.0", "5.13.1"),
    ],
)
def test_every_nearby_mismatch_still_fails_closed(
    constraints, owner: str, owner_version: str, requirement: str, actual: str
) -> None:
    assert not constraints.is_validated_compatibility_exception(
        owner=owner,
        owner_version=owner_version,
        requirement=Requirement(requirement),
        actual=actual,
    )
