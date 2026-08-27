from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
_SCRIPT = _ROOT / "apps/rapid-mac/scripts/check-sidecar-smoke-cache.py"
_MANIFEST = _ROOT / "apps/rapid-mac/scripts/sidecar-smoke-models.json"
_SPEC = importlib.util.spec_from_file_location("sidecar_cache_preflight", _SCRIPT)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _populate(cache: Path, repository: str, revision: str) -> None:
    snapshot = _MODULE.snapshot_path(cache, repository, revision)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")


def test_manifest_is_the_exact_two_pin_source_of_truth() -> None:
    assert _MODULE.load_pins(_MANIFEST) == {
        "qwen": (
            "mlx-community/Qwen3.5-9B-4bit",
            "8b2b98c00a6b4d291155e4890773ca8f769aee53",
        ),
        "gemma": (
            "mlx-community/gemma-4-e2b-it-8bit",
            "03dcf209f3f549b4075e7191e77cf69b3d48e1b2",
        ),
    }


def test_cache_root_matches_hugging_face_environment_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub_cache = tmp_path / "explicit-hub"
    hf_home = tmp_path / "hf-home"
    xdg_cache = tmp_path / "xdg-cache"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_cache))
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg_cache))
    assert _MODULE.default_cache_root() == hub_cache

    monkeypatch.delenv("HF_HUB_CACHE")
    assert _MODULE.default_cache_root() == hf_home / "hub"

    monkeypatch.delenv("HF_HOME")
    assert _MODULE.default_cache_root() == xdg_cache / "huggingface" / "hub"


@pytest.mark.parametrize("missing_key", ["qwen", "gemma"])
def test_one_missing_pin_fails_and_names_exact_recovery(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], missing_key: str
) -> None:
    pins = _MODULE.load_pins(_MANIFEST)
    for key, (repository, revision) in pins.items():
        if key != missing_key:
            _populate(tmp_path, repository, revision)
    assert (
        _MODULE.main(["--manifest", str(_MANIFEST), "--cache-root", str(tmp_path)]) == 1
    )
    error = capsys.readouterr().err
    repository, revision = pins[missing_key]
    assert f"{repository}@{revision}" in error
    assert f"revision='{revision}'" in error
    assert "No download was attempted" in error


def test_both_missing_are_reported_together(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        _MODULE.main(["--manifest", str(_MANIFEST), "--cache-root", str(tmp_path)]) == 1
    )
    error = capsys.readouterr().err
    assert "2 immutable snapshot(s)" in error
    for repository, revision in _MODULE.load_pins(_MANIFEST).values():
        assert f"{repository}@{revision}" in error


def test_both_present_pass_without_reading_model_files(tmp_path: Path) -> None:
    for repository, revision in _MODULE.load_pins(_MANIFEST).values():
        _populate(tmp_path, repository, revision)
    assert (
        _MODULE.main(["--manifest", str(_MANIFEST), "--cache-root", str(tmp_path)]) == 0
    )


def test_broken_snapshot_symlink_fails_closed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pins = _MODULE.load_pins(_MANIFEST)
    for key, (repository, revision) in pins.items():
        if key == "qwen":
            snapshot = _MODULE.snapshot_path(tmp_path, repository, revision)
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").symlink_to(tmp_path / "missing-blob")
        else:
            _populate(tmp_path, repository, revision)

    assert (
        _MODULE.main(["--manifest", str(_MANIFEST), "--cache-root", str(tmp_path)]) == 1
    )
    assert f"{pins['qwen'][0]}@{pins['qwen'][1]}" in capsys.readouterr().err


def test_present_pins_emit_workflow_outputs(tmp_path: Path) -> None:
    pins = _MODULE.load_pins(_MANIFEST)
    for repository, revision in pins.values():
        _populate(tmp_path, repository, revision)
    output = tmp_path / "github-output"
    assert (
        _MODULE.main(
            [
                "--manifest",
                str(_MANIFEST),
                "--cache-root",
                str(tmp_path),
                "--github-output",
                str(output),
            ]
        )
        == 0
    )
    assert output.read_text().splitlines() == [
        f"qwen_model={pins['qwen'][0]}",
        f"qwen_revision={pins['qwen'][1]}",
        f"gemma_model={pins['gemma'][0]}",
        f"gemma_revision={pins['gemma'][1]}",
    ]


def test_malformed_manifest_fails_closed(tmp_path: Path) -> None:
    manifest = tmp_path / "pins.json"
    manifest.write_text(json.dumps({"schema": 1, "models": {}}))
    with pytest.raises(_MODULE.PreflightError, match="exactly qwen and gemma"):
        _MODULE.load_pins(manifest)


def test_repository_cannot_inject_a_workflow_output(tmp_path: Path) -> None:
    payload = json.loads(_MANIFEST.read_text())
    payload["models"]["qwen"]["repository"] = "owner/model\nevil=value"
    manifest = tmp_path / "pins.json"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(_MODULE.PreflightError, match="owner/name"):
        _MODULE.load_pins(manifest)


def test_workflow_preflight_precedes_every_expensive_command() -> None:
    workflow = (_ROOT / ".github/workflows/auto-release.yml").read_text()
    preflight = workflow.index("check-sidecar-smoke-cache.py")
    for expensive in (
        "/opt/homebrew/opt/python@3.12/bin/python3.12 -m venv",
        "tests/integrations/agent_smoke.sh",
        "apps/rapid-mac/scripts/build-sidecar.sh",
    ):
        assert preflight < workflow.index(expensive)
    assert "steps.sidecar-pins.outputs.qwen_model" in workflow
    assert "steps.sidecar-pins.outputs.gemma_revision" in workflow
