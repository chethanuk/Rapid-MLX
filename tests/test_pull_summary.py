# SPDX-License-Identifier: Apache-2.0
"""Tests for the 0.9.7 ``rapid-mlx pull`` post-download summary line.

A ~6 GB pull that succeeds silently leaves the user wondering "did
that actually finish, and how much disk did I just burn?". The
summary line printed by ``pull_command`` answers both in one line:

    Downloaded <repo_id> — <size with units> in <duration with units>

These tests pin three things and three things only:

1. The summary line is emitted on the HuggingFace-fallback success
   path (the common case once R2 misses).
2. The summary line is emitted on the R2 mirror success path.
3. The summary line is NOT emitted when the pull fails with a 404 —
   we exit before we'd otherwise mislead the user.

The actual HuggingFace download (``snapshot_download``) and the R2
prefetch (``_try_mirror_prefetch``) are mocked; we only exercise the
summary code path in ``pull_command``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from vllm_mlx import cli


def _make_fake_snapshot(root: Path, total_bytes: int) -> Path:
    """Create a snapshot dir on disk with one file of ``total_bytes`` bytes."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.safetensors").write_bytes(b"\x00" * total_bytes)
    return root


def _looks_like_size(token: str) -> bool:
    """Loose acceptance of either SI (``GB``) or IEC (``GiB``) suffixes.

    The task spec says ``X.Y GB`` but the project's shared
    ``_format_bytes`` helper renders IEC (``GiB``); we reuse it per
    the "do not invent a new size formatter" rule, so the test
    accepts whichever the helper produces.
    """
    return any(
        unit in token
        for unit in ("B", "KB", "KiB", "MB", "MiB", "GB", "GiB", "TB", "TiB")
    )


def _summary_line(captured: str) -> str:
    for line in captured.splitlines():
        if "Downloaded" in line and "in" in line:
            return line
    raise AssertionError(f"summary line missing from stdout, got:\n{captured!r}")


def test_summary_printed_on_hf_success(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """HF-fallback path prints ``Downloaded ... — <size> in <duration>``."""
    snapshot_dir = _make_fake_snapshot(tmp_path / "snap", total_bytes=2048)

    args = argparse.Namespace(model="mlx-community/Qwen3-0.6B-4bit")

    with (
        # Not cached at entry (no resolved snapshot), so this pull actually
        # transfers bytes — the summary reports a download.
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch(
            "huggingface_hub.snapshot_download",
            return_value=str(snapshot_dir),
        ),
    ):
        cli.pull_command(args)

    out = capsys.readouterr().out
    line = _summary_line(out)

    # Model name appears verbatim.
    assert "mlx-community/Qwen3-0.6B-4bit" in line
    # Some size token with a recognized unit.
    parts = line.split()
    assert any(_looks_like_size(p) for p in parts), line
    # Some duration token ending in 's' (e.g. ``4.2s`` or ``1m 23s``).
    assert any(p.endswith("s") and p[0].isdigit() for p in parts), line


def test_summary_printed_on_mirror_success(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2-mirror success path also prints the summary line.

    We point the HF cache root at ``tmp_path`` via the ``HF_HUB_CACHE``
    constant so ``pull_command`` resolves the snapshot dir under our fixture.
    The mirror mock simulates an actual fetch: it creates ``refs/main`` and
    populates the snapshot ONLY during the pull, so no snapshot exists at
    entry (pre-pull size is unknown) and the summary reports a real download.
    """
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "abc123" * 6  # 36 hex chars; shape doesn't matter for the test

    cache_root = tmp_path / "hub"
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    def _mirror_fetch(model_name: str) -> bool:
        """Simulate the mirror downloading the snapshot during this pull."""
        repo_root = cache_root / "models--mlx-community--Qwen3-0.6B-4bit"
        refs_dir = repo_root / "refs"
        refs_dir.mkdir(parents=True, exist_ok=True)
        (refs_dir / "main").write_text(revision)
        snapshot_dir = repo_root / "snapshots" / revision
        _make_fake_snapshot(snapshot_dir, total_bytes=4096)
        return True

    args = argparse.Namespace(model=repo_id)

    with patch.object(cli, "_try_mirror_prefetch", side_effect=_mirror_fetch):
        cli.pull_command(args)

    out = capsys.readouterr().out
    line = _summary_line(out)
    assert repo_id in line
    parts = line.split()
    assert any(_looks_like_size(p) for p in parts), line
    assert any(p.endswith("s") and p[0].isdigit() for p in parts), line


def test_cached_pull_reports_verified_not_downloaded(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fully-cached pull must say the cache was verified, not ``Downloaded``.

    Issue #2349: ``rapid-mlx pull <cached-model>`` printed
    ``Downloaded ... in 3.8s`` after "[10/10] ... cached". The final outcome
    now reserves ``Downloaded`` + transfer timing for a pull that actually
    fetched bytes; an already-complete snapshot reports the cache was reused.
    """
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "abc123" * 6
    cache_root = tmp_path / "hub"
    repo_root = cache_root / "models--mlx-community--Qwen3-0.6B-4bit"
    refs_dir = repo_root / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "main").write_text(revision)
    snapshot_dir = repo_root / "snapshots" / revision
    _make_fake_snapshot(snapshot_dir, total_bytes=4096)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    args = argparse.Namespace(model=repo_id)

    # The snapshot is already complete at entry AND the mirror pulls nothing
    # new, so pre-pull and post-pull sizes match -> "verified (nothing to
    # download)", not "Downloaded".
    with patch.object(cli, "_try_mirror_prefetch", return_value=True):
        cli.pull_command(args)

    out = capsys.readouterr().out
    assert "Already cached" in out, out
    assert "Downloaded" not in out, out
    assert "verified (nothing to download)" in out, out


def test_moved_main_reported_as_download(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local snapshot may be complete while ``main`` advanced remote-side.

    The codex BLOCKING on #2349: ``is_repo_cached`` (local presence) alone is
    wrong for a mutable ``main`` — the subsequent mirror/HF call can transfer
    new files while the summary falsely says "nothing to download". The
    summary must reflect the ACTUAL transfer. Here a stale rev_A is fully
    cached pre-pull, then the mirror populates a NEWER rev_B. To prove the
    label keys on the resolved revision (not byte count), rev_B is the SAME
    total size as rev_A; the pull is nonetheless reported as a download
    because the post-pull snapshot resolves to a different directory.
    """
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    stale_rev = "aaaaaa" * 6
    head_rev = "bbbbbb" * 6
    cache_root = tmp_path / "hub"
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    repo_root = cache_root / "models--mlx-community--Qwen3-0.6B-4bit"
    refs_dir = repo_root / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    # A fully-cached STALE rev exists before the pull.
    (refs_dir / "main").write_text(stale_rev)
    _make_fake_snapshot(repo_root / "snapshots" / stale_rev, total_bytes=4096)

    def _mirror_fetch(model_name: str) -> bool:
        # Remote main advanced mid-pull: refs/main now points at a NEWER rev.
        # Same total byte count as the stale rev, so only the resolved PATH
        # (not the size) proves a transfer happened.
        (refs_dir / "main").write_text(head_rev)
        _make_fake_snapshot(repo_root / "snapshots" / head_rev, total_bytes=4096)
        return True

    args = argparse.Namespace(model=repo_id)

    with patch.object(cli, "_try_mirror_prefetch", side_effect=_mirror_fetch):
        cli.pull_command(args)

    out = capsys.readouterr().out
    assert "Downloaded" in out, out
    assert "Already cached" not in out, out


def test_summary_not_printed_on_404(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A 404 must bail before the summary — we don't lie about success.

    ``pull_command`` matches 404 via either ``RepositoryNotFoundError``
    isinstance OR a ``"404" / "not found"`` substring on the exception
    string, so a plain ``Exception("404 Client Error")`` is enough to
    drive the error branch without constructing HF's response-bound
    exception class.
    """
    args = argparse.Namespace(model="mlx-community/does-not-exist")

    with (
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch(
            "huggingface_hub.snapshot_download",
            side_effect=Exception("404 Client Error"),
        ),
        pytest.raises(SystemExit) as excinfo,
    ):
        cli.pull_command(args)

    assert excinfo.value.code == 1
    out = capsys.readouterr().out
    assert "Downloaded" not in out, out


def test_format_pull_duration_units() -> None:
    """Sub-minute keeps decimals; ``>=60s`` switches to ``m`` + ``s``."""
    assert cli._format_pull_duration(0.0) == "0.0s"
    assert cli._format_pull_duration(4.2) == "4.2s"
    assert cli._format_pull_duration(59.9) == "59.9s"
    assert cli._format_pull_duration(60.0) == "1m 0s"
    assert cli._format_pull_duration(125.0) == "2m 5s"
    # Rounding rule: 119.9s reads as 2m 0s, not 1m 59s.
    assert cli._format_pull_duration(119.9) == "2m 0s"
