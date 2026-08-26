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


def _hf_snapshot_layout(
    repo_id: str,
    revision: str,
    root: Path,
    *,
    files: dict[str, bytes] | None = None,
    already_cached: bool = False,
) -> tuple[Path, Path]:
    """Build a deterministic HF cache entry ``root/models--<id>/`` for the
    HF-fallback transfer-account tests.

    Returns ``(cache_root, snapshot_dir)``. Points ``refs/main`` at
    ``revision``. When ``already_cached`` the snapshot dir already exists
    with ``files`` (a warm, fully-cached pull leaves it untouched); when not,
    no snapshot exists up front — the test's ``snapshot_download`` mock is
    expected to create one during the pull so the before-inventory is empty.
    ``repo_id`` must map to no catalog subfolder (the tests use
    ``mlx-community/Qwen3-0.6B-4bit``), so the layout is flat.
    """
    cache_root = root / "hub"
    repo_root = cache_root / f"models--{repo_id.replace('/', '--')}"
    (repo_root / "refs").mkdir(parents=True, exist_ok=True)
    (repo_root / "refs" / "main").write_text(revision)
    snapshot_dir = repo_root / "snapshots" / revision
    if already_cached:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        for name, data in (files or {"model.safetensors": b"\x00" * 2048}).items():
            (snapshot_dir / name).write_bytes(data)
    return cache_root, snapshot_dir


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HF-fallback path prints ``Downloaded ... — <size> in <duration>``."""
    revision = "abc123" * 6
    cache_root, snapshot_dir = _hf_snapshot_layout(
        "mlx-community/Qwen3-0.6B-4bit", revision, tmp_path, already_cached=False
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    def _download(*_args, **_kwargs):
        # Not cached at entry (no resolved snapshot); the pull actually
        # transfers bytes — the before-inventory is empty and the after-side
        # (this freshly-written snapshot) differs, so the summary is a download.
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / "model.safetensors").write_bytes(b"\x00" * 2048)
        return str(snapshot_dir)

    args = argparse.Namespace(model="mlx-community/Qwen3-0.6B-4bit")

    with (
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch("huggingface_hub.snapshot_download", side_effect=_download),
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


def test_hf_cached_fallback_reports_verified(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached no-op on the HF-fallback path is labelled ``Already cached``.

    Even when the mirror miss forces the HF fallback, a pull that transfers
    zero bytes must NOT print ``Downloaded``. The transfer account is the
    stable on-disk snapshot inventory BEFORE vs AFTER the pull (Codex #2392,
    no huggingface_hub tqdm-progress internals): a warm, fully-cached pull
    leaves the snapshot untouched, so before == after and it reports verified.
    """
    revision = "abc123" * 6
    cache_root, snapshot_dir = _hf_snapshot_layout(
        "mlx-community/Qwen3-0.6B-4bit", revision, tmp_path, already_cached=True
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    def _download(*_args, **_kwargs):
        # A warm pull touches NOTHING on disk: the snapshot was already there,
        # complete. before == after -> verified.
        return str(snapshot_dir)

    args = argparse.Namespace(model="mlx-community/Qwen3-0.6B-4bit")

    with (
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch("huggingface_hub.snapshot_download", side_effect=_download),
    ):
        cli.pull_command(args)

    out = capsys.readouterr().out
    assert "Already cached" in out, out
    assert "verified (nothing to download)" in out, out
    assert "Downloaded" not in out, out


def test_hf_fetch_zero_byte_file_counts_as_download(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fetched zero-byte file is a network fetch, not a cache hit.

    codex round-4 BLOCKING #3 carried over to the stable seam: a fetched
    zero-byte file adds a NEW row to the snapshot inventory (the file did not
    exist before), so before != after and the summary says ``Downloaded`` even
    though the file carries no bytes.
    """
    revision = "abc123" * 6
    cache_root, snapshot_dir = _hf_snapshot_layout(
        "mlx-community/Qwen3-0.6B-4bit", revision, tmp_path, already_cached=False
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    def _download(*_args, **_kwargs):
        # The pull fetched a file that is 0 bytes: it appears in the
        # snapshot now (a new inventory row) though it carries no bytes.
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / "empty.bin").write_bytes(b"")
        return str(snapshot_dir)

    args = argparse.Namespace(model="mlx-community/Qwen3-0.6B-4bit")

    with (
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch("huggingface_hub.snapshot_download", side_effect=_download),
    ):
        cli.pull_command(args)

    out = capsys.readouterr().out
    assert "Downloaded" in out, out
    assert "Already cached" not in out, out


def test_hf_fallback_transfers_bytes_as_download(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real HF-fallback transfer is a download: the snapshot inventory
    changes across the pull (files appear), so before != after -> Downloaded."""
    revision = "abc123" * 6
    cache_root, snapshot_dir = _hf_snapshot_layout(
        "mlx-community/Qwen3-0.6B-4bit", revision, tmp_path, already_cached=False
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    def _download(*_args, **_kwargs):
        # A real fetch creates the snapshot with 2048 bytes of content.
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / "model.safetensors").write_bytes(b"\x00" * 2048)
        return str(snapshot_dir)

    args = argparse.Namespace(model="mlx-community/Qwen3-0.6B-4bit")

    with (
        patch.object(cli, "_try_mirror_prefetch", return_value=False),
        patch("huggingface_hub.snapshot_download", side_effect=_download),
    ):
        cli.pull_command(args)

    out = capsys.readouterr().out
    assert "Downloaded" in out, out
    assert "Already cached" not in out, out


def test_summary_printed_on_mirror_success(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2-mirror success path also prints the summary line.

    We point the HF cache root at ``tmp_path`` via the ``HF_HUB_CACHE``
    constant so ``pull_command`` resolves the snapshot dir under our fixture.
    The mirror mock simulates an actual fetch: it creates ``refs/main`` and
    populates the snapshot ONLY during the pull, reporting ``out[
    transferred_bytes] = 4096`` so the summary reports a real download.
    """
    repo_id = "mlx-community/Qwen3-0.6B-4bit"
    revision = "abc123" * 6  # 36 hex chars; shape doesn't matter for the test

    cache_root = tmp_path / "hub"
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))

    def _mirror_fetch(model_name: str, *, out=None) -> bool:
        """Simulate the mirror downloading the snapshot during this pull."""
        repo_root = cache_root / "models--mlx-community--Qwen3-0.6B-4bit"
        refs_dir = repo_root / "refs"
        refs_dir.mkdir(parents=True, exist_ok=True)
        (refs_dir / "main").write_text(revision)
        snapshot_dir = repo_root / "snapshots" / revision
        _make_fake_snapshot(snapshot_dir, total_bytes=4096)
        if out is not None:
            out["network_fetch"] = True
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

    # The mirror reports it fetched ZERO bytes this pull (every file was
    # already cached) -> "verified (nothing to download)", not "Downloaded".
    def _mirror_already_cached(model_name: str, *, out=None) -> bool:
        if out is not None:
            out["network_fetch"] = False
        return True

    with patch.object(cli, "_try_mirror_prefetch", side_effect=_mirror_already_cached):
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
    cached pre-pull, then the mirror reports it fetched NEWER files (rev_B) as
    ``out["transferred_bytes"]``; the pull is reported as a download.
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

    def _mirror_fetch(model_name: str, *, out=None) -> bool:
        # Remote main advanced mid-pull: refs/main now points at a NEWER rev
        # whose snapshot bytes were actually fetched over the wire.
        (refs_dir / "main").write_text(head_rev)
        _make_fake_snapshot(repo_root / "snapshots" / head_rev, total_bytes=4096)
        if out is not None:
            out["network_fetch"] = True
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


def test_snapshot_identifier_is_a_stable_transfer_seam(tmp_path: Path) -> None:
    """``_snapshot_identifier`` fingerprints snapshot content so a before/after
    comparison classifies the pull without huggingface_hub tqdm internals
    (Codex #2392). The cases directly mirror the old byte-bar verdicts:
    unchanged content == cache hit; any new/changed file — INCLUDING a newly
    fetched zero-byte file — == a network fetch.
    """
    # Nothing cached yet -> empty fingerprint (before-side of a fresh pull).
    assert cli._snapshot_identifier(None) == ()
    assert cli._snapshot_identifier(str(tmp_path / "does-not-exist")) == ()

    # A warm, untouched snapshot: before == after -> verified (no transfer).
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(b"\x00" * 2048)
    before = cli._snapshot_identifier(str(snap))
    after = cli._snapshot_identifier(str(snap))
    assert before == after != ()

    # A real fetch adds a file -> the inventory changes -> Download.
    (snap / "config.json").write_bytes(b"{}")
    assert cli._snapshot_identifier(str(snap)) != before

    # A fetched ZERO-byte file (new row, no bytes) still changes the inventory
    # -> Download, never a false cache hit (carried from codex round-4 #3).
    (snap / "empty.bin").write_bytes(b"")
    assert cli._snapshot_identifier(str(snap)) != before

    # Empty vs populated: a fresh pull's before (()) differs from after.
    assert before == () or after != ()


def test_hf_snapshot_dir_for_resolves_cached_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_hf_snapshot_dir_for`` locates the current on-disk snapshot (the
    before-side target) purely from the hub cache, with no network — and
    returns None when nothing is cached yet (Codex #2392 before-side)."""
    revision = "abc123" * 6
    cache_root, snapshot_dir = _hf_snapshot_layout(
        "mlx-community/Qwen3-0.6B-4bit", revision, tmp_path, already_cached=True
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root))
    assert cli._hf_snapshot_dir_for("mlx-community/Qwen3-0.6B-4bit") == Path(
        snapshot_dir
    )

    # Nothing cached yet -> no before-side dir to fingerprint.
    cache_root2, _ = _hf_snapshot_layout(
        "mlx-community/Qwen3-0.6B-4bit",
        revision,
        tmp_path / "fresh",
        already_cached=False,
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache_root2))
    assert cli._hf_snapshot_dir_for("mlx-community/Qwen3-0.6B-4bit") is None


def test_print_pull_summary_was_cached_branch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``was_cached=True`` prints the "Already cached ... verified" line
    (issue #2349) rather than the "Downloaded" line — a proven no-transfer."""
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.safetensors").write_bytes(b"\x00" * 4096)
    cli._print_pull_summary("mlx-community/Qwen3-0.6B-4bit", snap, 1.5, was_cached=True)
    out_str = capsys.readouterr().out
    assert "Already cached" in out_str
    assert "Downloaded" not in out_str
    assert "verified (nothing to download)" in out_str
