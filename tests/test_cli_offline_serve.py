# SPDX-License-Identifier: Apache-2.0
"""Tests for offline-serve refusal when a model is uncached (#2357).

``rapid-mlx serve <uncached-model>`` under ``HF_HUB_OFFLINE=1`` /
``TRANSFORMERS_OFFLINE=1`` used to fall through every download attempt (each
printing "First-time download" / "Pre-download skipped; server will retry"),
let the serve subprocess start, and end in misleading ``--mllm``/``--no-mllm``
lane advice even though neither flag can supply the missing checkpoint.

The fix (in ``_ensure_model_downloaded``) detects the offline + uncached
condition ONCE, states which repository is missing, points to ``rapid-mlx
pull`` and the expected cache location, and exits(1) before server
initialization — mirroring the TimeoutError / disk-space exits. A lane override
is never recommended when the checkpoint is simply absent.
"""

from __future__ import annotations

import pytest

from vllm_mlx import cli


def _uncached_probe(monkeypatch):
    """Force the cache probes to report "not cached" (weights absent)."""
    import vllm_mlx._download_gate as gate

    monkeypatch.setattr(gate, "is_repo_cached", lambda name: False)
    monkeypatch.setattr(
        gate, "mflux_missing_weights", lambda name: ["model.safetensors"]
    )


def test_offline_hub_mode_detects_env_switches(monkeypatch):
    """Both offline switches flip the helper on; any truthy value counts."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert cli._offline_hub_mode_active() is True

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "true")
    assert cli._offline_hub_mode_active() is True

    # Both absent -> online.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    assert cli._offline_hub_mode_active() is False


def test_offline_uncached_serve_refuses_before_download(monkeypatch, capsys):
    """Offline + uncached must exit(1) with one actionable message and NOT
    attempt the download/mirror path (no repeated "First-time download")."""
    _uncached_probe(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(cli.os.path, "exists", lambda p: False)

    sentinel = []

    def _fail(*a, **k):
        sentinel.append(a)
        raise AssertionError("download/mirror path must not be reached offline")

    for name in ("_check_disk_space", "_try_mirror_prefetch"):
        monkeypatch.setattr(cli, name, _fail)

    with pytest.raises(SystemExit) as exc:
        cli._ensure_model_downloaded("badorg/offline-missing-model")
    assert exc.value.code == 1

    out = capsys.readouterr()
    assert "badorg/offline-missing-model is not cached" in out.err
    assert "network is unavailable (offline mode is enabled)" in out.err
    assert "rapid-mlx pull badorg/offline-missing-model" in out.err
    assert "cache location" in out.err
    # No repeated download phase, no "server will retry".
    assert "server will retry" not in out.err
    assert "First-time download" not in out.err
    assert sentinel == []  # neither disk-space nor mirror was attempted


def test_offline_refusal_counts_transformer_offline_too(monkeypatch, capsys):
    _uncached_probe(monkeypatch)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(cli.os.path, "exists", lambda p: False)

    with pytest.raises(SystemExit) as exc:
        cli._ensure_model_downloaded("badorg/offline-missing-model")
    assert exc.value.code == 1
    assert "not cached and the network is unavailable" in capsys.readouterr().err


def test_offline_local_path_is_noop_not_refused(monkeypatch, capsys):
    """A local path is a no-op even under offline mode — never refused."""
    import tempfile

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with tempfile.TemporaryDirectory() as d:
        cli._ensure_model_downloaded(d)  # os.path.exists -> early return
    assert "is not cached" not in capsys.readouterr().err


def test_online_uncached_still_attempts_download(monkeypatch, capsys):
    """Without offline switches, an uncached model still proceeds to the
    download path (no refusal) — connectivity may be available."""
    _uncached_probe(monkeypatch)
    monkeypatch.setenv("HF_HUB_OFFLINE", "")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "")
    monkeypatch.setattr(cli.os.path, "exists", lambda p: False)

    # The download path reaches _check_disk_space first; swallow it so the
    # only assertion is that we did NOT hard-refuse for offline.
    monkeypatch.setattr(
        cli, "_check_disk_space", lambda *a, **k: (_ for _ in ()).throw(StopIteration())
    )
    with pytest.raises(StopIteration):
        cli._ensure_model_downloaded("badorg/offline-missing-model")
    assert "is not cached and the network is unavailable" not in capsys.readouterr().err
