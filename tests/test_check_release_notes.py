import sys
from pathlib import Path

import pytest

from scripts.check_release_notes import check_release_notes, main


def _inputs(tmp_path: Path, version: str = "0.13.1") -> tuple[Path, Path]:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        f"# Changelog\n\n## [{version}] — 2026-08-26\n\nVisible change.\n",
        encoding="utf-8",
    )
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / f"v{version}.md").write_text("Release highlights.\n", encoding="utf-8")
    return changelog, notes_dir


@pytest.mark.parametrize("version", ["0.13.1", "0.13.1-rc1", "12.0.3-rc20"])
def test_accepts_complete_stable_and_rc_inputs(tmp_path: Path, version: str) -> None:
    changelog, notes_dir = _inputs(tmp_path, version)
    check_release_notes(version, changelog, notes_dir)


def test_rejects_missing_exact_changelog_section(tmp_path: Path) -> None:
    changelog, notes_dir = _inputs(tmp_path)
    changelog.write_text("## [0.13.10]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no exact"):
        check_release_notes("0.13.1", changelog, notes_dir)


def test_rejects_empty_exact_changelog_section(tmp_path: Path) -> None:
    changelog, notes_dir = _inputs(tmp_path)
    changelog.write_text(
        "## [0.13.1] — 2026-08-26\n\n<!-- draft -->\n\n## [0.13.0]\nold\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="empty '## \\[0.13.1\\]'"):
        check_release_notes("0.13.1", changelog, notes_dir)


def test_rejects_missing_version_bound_notes_file(tmp_path: Path) -> None:
    changelog, notes_dir = _inputs(tmp_path)
    (notes_dir / "v0.13.1.md").rename(notes_dir / "v0.13.2.md")
    with pytest.raises(ValueError, match="missing"):
        check_release_notes("0.13.1", changelog, notes_dir)


def test_rejects_empty_notes(tmp_path: Path) -> None:
    changelog, notes_dir = _inputs(tmp_path)
    (notes_dir / "v0.13.1.md").write_text(" \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        check_release_notes("0.13.1", changelog, notes_dir)


def test_rejects_notes_containing_only_stripped_drafting_comments(
    tmp_path: Path,
) -> None:
    changelog, notes_dir = _inputs(tmp_path)
    (notes_dir / "v0.13.1.md").write_text(
        "<!-- drafting guidance\nthat does not ship -->\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="empty"):
        check_release_notes("0.13.1", changelog, notes_dir)


@pytest.mark.parametrize("version", ["0.13", "0.13.1-rc0", "../0.13.1", "v0.13.1"])
def test_rejects_invalid_or_unsafe_version(tmp_path: Path, version: str) -> None:
    changelog, notes_dir = _inputs(tmp_path)
    with pytest.raises(ValueError, match="invalid release version"):
        check_release_notes(version, changelog, notes_dir)


def test_cli_success_from_script_entrypoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    changelog, notes_dir = _inputs(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_release_notes.py",
            "--version",
            "0.13.1",
            "--changelog",
            str(changelog),
            "--notes-dir",
            str(notes_dir),
        ],
    )
    assert main() == 0
    assert "preflight passed for 0.13.1" in capsys.readouterr().out


def test_cli_failure_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    changelog, notes_dir = _inputs(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_release_notes.py",
            "--version",
            "0.13.2",
            "--changelog",
            str(changelog),
            "--notes-dir",
            str(notes_dir),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
