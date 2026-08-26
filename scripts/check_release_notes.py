#!/usr/bin/env python3
"""Fail closed when a release candidate lacks its curated note inputs."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-rc[1-9][0-9]*)?$")
CHANGELOG_HEADING_RE = re.compile(r"^## \[([^]]+)\](?:\s|$)")


def check_release_notes(version: str, changelog: Path, notes_dir: Path) -> None:
    if not VERSION_RE.fullmatch(version):
        raise ValueError(f"invalid release version: {version!r}")

    headings = {
        match.group(1)
        for line in changelog.read_text(encoding="utf-8").splitlines()
        if (match := CHANGELOG_HEADING_RE.match(line))
    }
    if version not in headings:
        raise ValueError(f"{changelog} has no exact '## [{version}]' section")

    notes = notes_dir / f"v{version}.md"
    if not notes.is_file():
        raise ValueError(f"release notes are missing: {notes}")
    normalizer = Path(__file__).with_name("strip_release_note_comments.awk")
    normalized = subprocess.run(
        ["awk", "-f", str(normalizer), str(notes)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if not normalized.strip():
        raise ValueError(f"release notes are empty: {notes}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--changelog", type=Path, required=True)
    parser.add_argument("--notes-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        check_release_notes(args.version, args.changelog, args.notes_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"release notes preflight passed for {args.version}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
