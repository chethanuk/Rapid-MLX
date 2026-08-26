#!/usr/bin/env python3
"""Fail closed when a release candidate lacks its curated note inputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

if __package__:
    from scripts.strip_release_note_comments import strip_release_note_comments
else:
    # Workflows intentionally execute this file by path. In that mode Python
    # places scripts/ (not the repository root) on sys.path, so import the
    # sibling directly. Keep both entrypoints on the same implementation.
    from strip_release_note_comments import strip_release_note_comments

VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-rc[1-9][0-9]*)?$")
CHANGELOG_HEADING_RE = re.compile(r"^## \[([^]]+)\](?:\s|$)")


def check_release_notes(version: str, changelog: Path, notes_dir: Path) -> None:
    if not VERSION_RE.fullmatch(version):
        raise ValueError(f"invalid release version: {version!r}")

    changelog_lines = changelog.read_text(encoding="utf-8").splitlines()
    section_start = next(
        (
            index + 1
            for index, line in enumerate(changelog_lines)
            if (match := CHANGELOG_HEADING_RE.match(line)) and match.group(1) == version
        ),
        None,
    )
    if section_start is None:
        raise ValueError(f"{changelog} has no exact '## [{version}]' section")

    section_end = next(
        (
            index
            for index in range(section_start, len(changelog_lines))
            if changelog_lines[index].startswith("## ")
        ),
        len(changelog_lines),
    )

    notes = notes_dir / f"v{version}.md"
    if not notes.is_file():
        raise ValueError(f"release notes are missing: {notes}")

    def visible(markdown: str) -> bool:
        return bool(strip_release_note_comments(markdown).strip())

    section = "\n".join(changelog_lines[section_start:section_end])
    if not visible(section):
        raise ValueError(f"{changelog} has an empty '## [{version}]' section")
    if not visible(notes.read_text(encoding="utf-8")):
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
