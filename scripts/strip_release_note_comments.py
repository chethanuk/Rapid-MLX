#!/usr/bin/env python3
"""Remove drafting HTML comments without interpreting Markdown as regexes.

The publisher and its preflight share this byte-order state machine.  It walks
each line left-to-right, carries an HTML-comment state across line boundaries,
and deliberately leaves fenced and indented code untouched.  An unmatched
comment opener discards the remaining tail so it cannot hide generated release
content in GitHub's renderer.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _fence(line: str) -> tuple[str, int] | None:
    stripped = line.lstrip(" ")
    if len(line) - len(stripped) > 3 or not stripped:
        return None
    marker = stripped[0]
    if marker not in ("`", "~"):
        return None
    length = 0
    while length < len(stripped) and stripped[length] == marker:
        length += 1
    return (marker, length) if length >= 3 else None


def _is_fence_close(line: str, marker: str, minimum: int) -> bool:
    found = _fence(line)
    if found is None or found[0] != marker or found[1] < minimum:
        return False
    return line.lstrip(" ")[found[1] :].strip() == ""


def _only_markdown_containers(text: str) -> bool:
    """Return true for an empty stack of blockquote/list container markers."""

    value = text.strip("\r\n")
    index = 0
    while index < len(value) and value[index] == " " and index < 3:
        index += 1

    consumed = False
    while True:
        while index < len(value) and value[index].isspace():
            index += 1
        if index == len(value):
            return consumed
        if value[index] == ">":
            consumed = True
            index += 1
            continue
        if value[index] in "-+*":
            next_index = index + 1
            if next_index < len(value) and value[next_index].isspace():
                consumed = True
                index = next_index
                continue
            return False
        if value[index].isdigit():
            next_index = index
            while next_index < len(value) and value[next_index].isdigit():
                next_index += 1
            if (
                next_index < len(value)
                and value[next_index] in ".)"
                and next_index + 1 < len(value)
                and value[next_index + 1].isspace()
            ):
                consumed = True
                index = next_index + 1
                continue
        return False


def strip_release_note_comments(markdown: str) -> str:
    output: list[str] = []
    in_comment = False
    fence_marker = ""
    fence_length = 0

    for line in markdown.splitlines(keepends=True):
        if fence_marker:
            output.append(line)
            if _is_fence_close(line, fence_marker, fence_length):
                fence_marker = ""
                fence_length = 0
            continue

        if not in_comment and not line.startswith(("    ", "\t")):
            found_fence = _fence(line)
            if found_fence is not None:
                fence_marker, fence_length = found_fence
                output.append(line)
                continue

        if not in_comment and line.startswith(("    ", "\t")):
            output.append(line)
            continue

        cursor = 0
        visible: list[str] = []
        removed_comment = in_comment
        while cursor < len(line):
            if in_comment:
                close = line.find("-->", cursor)
                if close == -1:
                    cursor = len(line)
                    break
                in_comment = False
                removed_comment = True
                cursor = close + 3
                continue

            opener = line.find("<!--", cursor)
            if opener == -1:
                visible.append(line[cursor:])
                break
            visible.append(line[cursor:opener])
            in_comment = True
            removed_comment = True
            cursor = opener + 4

        rendered = "".join(visible)
        if removed_comment and _only_markdown_containers(rendered):
            if line.endswith("\n"):
                output.append("\n")
        else:
            output.append(rendered)

    return "".join(output)


def main(argv: list[str] | None = None) -> int:
    paths = [Path(value) for value in (argv if argv is not None else sys.argv[1:])]
    if not paths:
        sys.stdout.write(strip_release_note_comments(sys.stdin.read()))
        return 0
    for path in paths:
        sys.stdout.write(strip_release_note_comments(path.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by shell contracts
    raise SystemExit(main())
