#!/usr/bin/env python3
"""Exercise the converter's FAIL-CLOSED guards on the synthetic fixture.

Each case must raise with a clear message and must NOT publish output.
  * existing output dir            -> reject
  * missing index.json             -> reject
  * missing source shard (bad map) -> reject
  * Extreme SSD path               -> reject
  * below free-space floor         -> reject
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen38_streaming_convert import convert, GUARD_EXPERT_SSD  # noqa: E402

FIXTURES = Path("/tmp/synth-guard-fixtures")


def _fails(label: str, fn) -> tuple[bool, str]:
    try:
        fn()
        return False, "NO EXCEPTION"
    except Exception as exc:  # noqa: BLE001
        return True, str(exc)


def _empty_out(tag: str) -> Path:
    root = FIXTURES / tag
    if root.exists():
        shutil.rmtree(root)
    out = root / "out"
    out.mkdir(parents=True)
    return root


def _conv_bad_index(src: Path, out: Path, mutate) -> None:
    """Run convert on a throwaway copy of src with a mutated / missing index."""
    copy = FIXTURES / "bad-index" / "src"
    if copy.exists():
        shutil.rmtree(FIXTURES / "bad-index")
    shutil.copytree(src, copy)
    mutate(copy)
    convert(copy, out, max_shard_bytes=2_000_000)


def main(src: Path) -> int:
    src = src.resolve()
    if FIXTURES.exists():
        shutil.rmtree(FIXTURES)
    cases: list[tuple[str, bool, str]] = []

    # 1 existing output dir
    out1 = _empty_out("existing-out")
    (out1 / "sentinel").write_text("x")
    ok, msg = _fails("existing-out", lambda: convert(src, out1 / "out", max_shard_bytes=2_000_000))
    cases.append(("existing-output-dir", ok, msg))

    # 2 missing index
    def _no_index(copy: Path):
        (copy / "model.safetensors.index.json").unlink()

    ok, msg = _fails("missing-index", lambda: _conv_bad_index(src, out1 / "x", _no_index))
    cases.append(("missing-index", ok, msg))

    # 3 missing source shard
    def _missing_shard(copy: Path):
        idx = json.loads((copy / "model.safetensors.index.json").read_text())
        first = next(iter(idx["weight_map"]))
        idx["weight_map"][first] = "does-not-exist.safetensors"
        (copy / "model.safetensors.index.json").write_text(json.dumps(idx))

    ok, msg = _fails("missing-shard", lambda: _conv_bad_index(src, out1 / "y", _missing_shard))
    cases.append(("missing-source-shard", ok, msg))

    # 4 Extreme SSD
    ok, msg = _fails(
        "extreme-ssd",
        lambda: convert(Path(GUARD_EXPERT_SSD + "/src"), Path("/tmp/x/out"), max_shard_bytes=2_000_000),
    )
    cases.append(("extreme-ssd", ok, msg))

    # 5 free-space floor (absurd floor must reject)
    ok, msg = _fails(
        "free-space-floor",
        lambda: convert(
            src, out1 / "z", max_shard_bytes=2_000_000, min_free_bytes=10**30
        ),
    )
    cases.append(("free-space-floor", ok, msg))

    for label, ok, msg in cases:
        print(f"  [{'ok' if ok else 'FAIL'}] {label}"
              + ("" if ok else f": {msg}"))
    bad = [c for c in cases if not c[1]]
    if bad:
        print("\nFAILED GUARDS:", ", ".join(c[0] for c in bad))
        return 1
    print("\nALL FAIL-CLOSED GUARDS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
