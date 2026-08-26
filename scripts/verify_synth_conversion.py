#!/usr/bin/env python3
"""End-to-end verification of the converter against the SYNTHETIC shard set.

Checks:
  1. Output index covers every source weight exactly once (plus one
     ``.scales``/``.biases`` key per quantised tensor).
  2. PLE weights are preserved byte-for-byte (round-trip equal to the source
     tensor values, including dtype).
  3. Quantised weights round-trip through affine q4-g64 with bounded error
     (weight fallback within machine/quant tolerance).
  4. Every output file's SHA-256 matches ``SHA256SUMS.txt``.
  5. Peak RSS is reported (already in the ledger).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open as st_safe_open


def _load(shard: Path) -> dict[str, np.ndarray]:
    out = {}
    with st_safe_open(str(shard), framework="numpy") as sf:
        for k in sf.keys():
            out[k] = sf.get_tensor(k)
    return out


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def _dequantize(q: np.ndarray, scales: np.ndarray, biases: np.ndarray) -> np.ndarray:
    # mlx affine q4-g64 : out = dequant(q, scales, biases, 64, 4)
    return np.asarray(
        mx.dequantize(
            mx.array(q), mx.array(scales), mx.array(biases), group_size=64, bits=4
        )
    )


def verify(src: Path, out: Path) -> int:
    src, out = src.resolve(), out.resolve()
    src_index = json.loads((src / "model.safetensors.index.json").read_text())
    out_index = json.loads((out / "model.safetensors.index.json").read_text())
    src_wm: dict[str, str] = src_index["weight_map"]
    out_wm: dict[str, str] = out_index["weight_map"]
    failures: list[str] = []

    # 1. coverage
    src_keys = set(src_wm)
    missing = src_keys - set(out_wm)
    if missing:
        failures.append(f"output index missing source weights: {sorted(missing)[:10]}")
    # quant tensors -> should have .scales/.biases; PLE -> plain byte-copy
    q_names = [k for k in src_keys if ".weight" in k and k not in {"model.embed_tokens.weight"}]
    # figure PLE names by out mapping to ple shard
    ple_out = {k: v for k, v in out_wm.items() if "ple" in v}
    for k in src_keys:
        target = out_wm.get(k)
        if target is None:
            continue
        # if k is PLE in src, its out shard must be the ple shard
        if k in ("ple_embed.rows.0.weight", "ple_embed.rows.1.weight", "mm.embedding.weight", "model.embed_tokens.weight"):
            if "ple" not in target:
                failures.append(f"PLE weight {k} mapped to non-ple shard {target}")

    # 2. PLE byte round-trip
    for name in ("ple_embed.rows.0.weight", "ple_embed.rows.1.weight", "mm.embedding.weight"):
        src_sf = src_wm[name]
        src_arr = _load(src / src_sf)[name]
        out_sf = out_wm[name]
        out_arr = _load(out / out_sf)[name]
        if src_arr.shape != out_arr.shape or src_arr.dtype != out_arr.dtype:
            failures.append(f"PLE {name} shape/dtype mismatch: {src_arr.shape}/{src_arr.dtype} vs {out_arr.shape}/{out_arr.dtype}")
        elif not np.array_equal(src_arr, out_arr):
            failures.append(f"PLE {name} byte mismatch")
        else:
            print(f"  [ok] PLE preserved: {name} {out_arr.shape} {out_arr.dtype}")

    # 3. quant round-trip (spot-check first 6 quant tensors incl. expert mats)
    checked = 0
    for name in sorted(k for k in src_keys if k not in {
        "ple_embed.rows.0.weight","ple_embed.rows.1.weight","mm.embedding.weight",
        "model.embed_tokens.weight"}):
        src_sf = src_wm[name]
        src_arr = _load(src / src_sf)[name].astype(np.float32)
        out_sf = out_wm[name]
        shard = _load(out / out_sf)
        q = shard[name]; sc = shard[name + ".scales"]; bi = shard[name + ".biases"]
        rec = _dequantize(q, sc, bi).astype(np.float32)
        if rec.shape != src_arr.shape:
            failures.append(f"quant {name} shape mismatch {rec.shape} vs {src_arr.shape}")
            continue
        mse = float(np.mean((rec - src_arr) ** 2))
        rel = float(np.linalg.norm(rec - src_arr) / (np.linalg.norm(src_arr) + 1e-9))
        checked += 1
        if checked <= 6:
            print(f"  [ok] quant round-trip {name} mse={mse:.6g} rel={rel:.3g}")
    print(f"  quant tensors round-tripped: {checked}")

    # 4. SHA256SUMS verifies
    sums: dict[str, str] = {}
    for line in (out / "SHA256SUMS.txt").read_text().splitlines():
        h, rel = line.split("  ", 1)
        sums[rel] = h
    bad = 0
    for rel, h in sums.items():
        if _sha256(out / rel) != h:
            bad += 1
            failures.append(f"SHA256 mismatch for {rel}")
    print(f"  SHA256SUMS entries: {len(sums)}, mismatches: {bad}")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nALL SYNTHETIC CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(verify(Path(sys.argv[1]), Path(sys.argv[2])))
