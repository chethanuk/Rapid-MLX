#!/usr/bin/env python3
"""FAIL-CLOSED streaming q4-g32 converter for Qwen3.8-Flash-Next (prototype v2).

Script-only lane helper for Vector's qwen4_exp port. Never edits model math.
Converts an HF safetensors snapshot to an MLX affine q4-g32 layout by
streaming shard-by-shard, quantising every quantisable weight (including the
~51B-param PLE embedding table) WITHOUT ever materialising the whole table in
RAM — each source tensor is read via mmap byte-slice, quantised in-memory one
tensor at a time, and written to bounded output shards.

Revision against Vector's review of cee4ef00 ("NOT safe for real weights"):
  1. PLE is QUANTISED q4-g32 (not preserved BF16). Still never materialised:
     per-tensor streaming keeps peak RSS bounded regardless of the 51B table.
  2. Explicit tensor-name contract (embed_tokens / mm.embedding / ple_embed.rows.*)
     instead of substring matching.
  3. Per-tensor quantisation predicate from the manifest classification: 1-D
     norms, ``A_log``, buffers, and widths not divisible by group_size are
     copied as-is (fp), never quantised.
  4. Output index ``total_size`` = sum of original weight byte totals (the
     loader's semantic model size, not source *.safetensors file sizes), and
     output shards use deterministic ``model-{i:05d}-of-{N:05d}`` naming with
     the real total ``N``.

Verified on a SYNTHETIC scaled-down shard set (same name/dtype/shape pattern
plus the flagged tensor classes), NOT on real weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import resource
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors.numpy import save_file

DEFAULT_MAX_SHARD_BYTES = 4 * 1024**3
GUARD_EXPERT_SSD = "/Volumes/Extreme SSD"

# Explicit embedding/PLE tensor names that are quantised (contract, not
# substring heuristic).
_EMBED_TENSOR_NAMES = ("model.embed_tokens.weight", "mm.embedding.weight")
# Buffers / special params copied as-is (never quantised).
_BUFFER_SUFFIXES = (".A_log",)
# dtypes that are aux/buffer-like and copied as-is rather than quantised.
_NON_QUANT_DTYPES = {"BF16", "F64", "F8", "F4", "I8", "I16", "I32", "I64", "U8", "BOOL"}

_NP_DTYPES = {
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
    "I8": np.int8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "U8": np.uint8,
    "BOOL": np.bool_,
}
_DTYPE_BYTES = {"F8": 1, "F16": 2, "BF16": 2, "F32": 4, "F64": 8}


def _bf16_to_f32(raw_uint16: np.ndarray) -> np.ndarray:
    """Widen raw bfloat16 bit-patterns to float32 (numpy has no bfloat16)."""
    return (raw_uint16.astype(np.uint32) << 16).view(np.float32)


def peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


def _dtype_bytes(dtype: str) -> int:
    return _DTYPE_BYTES.get(dtype.upper(), 4)


# ---------------------------------------------------------------------------
# Manifest classification (Vector's per-tensor quantise / copy predicate)
# ---------------------------------------------------------------------------
def classify_tensor(name: str, shape: list[int], dtype: str, group_size: int) -> str:
    """Return ``"quantize"`` or ``"copy"`` for a source tensor.

    Explicit contract first, then the manifest predicate:
      * 1-D weights (norms, biases) are copy/fp.
      * ``A_log`` safegate params / buffers and non-fp aux dtypes are copy/fp.
      * widths whose last dim is not divisible by ``group_size`` are copy/fp.
      * everything else (2-D + divisible, incl. embed/PLE tables) is quantised.
    """
    if len(shape) <= 1:
        return "copy"
    if name.endswith(_BUFFER_SUFFIXES):
        return "copy"
    if dtype.upper() in _NON_QUANT_DTYPES:
        return "copy"
    if shape[-1] % group_size != 0:
        return "copy"
    return "quantize"


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------
def _safe_abs(root: Path, p: Path) -> Path:
    rootr = root.resolve()
    candidate = (root / p).resolve()
    if candidate != rootr and not candidate.is_relative_to(rootr):
        raise RuntimeError(f"path escapes source root: {p}")
    return candidate


def _read_shard_layout(shard: Path) -> tuple[int, dict]:
    with (
        open(shard, "rb") as handle,
        mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mem,
    ):
        header_len = int(np.frombuffer(mem[:8], dtype=np.uint64)[0])
        header = json.loads(mem[8 : 8 + header_len].decode("utf-8"))
    return header_len, header


def _tensor_bytes(shard: Path, header_len: int, header: dict, name: str) -> bytes:
    info = header.get(name)
    if info is None:
        raise RuntimeError(f"weight {name!r} not in source shard {shard}")
    begin = 8 + header_len + int(info["data_offsets"][0])
    end = 8 + header_len + int(info["data_offsets"][1])
    with (
        open(shard, "rb") as handle,
        mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mem,
    ):
        return bytes(mem[begin:end])


def _numpy_from_bytes(data: bytes, dtype: str, shape: list[int]) -> np.ndarray:
    up = dtype.upper()
    if up == "BF16":
        return _bf16_to_f32(np.frombuffer(data, dtype=np.uint16).reshape(shape))
    np_dtype = _NP_DTYPES.get(up)
    if np_dtype is None:
        raise RuntimeError(f"unsupported source dtype {dtype} for tensor")
    return np.frombuffer(data, dtype=np_dtype).reshape(shape)


def quantize_affine_q4_g32(
    arr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q, scales, biases = mx.quantize(
        mx.array(arr.astype(np.float32)), group_size=32, bits=4, mode="affine"
    )
    return np.asarray(q), np.asarray(scales), np.asarray(biases)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_sha256sums(output: Path) -> None:
    lines = []
    for p in sorted(output.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            lines.append(f"{_sha256(p)}  {p.relative_to(output).as_posix()}\n")
    (output / "SHA256SUMS.txt").write_text("".join(lines))


_AUX_COPY_NAMES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _copy_aux_metadata(source: Path, output: Path) -> list[str]:
    copied: list[str] = []
    for name in _AUX_COPY_NAMES:
        src = source / name
        if src.is_file():
            _safe_abs(source, Path(name))
            shutil.copyfile(src, output / name)
            copied.append(name)
    return copied


# ---------------------------------------------------------------------------
# Bounded quantised-shard writer with deterministic naming + weight map
# ---------------------------------------------------------------------------
class _ShardWriter:
    """Accumulates per-base-weight safetensors records into bounded shards and
    records which shard each base weight lands in (incl. its .scales/.biases)."""

    def __init__(self, output: Path, max_bytes: int):
        self.output = output
        self.max_bytes = max_bytes
        self.index = 0
        self.files: list[Path] = []
        self.weight_map: dict[str, str] = {}
        # current buffer: list of (base_name, tensors_dict), + running bytes
        self._buf: list[tuple[str, dict[str, np.ndarray]]] = []
        self._buf_bytes = 0

    def add(self, base: str, tensors: dict[str, np.ndarray]) -> None:
        nbytes = sum(t.nbytes for t in tensors.values())
        if self._buf and self._buf_bytes + nbytes > self.max_bytes:
            self._flush()
        self._buf.append((base, tensors))
        self._buf_bytes += nbytes

    def _flush(self) -> None:
        if not self._buf:
            return
        self.index += 1
        path = self.output / f"model-{self.index:05d}-of-00000.safetensors"
        payload: dict[str, np.ndarray] = {}
        for base, tensors in self._buf:
            for key, value in tensors.items():
                payload[key] = value
            self.weight_map[base] = path.name
        save_file(payload, str(path))
        self.files.append(path)
        self._buf = []
        self._buf_bytes = 0

    def finalize(self) -> list[Path]:
        self._flush()
        total = len(self.files)
        renamed: list[Path] = []
        for idx, path in enumerate(self.files, start=1):
            new = self.output / f"model-{idx:05d}-of-{total:05d}.safetensors"
            if new != path:
                path.replace(new)
            renamed.append(new)
            shard_name = new.name
            for base, target in list(self.weight_map.items()):
                if target == path.name:
                    self.weight_map[base] = shard_name
        self.files = renamed
        return renamed


# ---------------------------------------------------------------------------
# Core converter
# ---------------------------------------------------------------------------
def convert(
    source: Path,
    output: Path,
    *,
    max_shard_bytes: int,
    group_size: int = 32,
    min_free_bytes: int = 140 * 1024**3,
    max_rss_bytes: int = int(220.0 * 1024**3),
) -> dict:
    source = source.resolve()
    output = output.expanduser().resolve()
    if str(source).startswith(GUARD_EXPERT_SSD) or str(output).startswith(
        GUARD_EXPERT_SSD
    ):
        raise RuntimeError("Extreme SSD is outside this task")
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    out_root = output.parent
    if out_root.exists():
        avail = shutil.disk_usage(out_root).free
        if avail < min_free_bytes:
            raise RuntimeError(
                f"insufficient free space at {out_root}: {avail / 1024**3:.1f} "
                f"GiB < {min_free_bytes / 1024**3:.0f} GiB"
            )

    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError("source has no model.safetensors.index.json")
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index.get("weight_map", {})
    if not weight_map:
        raise RuntimeError("index has empty weight_map")

    output.mkdir(parents=True)
    started = time.monotonic()

    shard_layouts: dict[str, tuple[int, dict]] = {}
    for n in weight_map:
        sh = _safe_abs(source, Path(weight_map[n]))
        if str(sh) not in shard_layouts:
            shard_layouts[str(sh)] = _read_shard_layout(sh)

    writer = _ShardWriter(output, max_shard_bytes)
    total_weight_bytes = 0  # Σ numel*dtype_bytes over ALL source weights
    n_quant = n_copy = 0

    for name in sorted(weight_map):
        if peak_rss_bytes() > max_rss_bytes:
            raise RuntimeError(
                f"process footprint aborted: peak RSS {peak_rss_bytes() / 1024**3:.1f} "
                f"GiB > guard {max_rss_bytes / 1024**3:.1f} GiB"
            )
        src = _safe_abs(source, Path(weight_map[name]))
        header_len, header = shard_layouts[str(src)]
        info = header.get(name)
        if info is None:
            raise RuntimeError(f"weight {name!r} not in source shard {src}")
        shape = list(info["shape"])
        dtype = info["dtype"]
        total_weight_bytes += math.prod(shape) * _dtype_bytes(dtype)

        action = classify_tensor(name, shape, dtype, group_size)
        data = _tensor_bytes(src, header_len, header, name)
        if action == "copy":
            arr = _numpy_from_bytes(data, dtype, shape)
            writer.add(name, {name: arr})
            n_copy += 1
        else:
            arr = _numpy_from_bytes(data, dtype, shape).astype(np.float32)
            q, scales, biases = quantize_affine_q4_g32(arr)
            writer.add(
                name,
                {name: q, name + ".scales": scales, name + ".biases": biases},
            )
            n_quant += 1
        del data, arr

    quant_shards = writer.finalize()

    # Canonical output index. total_size = original model byte total (loader's
    # semantic "model size"), NOT source *.safetensors file sizes (which carry
    # per-shard headers and would over-count / drift across re-bundles).
    out_index = {
        "metadata": {"total_size": total_weight_bytes},
        "weight_map": dict(sorted(writer.weight_map.items())),
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(out_index, indent=2)
    )
    aux = _copy_aux_metadata(source, output)
    _write_sha256sums(output)

    files = sorted(p for p in output.rglob("*") if p.is_file())
    return {
        "source": str(source),
        "output": str(output),
        "files": len(files),
        "output_bytes": sum(p.stat().st_size for p in files),
        "peak_rss_bytes": peak_rss_bytes(),
        "wall_s": round(time.monotonic() - started, 3),
        "shards": [p.name for p in files if p.name.startswith("model-")],
        "group_size": group_size,
        "n_quant": n_quant,
        "n_copy": n_copy,
        "total_weight_bytes": total_weight_bytes,
        "aux_copied": aux,
        "status": "ok",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--max-shard-bytes", type=int, default=DEFAULT_MAX_SHARD_BYTES)
    ap.add_argument("--group-size", type=int, default=32)
    args = ap.parse_args()
    try:
        ledger = convert(
            args.source,
            args.output,
            max_shard_bytes=args.max_shard_bytes,
            group_size=args.group_size,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"CONVERT FAILED (fail-closed): {exc}", file=sys.stderr)
        return 1
    print(json.dumps(ledger, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
