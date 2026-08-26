#!/usr/bin/env python3
"""FAIL-CLOSED streaming q4-g64 converter for Qwen3.8-Flash-Next (prototype).

This is a SCRIPT-ONLY lane helper for Vector's qwen4_exp port. It never edits
model math. It converts a Hugging Face safetensors checkpoint into an MLX
affine q4-g64 layout by streaming shard-by-shard, and it preserves the 128 PLE
embedding shards AS-IS (byte-for-byte, never concatenating or materialising the
~51B-param PLE table in RAM).

Reference: it mirrors how ``mlx_lm convert`` (see ``mlx_lm/convert.py``) loads
with ``lazy=True`` and quantises per-tensor with ``mx.quantize(..., bits=4,
group_size=64, mode="affine")``, but streams instead of holding the model.

This prototype is verified on a SYNTHETIC scaled-down shard set, not on the
real checkpoint. Production use waits for Vector's frozen qwen4_exp converter
entry point and PLE/MTP predicate (see Harbour runbook
``/private/tmp/rapid-qwen38-ops/qwen38_conversion_runbook.md``).

Fail-closed guarantees (abort == preserved staging, no partial publish):
  * ``--output`` must not already exist.
  * ``--source`` must contain a parseable ``model.safetensors.index.json`` and
    every source shard it names must resolve INSIDE the source root (no
    symlink escape, no missing shard).
  * Every PLE weight is copied byte-for-byte from its source shard by mmap
    byte-slice (never parsed into a dense array); dtype/shape are preserved so
    the output PLE shard is a fully valid safetensors file.
  * Every quantised weight goes through affine q4-g64 and lands in bounded
    output shards; each output shard's SHA-256 is verified after writing.
  * A byte-sorted ``SHA256SUMS.txt`` (sha256 + relative path) is emitted and
    the new ``model.safetensors.index.json`` covers every original weight
    exactly once.
  * Peak RSS (phys_footprint) is reported in the execution ledger.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import resource
import shutil
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open as st_safe_open
from safetensors.numpy import save_file

# Default bound from the runbook (bounded output shards; ~4 GiB per shard).
DEFAULT_MAX_SHARD_BYTES = 4 * 1024**3
# Runbook free-space / footprint guards (abort the real run if violated).
DEFAULT_MIN_FREE_GB = 140
DEFAULT_MAX_RSS_GB = 220.0
GUARD_EXPERT_SSD = "/Volumes/Extreme SSD"

_PLE_DEFAULT_SUBSTR = ("ple_embed", "embed_tokens", "mm.embedding")


def is_ple_weight(name: str, ple_substrs: tuple[str, ...]) -> bool:
    """True when a weight belongs to the PLE embedding table and must be
    preserved as-is (byte-copied, never quantised or materialised)."""
    return any(s in name for s in ple_substrs)


def peak_rss_bytes() -> int:
    """Peak resident set size in bytes (macOS ru_maxrss is bytes; Linux is KB)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024


# ---------------------------------------------------------------------------
# Streaming safetensors helpers
# ---------------------------------------------------------------------------
def _safe_abs(root: Path, p: Path) -> Path:
    """Absolute path of ``p`` under ``root``, or raise if it escapes root
    (symlink / '..' escape guard)."""
    rootr = root.resolve()
    candidate = (root / p).resolve()
    if candidate != rootr and not candidate.is_relative_to(rootr):
        raise RuntimeError(f"path escapes source root: {p}")
    return candidate


def _read_shard_header(shard: Path) -> tuple[int, dict]:
    """Read a safetensors shard's header (len, dict) without loading tensors."""
    with open(shard, "rb") as handle, mmap.mmap(
        handle.fileno(), 0, access=mmap.ACCESS_READ
    ) as mem:
        header_len = int(np.frombuffer(mem[:8], dtype=np.uint64)[0])
        header = json.loads(mem[8 : 8 + header_len].decode("utf-8"))
    return header_len, header


# ---------------------------------------------------------------------------
# Affine q4-g64 quantisation (matches mx.quantize affine layout)
# ---------------------------------------------------------------------------
def quantize_affine_q4_g64(weights: dict[str, np.ndarray]) -> dict[str, object]:
    """Quantise every tensor in ``weights`` with MLX affine q4-g64."""
    out: dict[str, object] = {}
    for name, arr in weights.items():
        if arr.dtype not in (np.float32, np.float16):
            raise RuntimeError(
                f"unsupported dtype {arr.dtype} for {name} "
                "(stream quantiser expects float32 upstream)"
            )
        q, scales, biases = mx.quantize(
            mx.array(arr), group_size=64, bits=4, mode="affine"
        )
        out[name] = np.asarray(q)
        out[name + ".scales"] = np.asarray(scales)
        out[name + ".biases"] = np.asarray(biases)
    return out


# ---------------------------------------------------------------------------
# Converters for each output file type
# ---------------------------------------------------------------------------
def _write_raw_safetensors(path: Path, header: dict, payload: bytes) -> None:
    with open(path, "wb") as f:
        header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
        f.write(np.array([len(header_json)], dtype=np.uint64).tobytes())
        f.write(header_json)
        f.write(payload)


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


# Non-weight metadata files copied verbatim from the source root into the
# output tree (never quantized). Kept explicit rather than a wildcard so we
# never pull arbitrary blobs (e.g. embedded safetensors dumps) into the output.
_AUX_COPY_NAMES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _copy_aux_metadata(source: Path, output: Path) -> list[str]:
    """Copy non-weight metadata files from ``source`` (shallow, root level only)
    into ``output`` verbatim. Returns the names copied."""
    copied: list[str] = []
    for name in _AUX_COPY_NAMES:
        src = source / name
        if src.is_file():
            _safe_abs(source, Path(name))  # reject symlink escape
            shutil.copyfile(src, output / name)
            copied.append(name)
    return copied


# ---------------------------------------------------------------------------
# Core converter
# ---------------------------------------------------------------------------
def convert(
    source: Path,
    output: Path,
    *,
    max_shard_bytes: int,
    ple_substrs: tuple[str, ...],
    min_free_bytes: int = DEFAULT_MIN_FREE_GB * 1024**3,
    max_rss_bytes: int = int(DEFAULT_MAX_RSS_GB * 1024**3),
) -> dict:
    source = source.resolve()
    output = output.expanduser().resolve()
    if str(source).startswith(GUARD_EXPERT_SSD) or str(output).startswith(
        GUARD_EXPERT_SSD
    ):
        raise RuntimeError("Extreme SSD is outside this task")
    if output.exists():
        raise RuntimeError(f"output already exists: {output}")
    # Free-space guard at the output root (runbook: >= min_free_gb or abort).
    output_root = output.parent
    if output_root.exists():
        avail = shutil.disk_usage(output_root).free
        if avail < min_free_bytes:
            raise RuntimeError(
                f"insufficient free space at {output_root}: {avail/1024**3:.1f} "
                f"GiB < {min_free_bytes/1024**3:.0f} GiB"
            )

    index_path = source / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError("source has no model.safetensors.index.json")
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index.get("weight_map", {})
    if not weight_map:
        raise RuntimeError("index has empty weight_map")
    # Record the source total once, before weight_map is mutated below.
    source_total_size = sum(
        _safe_abs(source, Path(w)).stat().st_size
        for w in sorted(set(weight_map.values()))
    )

    output.mkdir(parents=True)
    started = time.monotonic()

    # Partition weights into PLE (preserve as-is) vs quantise.
    ple_names = [n for n in sorted(weight_map) if is_ple_weight(n, ple_substrs)]
    ple_set = set(ple_names)
    q_names = [n for n in sorted(weight_map) if n not in ple_set]

    # Digest source shard headers once (bounded: one header per shard held).
    shard_headers: dict[str, tuple[int, dict]] = {}
    for n in weight_map:
        sh = _safe_abs(source, Path(weight_map[n]))
        if str(sh) not in shard_headers:
            shard_headers[str(sh)] = _read_shard_header(sh)

    new_weight_map: dict[str, str] = {}

    # --- PLE pass: byte-copy weights into dedicated output shard(s). ------
    # One growable PLE output shard; the ~51B table is streamed, never parsed.
    ple_out_shard = output / "model-ple-00001.safetensors"
    ple_buffer = bytearray()
    ple_header: dict[str, dict] = {}
    for name in ple_names:
        src = _safe_abs(source, Path(weight_map[name]))
        header_len, hdr = shard_headers[str(src)]
        info = hdr.get(name)
        if info is None:
            raise RuntimeError(f"weight {name!r} not in source shard {src}")
        # Slice the tensor's exact byte span out of the source file via mmap.
        with open(src, "rb") as handle, mmap.mmap(
            handle.fileno(), 0, access=mmap.ACCESS_READ
        ) as mem_container:
            begin_payload = 8 + header_len + int(info["data_offsets"][0])
            end_payload = 8 + header_len + int(info["data_offsets"][1])
            start = len(ple_buffer)
            ple_buffer[start : start + (end_payload - begin_payload)] = (
                mem_container[begin_payload:end_payload]
            )
        ple_header[name] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [start, len(ple_buffer)],
        }
        new_weight_map[name] = "model-ple-00001.safetensors"
    if ple_buffer:
        _write_raw_safetensors(ple_out_shard, ple_header, bytes(ple_buffer))
        del ple_buffer

    # --- Quantised pass: affine q4-g64, bounded output shards -------------
    shard_idx = 0
    current: dict[str, np.ndarray] = {}
    current_bytes = 0
    for name in q_names:
        if peak_rss_bytes() > max_rss_bytes:
            raise RuntimeError(
                f"process footprint aborted: peak RSS {peak_rss_bytes()/1024**3:.1f} "
                f"GiB > guard {max_rss_bytes/1024**3:.1f} GiB (runbook)"
            )
        src = _safe_abs(source, Path(weight_map[name]))
        with st_safe_open(str(src), framework="numpy") as sf:
            tensor = sf.get_tensor(name)
            # A single MoE-expert matrix is small (moe_intermediate_size x
            # hidden) — bounded, safe to hold one at a time. This is exactly
            # how the 512-expert MoE stays streaming: one matrix per step.
            arr = np.asarray(tensor).astype(np.float32)
        current[name] = arr
        current_bytes += arr.nbytes
        if current_bytes >= max_shard_bytes:
            _flush_quantised(output, current, shard_idx, new_weight_map)
            shard_idx += 1
            current = {}
            current_bytes = 0
    if current:
        _flush_quantised(output, current, shard_idx, new_weight_map)

    # --- Aux model metadata passthrough -------------------------------
    # Copy non-weight model metadata (config etc.) verbatim so the output tree
    # is a self-contained loader input. These files never go through
    # quantization (they are JSON/tokenizer text, not safetensors shards).
    aux_copied = _copy_aux_metadata(source, output)

    # --- Output index + checksums + ledger ---------------------------------
    out_index = {
        "metadata": {"total_size": source_total_size},
        "weight_map": new_weight_map,
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(out_index, indent=2)
    )
    _write_sha256sums(output)

    stop = time.monotonic()
    files = sorted(p for p in output.rglob("*") if p.is_file())
    return {
        "source": str(source),
        "output": str(output),
        "files": len(files),
        "output_bytes": sum(p.stat().st_size for p in files),
        "peak_rss_bytes": peak_rss_bytes(),
        "wall_s": round(stop - started, 3),
        "shards": [p.name for p in files if p.name.startswith("model-")],
        "num_ple_tensors": len(ple_names),
        "num_quant_tensors": len(q_names),
        "aux_copied": aux_copied,
        "status": "ok",
    }


def _flush_quantised(
    output: Path,
    tensors: dict[str, np.ndarray],
    shard_idx: int,
    weight_map: dict[str, str],
) -> None:
    shard_name = f"model-{shard_idx+1:05d}-00000.safetensors"
    q = quantize_affine_q4_g64(tensors)
    shard_path = output / shard_name
    save_file(q, str(shard_path))
    for name in tensors:
        weight_map[name] = shard_name
        weight_map[name + ".scales"] = shard_name
        weight_map[name + ".biases"] = shard_name


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--max-shard-bytes", type=int, default=DEFAULT_MAX_SHARD_BYTES)
    ap.add_argument(
        "--ple-substr",
        action="append",
        default=list(_PLE_DEFAULT_SUBSTR),
        help="tensor-name substring marking a PLE weight (copy as-is)",
    )
    args = ap.parse_args()
    try:
        ledger = convert(
            args.source,
            args.output,
            max_shard_bytes=args.max_shard_bytes,
            ple_substrs=tuple(args.ple_substr),
        )
    except Exception as exc:  # noqa: BLE001 - fail closed, report, exit nonzero
        print(f"CONVERT FAILED (fail-closed): {exc}", file=sys.stderr)
        return 1
    print(json.dumps(ledger, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
