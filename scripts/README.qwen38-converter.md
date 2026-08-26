# Qwen3.8-Flash-Next streaming q4-g64 converter (prototype)

**Lane:** script-only helper for Vector's `qwen4_exp` port. Never edits model
math. This is a **prototype** to be reviewed and transplanted by Vector — it is
verified on a SMALL SYNTHETIC shard set, **not** on the real checkpoint.

## Why this shape

The real checkpoint (`Qwen/Qwen3.8-Flash-Next`, revision
`f5d08274bafd880402bd16f5e3e6c514136ec06c`) has ~113–131 safetensors shards and
a ~51B-param PLE embedding table spread across **128 PLE shards**. Loading the
PLE table into memory (or concatenating it) blows past any reasonable footprint
guard. So this converter:

1. processes the checkpoint **shard-by-shard** (one MoE expert matrix at a time,
   exactly as `mlx_lm convert.py` streams with `lazy=True`), and
2. preserves the PLE table **as-is** by streaming raw **byte-slices** out of each
   source shard via `mmap` — it never parses a PLE tensor into a dense array and
   never concatenates the table in RAM.

## Files

| file | purpose |
|---|---|
| `scripts/qwen38_streaming_convert.py` | the converter |
| `scripts/synthetic_qwen38_fixture.py` | generate a scaled-down synthetic shard set |
| `scripts/verify_synth_conversion.py` | end-to-end round-trip + SHA-256 verifier |
| `scripts/test_fail_closed_guards.py` | fail-closed guard tests |

## Requirements

Python 3.11+, `mlx`, `safetensors`, `numpy`. This worktree has a ready venv:

```sh
source .venv/bin/activate   # mlx 0.32.2, safetensors, numpy
```

## Run the synthetic verification

```sh
# 1. build a tiny stand-in checkpoint (same name/dtype/shape pattern, scaled down)
.venv/bin/python scripts/synthetic_qwen38_fixture.py /tmp/synth-src

# 2. convert it (fail-closed)
.venv/bin/python scripts/qwen38_streaming_convert.py \
    --source /tmp/synth-src \
    --output /tmp/synth-out \
    --max-shard-bytes 2000000

# 3. verify PLE byte-preservation, quant round-trip, SHA-256 manifest
.venv/bin/python scripts/verify_synth_conversion.py /tmp/synth-src /tmp/synth-out

# 4. prove the guards abort closed
.venv/bin/python scripts/test_fail_closed_guards.py /tmp/synth-src
```

Expected: all green, with the converter ledger reporting `peak_rss_bytes`
(phys_footprint). On the synthetic set peak RSS is a few hundred MB.

## Converter CLI

```
--source   <snapshot dir>           must contain model.safetensors.index.json
--output   <dir>                    must NOT exist (aborts if it does)
--max-shard-bytes <int>             quant output shard cap (default 4 GiB)
--ple-substr <str> [repeatable]     tensor-name substring = PLE (default: ple_embed,
                                    embed_tokens, mm.embedding)
```

Runbook-guard parameters (only via the Python API today): `min_free_bytes`
(default 140 GiB) and `max_rss_bytes` (default 220 GiB). On a real run the
operator confirms ≥140 GiB free at the output root before starting.

## Output contract

* Quantised weights → affine **q4-g64** (`mx.quantize(bits=4, group_size=64,
  mode='affine')`), each tensor emitted with `.scales` / `.biases`, packed into
  bounded output shards.
* PLE weights → copied **byte-for-byte** into `model-ple-00001.safetensors` via
  mmap byte-slices (never materialised), preserving source dtype/shape so the
  shard is a fully valid safetensors file.
* `model.safetensors.index.json` → covers every original weight exactly once
  (2 extra keys per quantised tensor for scales/biases).
* `SHA256SUMS.txt` → byte-sorted `sha256  <relative path>` per output file.
* Execution ledger on stdout: file count, output bytes, shard list, **peak RSS**,
  wall time, `status`.

## Fail-closed guarantees

| guard | behavior on violation |
|---|---|
| output dir exists | abort, no publish |
| missing / empty `model.safetensors.index.json` | abort |
| weight_map references a missing source shard | abort |
| source shard escapes source root (symlink / `..`) | abort |
| source or output under `/Volumes/Extreme SSD` | abort |
| free space at output root < 140 GiB | abort |
| process peak RSS > 220 GiB | abort mid-run |

## Known prototype limits (for Vector to finalise)

* `total_size` in the output index is the SOURCE byte total (computed before the
  weight_map is re-keyed). Vector should confirm the loader's expectation here.
* The PLE output shard keeps source dtype (often fp16); the Rapid loader's PLE
  fast-path must read by byte-span / dtype-shape, not assume fp32.
* `model.embed_tokens.weight` is classified PLE by substring and preserved
  (correct for this embedding-table-as-PLE model). If embed_tokens must instead
  be quantised in some variant, adjust `--ple-substr`.
* No `--upload-repo` (forbidden on the real run by the runbook).
* Shard count in the ledger is the raw quant shard set; a final re-bundling
  pass (packing small quant shards up to `--max-shard-bytes`) is not yet wired
  and is the obvious next step before a real production run.

## Reference

* `mlx_lm convert.py` (`load(..., lazy=True)` → `quantize_model` → `save`) — the
  streaming MoE pattern this mirrors.
* `/private/tmp/rapid-qwen38-ops/qwen38_conversion_runbook.md` — operator
  runbook (source revision, output root, guards, command shape) this prototype
  implements.
