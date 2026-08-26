# Qwen3.8-Flash-Next streaming q4-g32 converter (prototype v2)

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
2. **quantises** every quantisable weight (including PLE/embed tables) to affine
   **q4-g32** while streaming each tensor via `mmap` byte-slice — it never
   materialises the whole ~51B PLE table in RAM.

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

# 3. verify COPY preservation, q4-g32 quant round-trip, total_size, manifests
.venv/bin/python scripts/verify_synth_conversion.py /tmp/synth-src /tmp/synth-out

# 4. prove the guards abort closed
.venv/bin/python scripts/test_fail_closed_guards.py /tmp/synth-src
```

Expected: all green, with the converter ledger reporting `peak_rss_bytes`
(phys_footprint). On the synthetic set peak RSS is a few hundred MB.

## Converter CLI

```
--source           <snapshot dir>   must contain model.safetensors.index.json
--output           <dir>            must NOT exist (aborts if it does)
--max-shard-bytes  <int>            output shard cap (default 4 GiB)
--group-size       <int>            quant group size (default 32 = q4-g32)
```

Runbook-guard parameters (only via the Python API today): `min_free_bytes`
(default 140 GiB) and `max_rss_bytes` (default 220 GiB). On a real run the
operator confirms ≥140 GiB free at the output root before starting.

## Output contract

* **Quantised** weights (all 2-D, group-size-divisible tensors incl. the PLE /
  embed tables) → affine **q4-g32** (`mx.quantize(bits=4, group_size=32,
  mode='affine')`), each emitted with `.scales` / `.biases`, packed into
  bounded output shards. PLE is quantised, never preserved BF16, and never
  materialised as a whole table (per-tensor mmap streaming).
* **Copy** tensors (1-D norms/biases, `A_log`, aux/buffer dtypes, widths not
  divisible by group_size) → carried through value-for-value with their source
  shape/dtype, so no quantisable-unfriendly tensor is dropped or mangled.
* Non-weight metadata (config.json, generation_config.json, tokenizer files)
  copied verbatim (root level) so the output tree is self-contained loader
  input; these are never quantized.
* `model.safetensors.index.json` → `total_size` = Σ over every source weight of
  `numel × dtype_bytes` (the loader's semantic model size, not source
  *.safetensors file sizes), and a `weight_map` covering every original weight
  (2 extra keys per quantised tensor for scales/biases).
* Output shards use deterministic `model-{i:05d}-of-{N:05d}.safetensors` with
  the real total `N` (bounded by `--max-shard-bytes`).
* `SHA256SUMS.txt` → byte-sorted `sha256  <relative path>` per output file.
* Execution ledger on stdout: file count, output bytes, shard list, **peak RSS**,
  group_size, quant/copy counts, `total_weight_bytes`, `status`.

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

* bf16 **copy** tensors (norms etc.) are widened to fp32 in the output because
  numpy safetensors cannot carry bfloat16; a bf16-preserving copy needs mlx
  `framework="mlx"` in the transplant (Vector's loader). Quantised PATH handles
  bf16 correctly (widens via bit-manip → q4-g32).
* The manifest `classify_tensor` conjoins explicit embed/PLE names with the
  predicate; Vector may want to gate embed/PLE q4-g32 against the loader's
  `nn.Embedding.as_linear` / quantised-embedding contract (rows must be
  divisible by 32 — true for hidden=2560).
* `--group-size` defaults to 32; swap to 64 for a q4-g64 variant is a one-line
  change (shard sizes / packed dtype shift accordingly).
* No `--upload-repo` (forbidden on the real run by the runbook).

## Reference

* `mlx_lm convert.py` (`load(..., lazy=True)` → `quantize_model` → `save`) — the
  streaming MoE pattern this mirrors.
* `/private/tmp/rapid-qwen38-ops/qwen38_conversion_runbook.md` — operator
  runbook (source revision, output root, guards, command shape) this prototype
  implements.
