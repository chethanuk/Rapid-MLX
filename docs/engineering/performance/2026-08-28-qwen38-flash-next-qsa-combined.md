# Qwen3.8 Flash-Next direct sparse QSA follow-up

This experiment retests the default-off direct block-sparse QSA prefill kernel
after the native MTP K=1 and batched compressed-key prefill optimizations landed
on `main`. It answers whether the earlier sparse-attention prototype provides a
large enough incremental win to justify a production PR.

## Environment

| Component | Value |
| --- | --- |
| Machine | Mac Studio (`Mac15,14`) |
| Chip | Apple M3 Ultra, 28 CPU cores |
| Unified memory | 256 GB |
| macOS | 26.5.2 (25F84) |
| Architecture | arm64 |
| Python | 3.12.14 |
| Baseline | `main` at `4c94abfef79002fcbad10c5a283b163f48d1ba1d` |
| Candidate | `f8e3c81d` on `experiment/qwen4-qsa-sparse-combined` |
| MLX | 0.32.2 |
| MLX-LM | 0.31.3 |
| Transformers | 5.12.1 |
| Artifact | `rapid-mlx/Qwen3.8-Flash-Next-4bit` at `dcf657e4acda2aae72da99cde65b6c491cd96998` |
| MTP | Native K=1, fixed depth, enabled on both variants |

No other model server was resident. Each variant ran in a fresh process with
the prefix cache disabled and cleared before every request. The candidate
changed only `RAPID_MLX_QSA_BLOCK_SPARSE=1`.

## Exact commands

```bash
export SNAPSHOT=/path/to/dcf657e4acda2aae72da99cde65b6c491cd96998
export WORKTREE=/path/to/qwen4-qsa-sparse-combined

# Omit RAPID_MLX_QSA_BLOCK_SPARSE for the baseline process.
RAPID_MLX_QSA_BLOCK_SPARSE=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$WORKTREE" \
python3.12 -m vllm_mlx.cli serve "$SNAPSHOT" \
  --served-model-name rapid-mlx/Qwen3.8-Flash-Next-4bit \
  --host 127.0.0.1 --port 8464 --no-thinking \
  --disable-prefix-cache --max-num-seqs 1 \
  --prefill-batch-size 1 --completion-batch-size 1 \
  --speculative-config '{"method":"mtp","disable_auto_k":true}'

python3.12 .orca/flash-next-eval/benchmark.py \
  --url http://127.0.0.1:8464/v1 \
  --model rapid-mlx/Qwen3.8-Flash-Next-4bit \
  --tokenizer-path "$SNAPSHOT" --server-pid SERVER_PID \
  --label VARIANT --runs 3 --decode-tokens 256 \
  --prompt-tokens 128,2048,8192,32768 \
  --artifact-revision dcf657e4acda2aae72da99cde65b6c491cd96998 \
  --rapid-sha RAPID_SHA --output OUTPUT.json --timeout 3600
```

## Results

| Target (reported) | Baseline TTFT | Sparse TTFT | TTFT change | Baseline prefill | Sparse prefill | Baseline decode | Sparse decode |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 (92) | 0.317 s | 0.319 s | +0.4% | 289.9 tok/s | 288.8 tok/s | 43.72 tok/s | 43.91 tok/s |
| 2,048 (2,012) | 2.237 s | 2.246 s | +0.4% | 899.5 tok/s | 896.0 tok/s | 40.46 tok/s | 39.60 tok/s |
| 8,192 (8,156) | 9.264 s | 9.286 s | +0.2% | 880.4 tok/s | 878.3 tok/s | 38.72 tok/s | 37.99 tok/s |
| 32,768 (32,732) | **45.105 s** | **38.724 s** | **-14.15%** | 725.7 tok/s | **845.3 tok/s** | 33.52 tok/s | 34.33 tok/s |

The sparse kernel activates only at 16,384 physical KV tokens. The differences
below that boundary are process-to-process noise. Peak process RSS was 55.87
GiB on both variants. The engine reported 109.4 GB baseline versus 109.5 GB
candidate MLX active memory after the third 32K run, so the candidate did not
materially change the practical memory requirement. A 128 GB machine was not
physically tested; 192 GB remains the practical recommended tier.

## Rejected follow-ups

- Lowering the crossover to 4,096 physical KV tokens made the 8K median TTFT
  0.8% slower (9.264 to 9.336 seconds), so the 16K threshold is retained.
- Loading eight selected four-token blocks per threadgroup tile reduced barrier
  count but lowered occupancy. The 32K median regressed from 38.724 to 39.142
  seconds.
- Splitting indexer scoring into causal 8K query chunks made an isolated score
  and top-k microbenchmark 48% faster, but only 96.9% of selected indices
  matched the single-matmul path and end-to-end 32K TTFT regressed to 39.100
  seconds. The experiment was reverted.

## Verdict

**No production PR.** The stable candidate saves 6.38 seconds at 32K and is
mathematically exact for the selected block set, but its 14.15% incremental
TTFT improvement misses the 15% gate and has no benefit at 8K. That is not
enough to justify adding and maintaining a model-specific Metal attention
backend of roughly 670 changed lines. The default engine remains the landed
batched compressed-key implementation; this branch is retained as reproducible
evidence for a future fused score/top-k/attention backend.
