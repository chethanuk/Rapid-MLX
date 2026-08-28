# Qwen3.8 Flash-Next MTP depth experiment (K=1/2/3)

## Outcome

Keep the production limit at K=1. Chaining the checkpoint's single MTP layer
to K=2 or K=3 improved 128-token-prompt decode by only 7--8%, regressed at
2K, and became substantially slower at 8K and 32K. The deeper variants do not
meet the break-even requirement for coding-agent and long-context workloads,
so the experimental capability change must not be promoted.

## Environment

| Component | Value |
| --- | --- |
| Machine | Mac Studio, Apple M3 Ultra |
| Unified memory | 256 GB |
| macOS | 26.5.2 (25F84) |
| Python | 3.12.13 |
| Experiment commit | `f2249272` (native-MTP production head plus the depth experiment) |
| MLX / MLX-LM | 0.32.1 / 0.31.3 |
| Artifact | `rapid-mlx/Qwen3.8-Flash-Next-4bit` at `dcf657e4acda2aae72da99cde65b6c491cd96998` |
| Quantization | PLE q4-g32; routing gates q8-g64; remainder q4-g64 |

No other large model was resident. Every depth used a fresh server process,
fixed greedy decoding, disabled thinking, disabled adaptive K, and the same
local snapshot. Prefix cache was cleared before every timed request.

## Exact method

For each value of `K` in 1, 2, and 3, start a fresh server:

```bash
export SNAPSHOT=/path/to/dcf657e4acda2aae72da99cde65b6c491cd96998
export K=1  # repeat with 2 and 3

HF_HUB_OFFLINE=1 PYTHONPATH="$PWD" python3.12 -m vllm_mlx.cli serve \
  "$SNAPSHOT" --host 127.0.0.1 --port 8465 --no-thinking \
  --speculative-config \
  "{\"method\":\"mtp\",\"num_speculative_tokens\":$K,\"disable_auto_k\":true}"
```

Then run the unchanged four-length harness. It requests 256 decode tokens at
128, 2,048, 8,192, and 32,768 target prompt tokens, with three cold-cache runs
per length and medians reported below:

```bash
python3.12 .orca/flash-next-eval/benchmark.py \
  --url http://127.0.0.1:8465/v1 \
  --model "$SNAPSHOT" --tokenizer-path "$SNAPSHOT" \
  --server-pid SERVER_PID --label "mtp-k${K}-f2249272" \
  --rapid-sha f2249272 \
  --artifact-revision dcf657e4acda2aae72da99cde65b6c491cd96998 \
  --output "/private/tmp/vector-flashnext-mtp-k${K}-benchmark.json"
```

## Results

| Prompt target | K=1 TTFT | K=2 TTFT | K=3 TTFT | K=1 decode | K=2 decode (delta) | K=3 decode (delta) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.383 s | 0.377 s | 0.372 s | 42.35 tok/s | 45.35 tok/s (+7.1%) | 45.62 tok/s (+7.7%) |
| 2,048 | 3.617 s | 3.542 s | 3.503 s | 35.63 tok/s | 34.41 tok/s (-3.4%) | 34.23 tok/s (-3.9%) |
| 8,192 | 14.623 s | 14.759 s | 14.663 s | 37.73 tok/s | 31.03 tok/s (-17.8%) | 27.14 tok/s (-28.1%) |
| 32,768 | 67.877 s | 71.217 s | 68.020 s | 33.14 tok/s | 17.93 tok/s (-45.9%) | 17.50 tok/s (-47.2%) |

| Depth | Accepted / proposed | Accept ratio | Final per-round target+draft cost | Largest observed MLX active memory | Peak RSS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| K=1 | 1,574 / 1,982 | 79.41% | 54.7 ms | 110.5 GB | 56.21 GiB |
| K=2 | 1,830 / 3,466 | 52.80% | 108.2 ms | 113.1 GB | 56.83 GiB |
| K=3 | 1,894 / 4,800 | 39.46% | 110.6 ms | 114.0 GB | 56.27 GiB |

The first 128-token timing at each depth included a new-shape compilation;
the median excludes that one-off effect. MLX active memory is the relevant
unified-memory sizing signal; RSS does not include all Metal allocations.

## Correctness

The generic chain-of-K all-accept, partial-accept, and coupled SSM/KV rollback
tests passed at K=3. Qwen4-specific GDN, PLE, QSA, KV, block-verification, and
rollback tests passed for K=1/2/3. Two deterministic real-checkpoint captures
(128 and 2K prompts, 256 generated tokens each) completed coherently at every
depth. K=1 and K=2 were token-identical for the 128 case; the remaining deeper
captures diverged only into coherent alternative explanations after target
verification. This is compatible with the documented near-tied-logit
accumulation boundary and showed no unverified draft leakage or cache
corruption.

Because K=2 and K=3 failed the performance gate, the 45-case release battery
was not rerun for them and no deeper-depth serving change is eligible for
production.

## Interpretation

The checkpoint owns one autoregressive MTP layer. K>1 repeatedly invokes that
same layer to build a chain, so later-position acceptance compounds downward
while draft work and target-verification width increase. At long context, the
extra cache-bearing draft step approximately doubled the measured round cost.
The modest short-prompt gain cannot repay the long-context regression expected
in software-engineering workloads.

The existing expected-value controller cannot make this an attractive default
by itself: its cost curve is model-level, while the experiment shows a strong
context-length crossover. Production should retain K=1. A future revisit
would first require a context-aware cost model or a genuinely parallel/tree
draft mechanism, followed by this same four-length and correctness gate.

## Additional observation

All three fresh server processes completed engine shutdown and cache cleanup,
then faulted during Python process teardown with worker threads still present.
This repeated 3/3 times and is independent of the K comparison. It should be
handled under the engine-lifecycle work rather than folded into a speculative-
depth change.
