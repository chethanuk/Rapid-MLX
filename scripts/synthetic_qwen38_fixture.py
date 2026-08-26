#!/usr/bin/env python3
"""Generate a SMALL SYNTHETIC Qwen3.8-Flash-Next-style shard set.

Same tensor-name/dtype/shape *pattern* as the real checkpoint, scaled down so
the converter can be verified end-to-end in seconds without any real weights:

  * a few quantisable MoE blocks  (model.layers.N.mlp.experts.M.down_proj etc.)
    spread across input shards, one expert-sized matrix at a time
  * a whole input shard that is PLE-pure (only ``ple_embed.*`` tensors) to
    exercise the byte-copy-as-is path that preserves the ~51B table upstream
  * a mixed input shard (some MoE + one PLE tensor) to prove the PLE byte
    slicer handles non-PLE-pure shards too

Produces ``<out>/{model-0000X-of-00007.safetensors, model.safetensors.index.json}``.
The real snapshot instead has 113-131 shards and a real index; this fixture only
mimics the shapes/dtypes and sharding mechanic. Never run against real weights.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


def _expert_tensors(prefix: str, seed: int) -> dict[str, np.ndarray]:
    """Small MoE expert block: down_proj (int-x hidden), gate/up (hidden x int)."""
    rng = np.random.default_rng(seed)
    hidden, inter = 2560, 640  # same hidden/inter shapes as the real model, tiny
    return {
        f"{prefix}.down_proj.weight": rng.standard_normal(
            (inter, hidden), dtype=np.float32
        ),
        f"{prefix}.gate_proj.weight": rng.standard_normal(
            (hidden, inter), dtype=np.float32
        ),
        f"{prefix}.up_proj.weight": rng.standard_normal(
            (hidden, inter), dtype=np.float32
        ),
    }


def build(out: Path, num_experts: int = 16, ple_tensors_per_shard: int = 2) -> None:
    out = out.resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing non-empty fixture dir: {out}")
    out.mkdir(parents=True, exist_ok=True)

    hidden = 2560  # same hidden dim so quant shapes are realistic
    ple_shape = (64, hidden)  # tiny stand-in for a 248320-row PLE row chunk
    total_shards = 7
    shards: list[dict[str, np.ndarray]] = []
    weight_map: dict[str, str] = {}

    # Four MoE shards, each a handful of experts.
    for s in range(4):
        tensors: dict[str, np.ndarray] = {}
        for e in range(num_experts // 2):
            blk = f"layers.0.mlp.experts.{s * (num_experts // 2) + e}"
            tensors.update(_expert_tensors(blk, seed=s * 100 + e))
        shard_name = f"model-0000{s+1}-of-0000{total_shards}.safetensors"
        for k in tensors:
            weight_map[k] = shard_name
        shards.append(tensors)

    # Shard 5: a few dense weights (non-PLE, quantisable).
    dense = {
        "model.embed_tokens.weight": np.zeros(ple_shape, dtype=np.float32),
        "layers.0.self_attn.q_proj.weight": np.random.default_rng(7).standard_normal(
            (hidden, hidden), dtype=np.float32
        ),
    }
    shard_name5 = f"model-00005-of-0000{total_shards}.safetensors"
    for k in dense:
        weight_map[k] = shard_name5
    shards.append(dense)

    # Shard 6: a PLE-pure shard (the byte-copy-path target, ~51B upstream).
    pure_ple: dict[str, np.ndarray] = {}
    for i in range(ple_tensors_per_shard):
        name = f"ple_embed.rows.{i}.weight"
        pure_ple[name] = np.random.default_rng(100 + i).standard_normal(
            ple_shape, dtype=np.float32
        ).astype(np.float16)  # PLE rows often fp16
        weight_map[name] = f"model-00006-of-0000{total_shards}.safetensors"
    shards.append(pure_ple)

    # Shard 7: a mixed shard (one dense + one PLE tensor) => PLE byte-slicer
    # must isolate the PLE tensor and leave the dense one to quantise.
    mixed = {
        "layers.1.self_attn.o_proj.weight": np.random.default_rng(9).standard_normal(
            (hidden, hidden), dtype=np.float32
        ),
    }
    mm_name = "mm.embedding.weight"
    mixed[mm_name] = np.random.default_rng(10).standard_normal(
        (32, hidden), dtype=np.float32
    ).astype(np.float16)
    shard_name7 = f"model-00007-of-0000{total_shards}.safetensors"
    for k in mixed:
        weight_map[k] = shard_name7
    shards.append(mixed)

    for tensors, slot in zip(shards, range(1, total_shards + 1)):
        save_file(tensors, str(out / f"model-0000{slot}-of-0000{total_shards}.safetensors"))

    index = {
        "metadata": {"total_size": sum(t.nbytes for sh in shards for t in sh.values())},
        "weight_map": weight_map,
    }
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    print(f"fixture written to {out}: {total_shards} shards, "
          f"{len(weight_map)} weights")


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic-kit")
    build(out)
