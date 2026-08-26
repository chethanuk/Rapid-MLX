# SPDX-License-Identifier: Apache-2.0
"""Vendored MLX text model for the Qwen4-Exp architecture.

The implementation is intentionally architecture-driven: every optional
component is admitted from the checkpoint's typed ``text_config`` fields.
There are no repository-name aliases or compatibility approximations here.

M1 implements the target text decoder.  The checkpoint's MTP and vision
modules are deliberately ignored by :meth:`Model.sanitize` until their own
milestones have independent numerical and lifecycle coverage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .. import _mlx_compat as _mlx_compat

_mlx_compat.install()

from mlx_lm.models.base import (  # noqa: E402
    BaseModelArgs,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache  # noqa: E402
from mlx_lm.models.gated_delta import gated_delta_update  # noqa: E402
from mlx_lm.models.rope_utils import initialize_rope  # noqa: E402
from mlx_lm.models.switch_layers import SwitchGLU  # noqa: E402

from .qwen4_exp_cache import QSAIndexCache  # noqa: E402


@dataclass
class TextModelArgs(BaseModelArgs):
    model_type: str = "qwen4_exp_text"
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    attention_dropout: float = 0.0

    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    rope_parameters: dict[str, Any] | None = field(
        default_factory=lambda: {
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
        }
    )
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10_000_000.0

    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    output_gate_type: str = "sigmoid"

    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True

    hc_count: int = 4
    hc_lowrank: int = 320

    layer_types: list[str] | None = None
    full_attention_interval: int = 4
    indexer_n_heads: int | None = None
    indexer_kv_heads: int | None = None
    indexer_head_dim: int | None = None
    indexer_budget: int | None = None
    indexer_compress_ratio: int | None = None

    ple_layer_ids: list[int] = field(default_factory=list)
    ple_embed_dim: int | None = None
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    split_ngram_parts: int = 128
    seed: int = 1234
    eos_token_id: int | list[int] | None = None

    def __post_init__(self):
        rope = dict(self.rope_parameters or {})
        self.partial_rotary_factor = float(
            rope.get("partial_rotary_factor", self.partial_rotary_factor)
        )
        self.rope_theta = float(rope.get("rope_theta", self.rope_theta))
        self.ple_embed_dim = self.hidden_size if self.ple_embed_dim is None else self.ple_embed_dim
        self.ple_layer_ids = sorted(set(self.ple_layer_ids))
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention"
                if (index + 1) % self.full_attention_interval
                else "qwen_sparse_attention"
                for index in range(self.num_hidden_layers)
            ]
        else:
            self.layer_types = [
                "qwen_sparse_attention" if kind == "full_attention" else kind
                for kind in self.layer_types
            ]
        self._validate()

    def _validate(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                "Qwen4-Exp layer_types must have one entry per decoder layer"
            )
        unsupported = set(self.layer_types) - {
            "linear_attention",
            "qwen_sparse_attention",
        }
        if unsupported:
            raise ValueError(f"Unsupported Qwen4-Exp layer types: {sorted(unsupported)}")
        if self.hc_count <= 1 or self.hidden_size <= 0 or self.hc_lowrank <= 0:
            raise ValueError("Qwen4-Exp requires positive four-stream HC dimensions")
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError("linear value heads must be divisible by key heads")
        if self.output_gate_type != "sigmoid":
            raise ValueError(
                "Qwen4-Exp M1 supports the checkpoint-declared sigmoid GDN gate only"
            )
        qsa = (
            self.indexer_n_heads,
            self.indexer_kv_heads,
            self.indexer_head_dim,
            self.indexer_budget,
            self.indexer_compress_ratio,
        )
        if any(value is None for value in qsa):
            raise ValueError("Qwen4-Exp QSA requires the complete indexer contract")
        if any(int(value) <= 0 for value in qsa if value is not None):
            raise ValueError("Qwen4-Exp indexer values must be positive")
        if self.indexer_kv_heads != 1:
            raise ValueError("Qwen4-Exp QSA requires one indexer KV head")
        if self.indexer_budget % self.indexer_compress_ratio:
            raise ValueError("indexer_budget must be divisible by indexer_compress_ratio")
        rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if rotary_dim > self.indexer_head_dim:
            raise ValueError("attention rotary dimensions must fit the QSA index head")
        if not 0 < self.num_experts_per_tok <= self.num_experts:
            raise ValueError("num_experts_per_tok must be within num_experts")
        ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ple_layer_ids and self.ple_embed_dim % ngram_heads:
            raise ValueError("PLE embedding width must divide evenly across n-gram heads")
        if any(layer < 1 or layer > self.num_hidden_layers for layer in self.ple_layer_ids):
            raise ValueError("ple_layer_ids are one-indexed decoder layer ids")
        if any(self.layer_types[layer - 1] != "linear_attention" for layer in self.ple_layer_ids):
            raise ValueError("PLE is only valid on linear-attention layers")
        if self.ple_layer_ids and self.eos_token_id is None:
            raise ValueError("PLE requires eos_token_id for segment-local n-grams")


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    text_config: dict[str, Any]

    @classmethod
    def from_dict(cls, params):
        if "text_config" not in params:
            return cls(model_type=params["model_type"], text_config=params)
        return super().from_dict(params)


class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm whose checkpoint weight represents an additive delta from one."""

    def __init__(self, dim: int, *, group_size: int | None = None, eps: float = 1e-6):
        super().__init__()
        if group_size is not None and dim % group_size:
            raise ValueError("grouped RMSNorm width must divide the feature width")
        self.weight = mx.zeros((dim,))
        self.group_size = group_size
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        original_shape = x.shape
        if self.group_size is not None:
            x = x.reshape(*x.shape[:-1], -1, self.group_size)
            weight = self.weight.reshape(-1, self.group_size)
        else:
            weight = self.weight
        dtype = x.dtype
        normalized = x.astype(mx.float32)
        normalized = normalized * mx.rsqrt(
            mx.mean(mx.square(normalized), axis=-1, keepdims=True) + self.eps
        )
        normalized = normalized * (1.0 + weight.astype(mx.float32))
        return normalized.astype(dtype).reshape(original_shape)


class SigmoidRMSNormGated(nn.Module):
    """Qwen4-Exp's norm-before-gate GDN output transform."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, hidden: mx.array, gate: mx.array) -> mx.array:
        dtype = hidden.dtype
        normalized = hidden.astype(mx.float32)
        normalized = normalized * mx.rsqrt(
            mx.mean(mx.square(normalized), axis=-1, keepdims=True) + self.eps
        )
        normalized = normalized * self.weight.astype(mx.float32)
        return (normalized * mx.sigmoid(gate.astype(mx.float32))).astype(dtype)


class GatedResidual(nn.Module):
    """Exact four-stream read mixer and per-branch write gate."""

    def __init__(self, args: TextModelArgs, *, use_combine: bool = True):
        super().__init__()
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        hc_width = self.hc_count * self.hidden_size
        self.hc_norm = ZeroCenteredRMSNorm(
            hc_width, group_size=self.hidden_size, eps=args.rms_norm_eps
        )
        self.input_mix_weight_down = nn.Linear(hc_width, args.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(args.hc_lowrank, hc_width, bias=False)
        self.block_inject_weight = (
            nn.Linear(hc_width, self.hc_count, bias=False) if use_combine else None
        )

    def __call__(self, hyper_input: mx.array):
        expected = self.hc_count * self.hidden_size
        if hyper_input.shape[-1] != expected:
            raise ValueError(
                f"Qwen4-Exp HC expected {expected} features, got {hyper_input.shape[-1]}"
            )
        normalized = self.hc_norm(hyper_input)
        mix = nn.silu(self.input_mix_weight_down(normalized) / self.hc_count)
        mix = mx.sigmoid(self.input_mix_weight_up(mix))
        mix = mix.reshape(*mix.shape[:-1], self.hc_count, self.hidden_size)
        streams = normalized.reshape(
            *normalized.shape[:-1], self.hc_count, self.hidden_size
        )
        mixed = mx.mean(mix * streams, axis=-2)
        if self.block_inject_weight is None:
            return mixed
        injection = 2 * mx.sigmoid(
            self.block_inject_weight(normalized) / self.hc_count
        )
        return mixed, hyper_input, injection


class MLP(nn.Module):
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoeBlock(nn.Module):
    """Softmax top-k routed experts plus the separately gated shared expert."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts
        )
        self.shared_expert = MLP(
            args.hidden_size, args.shared_expert_intermediate_size
        )
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)
        self.sharding_group = None

    def __call__(self, x: mx.array) -> mx.array:
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        indices = mx.argpartition(gates, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        scores = mx.take_along_axis(gates, indices, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        routed = self.switch_mlp(x, indices)
        routed = (routed * scores[..., None]).sum(axis=-2)
        shared = self.shared_expert(x)
        shared = mx.sigmoid(self.shared_expert_gate(x)) * shared
        return routed + shared


class GatedDeltaNet(nn.Module):
    """Qwen4-Exp GDN with 48 value heads and a sigmoid output gate."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
        )
        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.conv_dim, bias=False
        )
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.dt_bias = mx.ones((self.num_v_heads,))
        self.A_log = mx.log(
            mx.random.uniform(low=0.01, high=16.0, shape=(self.num_v_heads,))
        )
        self.norm = SigmoidRMSNormGated(self.head_v_dim, eps=args.rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array | None = None,
        cache: Any | None = None,
    ) -> mx.array:
        batch, length, _ = inputs.shape
        mixed = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(
            batch, length, self.num_v_heads, self.head_v_dim
        )
        beta = self.in_proj_b(inputs)
        alpha = self.in_proj_a(inputs)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (batch, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )
        if mask is not None:
            mixed = mx.where(mask[..., None], mixed, 0)
        conv_input = mx.concatenate([conv_state, mixed], axis=1)
        if cache is not None:
            keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, length)
                positions = (ends[:, None] + mx.arange(keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -keep:, :])
        convolved = nn.silu(self.conv1d(conv_input))
        query, key, value = [
            item.reshape(batch, length, heads, dim)
            for item, heads, dim in zip(
                mx.split(convolved, [self.key_dim, 2 * self.key_dim], axis=-1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        state = cache[1] if cache is not None else None
        inv_scale = key.shape[-1] ** -0.5
        query = (inv_scale**2) * mx.fast.rms_norm(query, None, 1e-6)
        key = inv_scale * mx.fast.rms_norm(key, None, 1e-6)
        output, state = gated_delta_update(
            query,
            key,
            value,
            alpha,
            beta,
            self.A_log,
            self.dt_bias,
            state,
            mask,
            use_kernel=not self.training,
        )
        if cache is not None:
            cache[1] = state
            cache.advance(length)
        output = self.norm(output, z)
        return self.out_proj(output.reshape(batch, length, -1))


def apply_rotary_positions(
    x: mx.array,
    positions: mx.array,
    *,
    rotary_dim: int,
    base: float,
) -> mx.array:
    """Apply non-interleaved partial RoPE at explicit logical positions.

    ``x`` has shape ``[batch, tokens, heads, dim]`` and ``positions`` is
    ``[tokens]`` or ``[batch, tokens]``. Explicit positions are required by
    QSA because compressed keys rotate at each group's first token.
    """

    if rotary_dim == 0:
        return x
    if rotary_dim % 2:
        raise ValueError("Qwen4-Exp rotary dimensions must be even")
    if positions.ndim == 1:
        positions = positions[None, :]
    inverse_frequency = 1.0 / (
        base
        ** (
            mx.arange(0, rotary_dim, 2, dtype=mx.float32)
            / float(rotary_dim)
        )
    )
    angles = positions.astype(mx.float32)[..., None] * inverse_frequency
    angles = mx.concatenate([angles, angles], axis=-1)[:, :, None, :]
    cosine = mx.cos(angles)
    sine = mx.sin(angles)
    rotary = x[..., :rotary_dim]
    half = rotary_dim // 2
    rotated_half = mx.concatenate(
        [-rotary[..., half:], rotary[..., :half]], axis=-1
    )
    rotated = rotary * cosine + rotated_half * sine
    return mx.concatenate([rotated.astype(x.dtype), x[..., rotary_dim:]], axis=-1)


class QSAIndexer(nn.Module):
    """Weight-bearing QSA selector backed by raw-ring/compressed-key state."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.num_heads = int(args.indexer_n_heads)
        self.num_kv_heads = int(args.indexer_kv_heads)
        self.head_dim = int(args.indexer_head_dim)
        self.token_budget = int(args.indexer_budget)
        self.compress_ratio = int(args.indexer_compress_ratio)
        self.block_topk = self.token_budget // self.compress_ratio
        self.rotary_dim = int(args.head_dim * args.partial_rotary_factor)
        self.rope_theta = args.rope_theta
        self.index_qk_proj = nn.Linear(
            args.hidden_size,
            (self.num_heads + self.num_kv_heads) * self.head_dim,
            bias=False,
        )
        self.q_layernorm = ZeroCenteredRMSNorm(
            self.head_dim, eps=args.rms_norm_eps
        )
        self.k_layernorm = ZeroCenteredRMSNorm(
            self.head_dim, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        cache: QSAIndexCache,
        *,
        physical_kv_length: int,
    ) -> mx.array:
        batch, length, _ = hidden_states.shape
        cache._ensure_batch(batch)
        offsets = list(cache._offsets)
        valid_spans = cache.valid_spans(length)
        projected = self.index_qk_proj(hidden_states)
        query_width = self.num_heads * self.head_dim
        query, raw_keys = mx.split(projected, [query_width], axis=-1)
        query = query.reshape(batch, length, self.num_heads, self.head_dim)
        raw_keys = raw_keys.reshape(batch, length, self.num_kv_heads, self.head_dim)
        if self.num_kv_heads != 1:
            raise ValueError("Qwen4-Exp QSA requires one indexer KV head")
        raw_keys = raw_keys.squeeze(2)
        starts = mx.array(
            [start for start, _ in valid_spans], dtype=mx.int64
        )
        positions = (
            mx.array(offsets, dtype=mx.int64)[:, None]
            + mx.arange(length, dtype=mx.int64)[None, :]
            - starts[:, None]
        )
        query = self.q_layernorm(query)
        query = apply_rotary_positions(
            query,
            positions,
            rotary_dim=self.rotary_dim,
            base=self.rope_theta,
        )

        def transform_group(group: mx.array, start: int) -> mx.array:
            normalized = self.k_layernorm(group[:, None, :])[:, 0, :]
            return apply_rotary_positions(
                normalized[:, None, None, :],
                mx.array([start], dtype=mx.int64),
                rotary_dim=self.rotary_dim,
                base=self.rope_theta,
            )[:, 0, 0, :]

        cache.update(raw_keys, transform_group)
        selected = mx.zeros((batch, length, physical_kv_length), dtype=mx.bool_)
        left_padding = (
            [0] * batch
            if cache.left_padding is None
            else [int(value) for value in cache.left_padding.tolist()]
        )
        for batch_index in range(batch):
            input_start, valid_length = valid_spans[batch_index]
            for query_index in range(input_start, input_start + valid_length):
                logical_position = (
                    offsets[batch_index] + query_index - input_start
                )
                complete_blocks = (logical_position + 1) // self.compress_ratio
                token_indices: list[int] = []
                if complete_blocks:
                    keys = cache.keys_for_blocks(batch_index, complete_blocks)
                    scores = mx.matmul(
                        query[batch_index, query_index].astype(mx.float32),
                        keys.astype(mx.float32).T,
                    )
                    scores = mx.sum(mx.maximum(scores, 0), axis=0) / math.sqrt(
                        self.head_dim
                    )
                    count = min(self.block_topk, complete_blocks)
                    if count == complete_blocks:
                        blocks = list(range(complete_blocks))
                    else:
                        blocks = [
                            int(value)
                            for value in mx.argpartition(
                                scores, kth=-count
                            )[-count:].tolist()
                        ]
                    for block in blocks:
                        start = block * self.compress_ratio
                        token_indices.extend(range(start, start + self.compress_ratio))
                tail_start = complete_blocks * self.compress_ratio
                token_indices.extend(range(tail_start, logical_position + 1))
                if token_indices:
                    physical_indices = [
                        left_padding[batch_index] + index for index in token_indices
                    ]
                    selected[batch_index, query_index, physical_indices] = True
        return selected[:, None, :, :]


class QSAAttention(nn.Module):
    """Qwen sparse attention with independent main-KV and index side caches."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.rotary_dim = int(args.head_dim * args.partial_rotary_factor)
        self.q_proj = nn.Linear(
            args.hidden_size,
            args.num_attention_heads * args.head_dim * 2,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * args.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * args.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * args.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )
        self.q_norm = ZeroCenteredRMSNorm(args.head_dim, eps=args.rms_norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(args.head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            self.rotary_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=None,
            max_position_embeddings=args.max_position_embeddings,
        )
        self.indexer = QSAIndexer(args)

    def __call__(self, x: mx.array, cache: Any | None = None) -> mx.array:
        batch, length, _ = x.shape
        kv_cache = None if cache is None else cache[0]
        index_cache = None if cache is None else cache[1]
        offset = 0 if kv_cache is None else kv_cache.offset
        physical_length = (
            length
            if kv_cache is None
            else int(getattr(kv_cache, "_idx", kv_cache.size())) + length
        )
        if index_cache is None:
            index_cache = QSAIndexCache(self.indexer.compress_ratio)
        selected = self.indexer(
            x,
            index_cache,
            physical_kv_length=physical_length,
        )

        projected = self.q_proj(x).reshape(
            batch, length, self.num_attention_heads, self.head_dim * 2
        )
        queries, gate = mx.split(projected, 2, axis=-1)
        gate = gate.reshape(batch, length, -1)
        keys = self.k_proj(x).reshape(
            batch, length, self.num_key_value_heads, self.head_dim
        )
        values = self.v_proj(x).reshape(
            batch, length, self.num_key_value_heads, self.head_dim
        )
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if kv_cache is not None:
            keys, values = kv_cache.update_and_fetch(keys, values)
        additive_mask = mx.where(
            selected,
            mx.array(0.0, dtype=queries.dtype),
            mx.array(-1e9, dtype=queries.dtype),
        )
        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=kv_cache,
            scale=self.scale,
            mask=additive_mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output * mx.sigmoid(gate))


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def build_layer_multipliers(
    unigram_vocab_size: int,
    ngram_size: int,
    ple_layer_index: int,
    seed: int,
) -> list[int]:
    multiplier_max = ((1 << 63) - 1) // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + 10007 * ple_layer_index
    return [
        2
        * (
            _splitmix64(base_seed + 0x9E3779B97F4A7C15 * (index + 1))
            % half_bound
        )
        + 1
        for index in range(ngram_size)
    ]


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor for divisor in range(3, math.isqrt(value) + 1, 2))


def find_nth_prime_after(start: int, count: int) -> int:
    value = start
    for _ in range(count):
        value += 1
        while not _is_prime(value):
            value += 1
    return value


class ShardedEmbedding(nn.Module):
    """Exact row-wise embedding split matching the checkpoint's 128 shards.

    Keeping shards as independent ``nn.Embedding`` leaves conversion and
    serving bounded: quantization never concatenates the 51B-parameter PLE
    table into one temporary tensor.
    """

    def __init__(self, num_embeddings: int, dims: int, parts: int):
        super().__init__()
        if num_embeddings % parts:
            raise ValueError("sharded embedding rows must divide evenly")
        self.rows_per_shard = num_embeddings // parts
        self.shards = [nn.Embedding(self.rows_per_shard, dims) for _ in range(parts)]

    def __call__(self, indices: mx.array) -> mx.array:
        shard_ids = indices // self.rows_per_shard
        # Evaluating at most 128 small integer ids selects only the embedding
        # shards needed by this request; it never materializes embedding rows.
        used = sorted({int(value) for value in shard_ids.reshape(-1).tolist()})
        output = None
        for shard_id in used:
            shard_id = int(shard_id)
            local = indices - shard_id * self.rows_per_shard
            mask = shard_ids == shard_id
            gathered = self.shards[shard_id](mx.clip(local, 0, self.rows_per_shard - 1))
            gathered = mx.where(mask[..., None], gathered, 0)
            output = gathered if output is None else output + gathered
        if output is None:
            return mx.zeros((*indices.shape, self.shards[0].weight.shape[-1]))
        return output


class NGramEmbedding(nn.Module):
    """Segment-aware hashed bigram/trigram embedding used by PLE."""

    def __init__(
        self,
        args: TextModelArgs,
        *,
        embedding_dim: int,
        ple_layer_index: int,
    ):
        super().__init__()
        self.ngram_size = args.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = args.heads_per_ngram
        self.ngram_heads = self.context_len * self.heads_per_ngram
        self.eos_token_id = (
            args.eos_token_id[0]
            if isinstance(args.eos_token_id, list)
            else args.eos_token_id
        )
        sizes = [
            find_nth_prime_after(
                args.ngram_vocab_size_base - 1,
                ple_layer_index * self.ngram_heads + head + 1,
            )
            for head in range(self.ngram_heads)
        ]
        offsets: list[int] = []
        offset = 0
        for size in sizes:
            offsets.append(offset)
            offset += size
        divisor = args.make_ngram_vocab_size_divisible_by
        padded_rows = ((offset + divisor - 1) // divisor) * divisor
        self.layer_multipliers = mx.array(
            build_layer_multipliers(
                args.vocab_size,
                args.ngram_size,
                ple_layer_index,
                args.seed,
            ),
            dtype=mx.int64,
        )
        self.ngram_heads_vocab_sizes = mx.array(sizes, dtype=mx.int64)
        self.ngram_heads_offsets = mx.array(offsets, dtype=mx.int64)
        self.ngram_embedding = ShardedEmbedding(
            padded_rows,
            embedding_dim // self.ngram_heads,
            args.split_ngram_parts,
        )

    def _shift_right_ignore_eos(self, token_ids: mx.array, shift: int) -> mx.array:
        if shift == 0:
            return token_ids
        _, length = token_ids.shape
        positions = mx.arange(length, dtype=mx.int64)
        eos_positions = mx.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = mx.cummax(eos_positions, axis=1)
        previous_eos = mx.concatenate(
            [mx.full((token_ids.shape[0], 1), -1, dtype=mx.int64), previous_eos_inclusive[:, :-1]],
            axis=1,
        )
        position_in_segment = positions[None, :] - previous_eos - 1
        source_positions = positions - shift
        gather_positions = mx.maximum(source_positions, 0)[None, :]
        gather_positions = mx.broadcast_to(gather_positions, token_ids.shape)
        shifted = mx.take_along_axis(token_ids, gather_positions, axis=1)
        valid = (position_in_segment >= shift) & (source_positions[None, :] >= 0)
        return mx.where(valid, shifted, self.eos_token_id)

    def compute_ids(self, input_ids: mx.array, cache: Any | None = None) -> mx.array:
        input_ids = input_ids.astype(mx.int64)
        if cache is not None and cache[3] is not None:
            previous = cache[3]
        else:
            previous = mx.full(
                (input_ids.shape[0], self.context_len),
                self.eos_token_id,
                dtype=mx.int64,
            )
        history = mx.concatenate([previous, input_ids], axis=1)
        if cache is not None:
            if cache.lengths is not None:
                valid = mx.clip(cache.lengths, 0, input_ids.shape[1])
                positions = (valid[:, None] + mx.arange(self.context_len))[..., None]
                cache[3] = mx.take_along_axis(
                    history[..., None], positions, axis=1
                ).squeeze(-1)
            else:
                cache[3] = mx.contiguous(history[:, -self.context_len :])
        shifted = [
            self._shift_right_ignore_eos(history, shift)
            for shift in range(self.ngram_size)
        ]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed = mx.bitwise_xor(
                    mixed,
                    shifted[position] * self.layer_multipliers[position],
                )
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ids = mx.remainder(mixed[..., None], sizes) + offsets
            blocks.append(ids)
        return mx.concatenate(blocks, axis=-1)[:, -input_ids.shape[1] :]

    def __call__(self, input_ids: mx.array, cache: Any | None = None) -> mx.array:
        ids = self.compute_ids(input_ids, cache)
        return self.ngram_embedding(ids).flatten(-2)


class PLELayer(nn.Module):
    """Exact PLE projection, gating, and dilation-three short convolution."""

    def __init__(
        self,
        args: TextModelArgs,
        *,
        ple_layer_index: int,
    ):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        hc_width = args.hc_count * args.hidden_size
        embed_dim = int(args.ple_embed_dim)
        self.ple_embedding = NGramEmbedding(
            args,
            embedding_dim=embed_dim,
            ple_layer_index=ple_layer_index,
        )
        self.key_proj = nn.Linear(embed_dim, hc_width, bias=False)
        self.value_proj = nn.Linear(embed_dim, args.hidden_size, bias=False)
        norm_kwargs = {
            "group_size": args.hidden_size,
            "eps": args.rms_norm_eps,
        }
        self.norm_key = ZeroCenteredRMSNorm(hc_width, **norm_kwargs)
        self.norm_query = ZeroCenteredRMSNorm(hc_width, **norm_kwargs)
        self.norm_conv = ZeroCenteredRMSNorm(hc_width, **norm_kwargs)
        self.conv_state_len = (args.ple_conv_kernel_size - 1) * args.ngram_size
        self.conv1d = nn.Conv1d(
            hc_width,
            hc_width,
            kernel_size=args.ple_conv_kernel_size,
            dilation=args.ngram_size,
            groups=hc_width,
            bias=False,
        )

    def _short_conv(self, x: mx.array, cache: Any | None) -> mx.array:
        if cache is not None and cache[2] is not None:
            state = cache[2]
        else:
            state = mx.zeros(
                (x.shape[0], self.conv_state_len, x.shape[-1]), dtype=x.dtype
            )
        conv_input = mx.concatenate([state, x], axis=1)
        if cache is not None:
            if cache.lengths is not None:
                valid = mx.clip(cache.lengths, 0, x.shape[1])
                positions = (
                    valid[:, None] + mx.arange(self.conv_state_len)
                )[..., None]
                cache[2] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[2] = mx.contiguous(
                    conv_input[:, -self.conv_state_len :, :]
                )
        return nn.silu(self.conv1d(conv_input))

    def __call__(
        self,
        hidden_states: mx.array,
        input_ids: mx.array,
        cache: Any | None,
        mask: mx.array | None = None,
    ) -> mx.array:
        if mask is not None:
            input_ids = mx.where(mask, input_ids, self.ple_embedding.eos_token_id)
        embeddings = self.ple_embedding(input_ids, cache)
        keys = self.norm_key(self.key_proj(embeddings)).reshape(
            *hidden_states.shape[:-1], self.hc_count, self.hidden_size
        )
        values = self.value_proj(embeddings)
        queries = self.norm_query(hidden_states).reshape(
            *hidden_states.shape[:-1], self.hc_count, self.hidden_size
        )
        gate = mx.sum(keys * queries, axis=-1, keepdims=True) / math.sqrt(
            self.hidden_size
        )
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated = mx.sigmoid(gate) * values[..., None, :]
        gated = gated.flatten(-2)
        normalized = self.norm_conv(gated)
        if mask is not None:
            gated = mx.where(mask[..., None], gated, 0)
            normalized = mx.where(mask[..., None], normalized, 0)
        return gated + self._short_conv(normalized, cache)


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_index: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_index]
        self.is_linear = self.layer_type == "linear_attention"
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = QSAAttention(args)
        self.mlp = SparseMoeBlock(args)
        ple_index = (
            args.ple_layer_ids.index(layer_index + 1)
            if layer_index + 1 in args.ple_layer_ids
            else None
        )
        self.ple = (
            PLELayer(args, ple_layer_index=ple_index)
            if ple_index is not None
            else None
        )
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)

    @staticmethod
    def _combine(
        output: mx.array,
        residual: mx.array,
        injection: mx.array,
    ) -> mx.array:
        injected = output[..., None, :] * injection[..., :, None]
        return residual + injected.flatten(-2)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        input_ids: mx.array,
        mask: mx.array | None,
        cache: Any | None,
    ) -> mx.array:
        if self.ple is not None:
            hidden_states = hidden_states + self.ple(
                hidden_states,
                input_ids,
                cache,
                mask,
            )
        mixed, residual, injection = self.attn_hyper_connection(hidden_states)
        if self.is_linear:
            output = self.linear_attn(mixed, mask=mask, cache=cache)
        else:
            output = self.self_attn(mixed, cache=cache)
        hidden_states = self._combine(output, residual, injection)

        mixed, residual, injection = self.mlp_hyper_connection(hidden_states)
        output = self.mlp(mixed)
        return self._combine(output, residual, injection)


class Qwen4ExpTextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer_index)
            for layer_index in range(args.num_hidden_layers)
        ]
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        hidden_states = (
            input_embeddings
            if input_embeddings is not None
            else self.embed_tokens(inputs)
        )
        hidden_states = mx.tile(hidden_states, (1, 1, self.args.hc_count))
        if cache is None:
            cache = [None] * len(self.layers)
        linear_index = next(
            (index for index, layer in enumerate(self.layers) if layer.is_linear),
            None,
        )
        linear_mask = (
            None
            if linear_index is None
            else create_ssm_mask(hidden_states, cache[linear_index])
        )
        for layer, layer_cache in zip(self.layers, cache):
            hidden_states = layer(
                hidden_states,
                input_ids=inputs,
                mask=linear_mask if layer.is_linear else None,
                cache=layer_cache,
            )
        return self.hyper_connection_mixer(hidden_states)


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen4ExpTextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: list[Any] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        hidden = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=4 if layer.ple is not None else 2))
            else:
                caches.append(
                    CacheList(
                        KVCache(),
                        QSAIndexCache(layer.self_attn.indexer.compress_ratio),
                    )
                )
        return caches

    def sanitize(self, weights):
        weights = {
            key: value
            for key, value in weights.items()
            if not key.startswith("mtp.") and ".visual." not in key
        }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        sanitized = {}
        for key, value in weights.items():
            if key.endswith("conv1d.weight") and value.ndim == 3 and value.shape[1] == 1:
                value = value.moveaxis(2, 1)
            if ".mlp.experts.gate_up_proj" in key:
                gate_up = value
                midpoint = gate_up.shape[-2] // 2
                base = key.replace("experts.gate_up_proj", "switch_mlp")
                sanitized[f"{base}.gate_proj.weight"] = gate_up[..., :midpoint, :]
                sanitized[f"{base}.up_proj.weight"] = gate_up[..., midpoint:, :]
                continue
            if ".mlp.experts.down_proj" in key:
                key = key.replace(
                    "experts.down_proj", "switch_mlp.down_proj.weight"
                )
            if ".ngram_embedding.shard_" in key:
                prefix, suffix = key.split(".ngram_embedding.shard_", 1)
                shard, leaf = suffix.split(".", 1)
                key = f"{prefix}.ngram_embedding.shards.{int(shard)}.{leaf}"
            sanitized[key] = value
        return sanitized

    @property
    def quant_predicate(self):
        def predicate(path, _module):
            if ".ple.ple_embedding.ngram_embedding.shards." in path:
                # The checkpoint's PLE tables have width 160.  Group 32 is the
                # largest established MLX group that divides that width, so it
                # preserves the source shape without a padding/slicing format.
                return {"group_size": 32, "bits": 4}
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    @property
    def cast_predicate(self):
        return lambda path: not path.endswith("A_log")


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = TextModel(TextModelArgs.from_dict(args.text_config))

    def __call__(self, inputs: mx.array, cache=None, input_embeddings=None):
        return self.language_model(inputs, cache, input_embeddings)

    @property
    def model(self):
        return self.language_model.model

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def sanitize(self, weights):
        mapped = {}
        for key, value in weights.items():
            if key.startswith("model.visual") or key.startswith("vision_tower"):
                continue
            if key.startswith("mtp."):
                continue
            if key.startswith("model.language_model"):
                key = key.replace(
                    "model.language_model", "language_model.model", 1
                )
            elif not key.startswith("language_model."):
                key = f"language_model.{key}"
            mapped[key] = value
        return self.language_model.sanitize(mapped)

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate
