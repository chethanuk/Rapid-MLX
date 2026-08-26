import json
from dataclasses import asdict

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_lm.models.cache import ArraysCache, BatchKVCache, CacheList, KVCache

import vllm_mlx.models.qwen4_exp as qwen4_exp
from scripts import qwen38_streaming_convert as converter
from scripts.qwen38_streaming_convert import quantized_tensor_names
from vllm_mlx.models.qwen4_exp import (
    GatedDeltaNet,
    GatedResidual,
    Model,
    ModelArgs,
    NGramEmbedding,
    PLELayer,
    QSAAttention,
    QSAIndexer,
    ShardedEmbedding,
    TextModelArgs,
    ZeroCenteredRMSNorm,
    apply_qwen4_exp_rope,
    build_layer_multipliers,
)
from vllm_mlx.models.qwen4_exp_cache import QSAIndexCache


def _args(**overrides):
    values = {
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "vocab_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 3,
        "linear_key_head_dim": 4,
        "linear_value_head_dim": 4,
        "linear_conv_kernel_dim": 3,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 4,
        "shared_expert_intermediate_size": 4,
        "hc_count": 4,
        "hc_lowrank": 3,
        "layer_types": ["linear_attention", "full_attention"],
        "indexer_n_heads": 2,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 4,
        "indexer_budget": 8,
        "indexer_compress_ratio": 2,
        "ple_layer_ids": [],
        "eos_token_id": 31,
    }
    values.update(overrides)
    return TextModelArgs(**values)


def _repair_fixture(tmp_path, shard_tensors, weight_map):
    output = tmp_path / "converted"
    output.mkdir()
    for shard_name, tensors in shard_tensors.items():
        mx.save_safetensors(str(output / shard_name), tensors)
    (output / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 8}, "weight_map": weight_map})
    )
    converter._write_sha256sums(output)
    return output


def test_config_normalizes_checkpoint_full_attention_to_qsa():
    args = _args()
    assert args.layer_types == ["linear_attention", "qwen_sparse_attention"]
    assert args.linear_num_value_heads // args.linear_num_key_heads == 3


def test_config_rejects_partial_indexer_contract():
    with pytest.raises(ValueError, match="complete indexer contract"):
        _args(indexer_budget=None)


def test_qwen4_rope_uses_reference_half_split_pairing():
    values = np.array([[[[0.25, -0.5, 0.75, 1.0]]]], dtype=np.float32)
    output = apply_qwen4_exp_rope(
        mx.array(values),
        mx.array([[39]], dtype=mx.int64),
        rotary_dim=4,
        base=10_000_000,
    )
    frequencies = 1.0 / (10_000_000 ** (np.arange(0, 4, 2, dtype=np.float32) / 4))
    angles = 39 * frequencies
    first = values[..., :2]
    second = values[..., 2:]
    expected = np.concatenate(
        [
            first * np.cos(angles) - second * np.sin(angles),
            second * np.cos(angles) + first * np.sin(angles),
        ],
        axis=-1,
    )
    np.testing.assert_allclose(np.asarray(output), expected, rtol=1e-5, atol=1e-5)


def test_zero_centered_grouped_rms_norm_matches_numpy():
    norm = ZeroCenteredRMSNorm(8, group_size=4, eps=1e-6)
    norm.weight = mx.array(np.linspace(-0.2, 0.2, 8, dtype=np.float32))
    x_np = np.arange(1, 17, dtype=np.float32).reshape(1, 2, 8) / 10
    out = norm(mx.array(x_np))
    grouped = x_np.reshape(1, 2, 2, 4)
    expected = grouped / np.sqrt(np.mean(grouped**2, axis=-1, keepdims=True) + 1e-6)
    expected *= (1 + np.linspace(-0.2, 0.2, 8, dtype=np.float32)).reshape(2, 4)
    np.testing.assert_allclose(
        np.array(out), expected.reshape(x_np.shape), rtol=2e-5, atol=2e-5
    )


def test_gated_residual_matches_reference_equations():
    args = _args()
    layer = GatedResidual(args)
    rng = np.random.default_rng(7)
    layer.hc_norm.weight = mx.array(rng.normal(0, 0.1, (32,)).astype(np.float32))
    layer.input_mix_weight_down.weight = mx.array(
        rng.normal(0, 0.1, (3, 32)).astype(np.float32)
    )
    layer.input_mix_weight_up.weight = mx.array(
        rng.normal(0, 0.1, (32, 3)).astype(np.float32)
    )
    layer.block_inject_weight.weight = mx.array(
        rng.normal(0, 0.1, (4, 32)).astype(np.float32)
    )
    x_np = rng.normal(0, 0.2, (1, 2, 32)).astype(np.float32)
    mixed, residual, injection = layer(mx.array(x_np))

    grouped = x_np.reshape(1, 2, 4, 8)
    weight = np.array(layer.hc_norm.weight).reshape(4, 8)
    normalized = grouped / np.sqrt(np.mean(grouped**2, axis=-1, keepdims=True) + 1e-6)
    normalized *= 1 + weight
    flat = normalized.reshape(1, 2, 32)
    down = flat @ np.array(layer.input_mix_weight_down.weight).T / 4
    silu = down / (1 + np.exp(-down))
    mix = 1 / (1 + np.exp(-(silu @ np.array(layer.input_mix_weight_up.weight).T)))
    expected_mixed = np.mean(mix.reshape(1, 2, 4, 8) * normalized, axis=-2)
    expected_injection = 2 / (
        1 + np.exp(-(flat @ np.array(layer.block_inject_weight.weight).T / 4))
    )
    np.testing.assert_allclose(np.array(mixed), expected_mixed, rtol=3e-5, atol=3e-5)
    np.testing.assert_array_equal(np.array(residual), x_np)
    np.testing.assert_allclose(
        np.array(injection), expected_injection, rtol=3e-5, atol=3e-5
    )


def test_gdn_ratio_three_state_shapes_and_cached_decode():
    args = _args()
    layer = GatedDeltaNet(args)
    cache = ArraysCache(size=2)
    prompt = mx.zeros((1, 3, args.hidden_size), dtype=mx.float32)
    prompt_out = layer(prompt, cache=cache)
    mx.eval(prompt_out, cache.state)
    assert prompt_out.shape == (1, 3, args.hidden_size)
    assert cache[0].shape == (1, args.linear_conv_kernel_dim - 1, 20)
    assert cache[1].shape == (1, 3, 4, 4)

    token_out = layer(mx.zeros((1, 1, args.hidden_size)), cache=cache)
    mx.eval(token_out, cache.state)
    assert token_out.shape == (1, 1, args.hidden_size)
    assert cache[0].shape == (1, 2, 20)
    assert cache[1].shape == (1, 3, 4, 4)


def test_sharded_embedding_preserves_global_row_identity():
    embedding = ShardedEmbedding(num_embeddings=16, dims=4, parts=4)
    for shard_id, shard in enumerate(embedding.shards):
        rows = np.arange(shard_id * 4, shard_id * 4 + 4, dtype=np.float32)
        shard.weight = mx.array(np.repeat(rows[:, None], 4, axis=1))
    ids = mx.array([[0, 3, 4, 9, 15]])
    output = embedding(ids)
    expected = np.repeat(
        np.array([[0, 3, 4, 9, 15]], dtype=np.float32)[..., None], 4, axis=-1
    )
    np.testing.assert_array_equal(np.array(output), expected)


def test_ngram_multipliers_match_released_reference_constants():
    assert build_layer_multipliers(248320, 3, 0, 1234) == [
        23703573157769,
        20109073645365,
        8052911324071,
    ]


def _ple_args():
    return _args(
        layer_types=["linear_attention", "qwen_sparse_attention"],
        ple_layer_ids=[1],
        ple_embed_dim=16,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        rope_parameters={"rope_theta": 10_000_000, "partial_rotary_factor": 0.5},
    )


def test_ngram_cache_matches_one_shot_across_request_chunks():
    args = _ple_args()
    embedding = NGramEmbedding(args, embedding_dim=16, ple_layer_index=0)
    one_shot = embedding.compute_ids(mx.array([[1, 2, 3, 4]]))

    cache = ArraysCache(size=4)
    first = embedding.compute_ids(mx.array([[1, 2]]), cache)
    second = embedding.compute_ids(mx.array([[3, 4]]), cache)
    mx.eval(one_shot, first, second, cache.state)
    np.testing.assert_array_equal(np.array(first), np.array(one_shot[:, :2]))
    np.testing.assert_array_equal(np.array(second), np.array(one_shot[:, 2:]))
    np.testing.assert_array_equal(np.array(cache[3]), np.array([[3, 4]]))


def test_ngram_context_resets_at_eos_boundary():
    args = _ple_args()
    embedding = NGramEmbedding(args, embedding_dim=16, ple_layer_index=0)
    with_history = embedding.compute_ids(mx.array([[7, 8, 31, 9]]))
    fresh_segment = embedding.compute_ids(mx.array([[9]]))
    mx.eval(with_history, fresh_segment)
    np.testing.assert_array_equal(
        np.array(with_history[:, -1]), np.array(fresh_segment[:, -1])
    )


def test_ple_state_shapes_survive_cached_decode():
    args = _ple_args()
    ple = PLELayer(args, ple_layer_index=0)
    cache = ArraysCache(size=4)
    hidden = mx.zeros((1, 3, args.hc_count * args.hidden_size))
    output = ple(hidden, mx.array([[1, 2, 3]]), cache)
    mx.eval(output, cache.state)
    assert output.shape == hidden.shape
    assert cache[2].shape == (1, 9, args.hc_count * args.hidden_size)
    assert cache[3].shape == (1, 2)

    token_output = ple(
        mx.zeros((1, 1, args.hc_count * args.hidden_size)),
        mx.array([[4]]),
        cache,
    )
    mx.eval(token_output, cache.state)
    assert token_output.shape == (1, 1, args.hc_count * args.hidden_size)
    assert cache[2].shape == (1, 9, args.hc_count * args.hidden_size)
    np.testing.assert_array_equal(np.array(cache[3]), np.array([[3, 4]]))


def test_ple_right_padded_batch_caches_only_each_rows_valid_history():
    args = _ple_args()
    ple = PLELayer(args, ple_layer_index=0)
    batch_cache = ArraysCache(size=4)
    batch_cache.prepare(lengths=[3, 1])
    batch_ids = mx.array([[1, 2, 3], [7, 0, 0]])
    mask = mx.array([[True, True, True], [True, False, False]])
    batch_output = ple(
        mx.zeros((2, 3, args.hc_count * args.hidden_size)),
        batch_ids,
        batch_cache,
        mask,
    )

    single_cache = ArraysCache(size=4)
    single_output = ple(
        mx.zeros((1, 1, args.hc_count * args.hidden_size)),
        mx.array([[7]]),
        single_cache,
    )
    mx.eval(batch_output, single_output, batch_cache.state, single_cache.state)
    np.testing.assert_array_equal(np.array(batch_cache[3][1]), np.array([31, 7]))
    np.testing.assert_allclose(
        np.array(batch_cache[2][1]),
        np.array(single_cache[2][0]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_qsa_cache_keeps_only_raw_ring_and_persistent_compressed_keys():
    cache = QSAIndexCache(compress_ratio=2)

    def transform(group, start):
        return group + start

    compressed = cache.update(mx.array([[[1.0], [3.0], [5.0], [7.0]]]), transform)
    mx.eval(compressed, cache.state)
    np.testing.assert_array_equal(np.array(compressed), np.array([[[2.0], [8.0]]]))
    assert cache.offset == 4
    assert cache.raw_ring.shape == (1, 2, 1)

    unchanged = cache.update(mx.array([[[9.0]]]), transform)
    mx.eval(unchanged, cache.state)
    np.testing.assert_array_equal(np.array(unchanged), np.array([[[2.0], [8.0]]]))
    np.testing.assert_array_equal(np.array(cache.raw_ring), np.array([[[9.0], [7.0]]]))
    assert cache.offset == 5
    assert cache.is_trimmable()

    restored = QSAIndexCache.from_state(cache.state, cache.meta_state)
    assert restored.offset == 5
    assert restored.compress_ratio == 2
    np.testing.assert_array_equal(
        np.array(restored.compressed_keys), np.array([[[2.0], [8.0]]])
    )


@pytest.mark.parametrize("length", [8, 9])
def test_qsa_cache_rewinds_recoverable_group_and_recomputes_divergence(length):
    def transform(group, start):
        return group + start

    original = QSAIndexCache(compress_ratio=4)
    values = np.arange(1, length + 1, dtype=np.float32)
    original.update(mx.array(values.reshape(1, -1, 1)), transform)
    assert original.trim(1) == 1
    original.update(mx.array([[[99.0]]]), transform)

    cold = QSAIndexCache(compress_ratio=4)
    expected_values = np.concatenate([values[:-1], np.array([99.0])])
    cold.update(mx.array(expected_values.reshape(1, -1, 1)), transform)
    mx.eval(original.state, cold.state)
    assert original.offset == cold.offset == length
    np.testing.assert_array_equal(np.array(original.state[1]), np.array(cold.state[1]))
    np.testing.assert_array_equal(np.array(original.raw_ring), np.array(cold.raw_ring))


def test_qsa_cache_refuses_rewind_beyond_retained_raw_group():
    cache = QSAIndexCache(compress_ratio=4)
    cache.update(
        mx.arange(9, dtype=mx.float32).reshape(1, 9, 1),
        lambda group, start: group + start,
    )
    assert cache.trim(2) == 0
    assert cache.offset == 9


def test_scheduler_rollback_preflights_qsa_cachelist_for_full_rejection():
    """A multi-token rejection cannot trim KV before QSA refuses it."""
    from vllm_mlx.cache_rollback import can_trim, trim_all

    kv = KVCache()
    keys = mx.arange(9, dtype=mx.float32).reshape(1, 1, 9, 1)
    kv.update_and_fetch(keys, -keys)
    qsa = QSAIndexCache(compress_ratio=4)
    qsa.update(
        mx.arange(9, dtype=mx.float32).reshape(1, 9, 1),
        lambda group, start: group + start,
    )
    cache = CacheList(kv, qsa)

    assert cache.is_trimmable()
    assert not can_trim(cache, 2)
    assert not trim_all([cache], 2)
    assert kv.offset == qsa.offset == 9

    assert can_trim(cache, 1)
    assert trim_all([cache], 1)
    assert kv.offset == qsa.offset == 8


def test_suffix_scheduler_falls_through_before_qsa_multitoken_verify():
    from vllm_mlx.scheduler import _install_suffix_decoding

    kv = KVCache()
    keys = mx.arange(9, dtype=mx.float32).reshape(1, 1, 9, 1)
    kv.update_and_fetch(keys, -keys)
    qsa = QSAIndexCache(compress_ratio=4)
    qsa.update(
        mx.arange(9, dtype=mx.float32).reshape(1, 9, 1),
        lambda group, start: group + start,
    )

    class GenerationBatch:
        _next_tokens = mx.array([7], dtype=mx.int32)
        uids = [11]
        logits_processors = []
        prompt_cache = [CacheList(kv, qsa)]
        tokens = [[]]
        model = None

        def __init__(self):
            self.original_calls = 0

        def _step(self):
            self.original_calls += 1
            return [7], []

        def next(self):
            return []

    class BatchGenerator:
        def __init__(self):
            self._generation_batch = GenerationBatch()

        def remove(self, _uids, return_prompt_caches=False):
            return {} if return_prompt_caches else None

    class TwoTokenDrafter:
        max_draft_tokens = 2

        def add_generated_token(self, _token):
            return None

        def get_draft(self):
            return [8, 9]

    batch_generator = BatchGenerator()
    _install_suffix_decoding(
        batch_generator,
        model=None,
        profile=None,
        max_draft=2,
        max_suffix_len=2,
        min_confidence=0.0,
        requests={},
        uid_to_request_id={},
    )
    generation_batch = batch_generator._generation_batch
    generation_batch._suffix_drafters[11] = TwoTokenDrafter()

    assert generation_batch._step() == ([7], [])
    assert generation_batch.original_calls == 1
    assert generation_batch._suffix_stats["ft_non_trimmable_cache"] == 1
    assert kv.offset == qsa.offset == 9


def test_qsa_attention_prefill_and_decode_keep_both_cache_owners_aligned():
    args = _args(
        indexer_budget=2,
        indexer_compress_ratio=2,
        rope_parameters={"rope_theta": 10_000_000, "partial_rotary_factor": 0.5},
    )
    attention = QSAAttention(args)
    cache = CacheList(KVCache(), QSAIndexCache(compress_ratio=2))
    prompt = mx.zeros((1, 5, args.hidden_size))
    prompt_output = attention(prompt, cache)
    mx.eval(prompt_output, cache.state)
    assert prompt_output.shape == (1, 5, args.hidden_size)
    assert cache[0].offset == 5
    assert cache[1].offset == 5
    assert cache[1]._compressed_count == 2

    token_output = attention(mx.zeros((1, 1, args.hidden_size)), cache)
    mx.eval(token_output, cache.state)
    assert token_output.shape == (1, 1, args.hidden_size)
    assert cache[0].offset == 6
    assert cache[1].offset == 6
    assert cache[1]._compressed_count == 3


def test_qsa_attention_uses_reference_dense_path_below_sparse_budget(monkeypatch):
    args = _args(
        indexer_budget=8,
        indexer_compress_ratio=2,
        rope_parameters={"rope_theta": 10_000_000, "partial_rotary_factor": 0.5},
    )
    attention = QSAAttention(args)
    cache = CacheList(KVCache(), QSAIndexCache(compress_ratio=2))
    observed = []

    def fake_attention(queries, keys, values, *, cache, scale, mask):
        observed.append(mask)
        return mx.zeros_like(queries)

    monkeypatch.setattr(
        "vllm_mlx.models.qwen4_exp.scaled_dot_product_attention",
        fake_attention,
    )
    output = attention(mx.zeros((1, 5, args.hidden_size)), cache)
    mx.eval(output, cache.state)

    assert observed == ["causal"]
    assert cache[0].offset == cache[1].offset == 5


def test_qsa_batch_prefill_builds_mask_before_kv_update(monkeypatch):
    args = _args(indexer_budget=8, indexer_compress_ratio=2)
    attention = QSAAttention(args)
    qsa = QSAIndexCache(compress_ratio=2)
    qsa.left_padding = mx.array([0])
    cache = CacheList(BatchKVCache([0]), qsa)
    observed = []

    def fake_attention(queries, keys, values, *, cache, scale, mask):
        observed.append((keys.shape[-2], mask.shape[-1]))
        return mx.zeros_like(queries)

    monkeypatch.setattr(
        "vllm_mlx.models.qwen4_exp.scaled_dot_product_attention",
        fake_attention,
    )
    output = attention(mx.zeros((1, 5, args.hidden_size)), cache)
    mx.eval(output, cache.state)
    assert observed == [(5, 5)]


def test_scheduler_mid_prefill_restores_qsa_cachelist():
    """The live restore path recognizes the same vendored QSA side-cache."""
    from vllm_mlx.scheduler import Scheduler

    scheduler = Scheduler.__new__(Scheduler)
    kv = KVCache()
    values = mx.arange(20, dtype=mx.float32).reshape(1, 1, 5, 4)
    kv.update_and_fetch(values, -values)
    qsa = QSAIndexCache(compress_ratio=2)
    qsa.update(
        mx.arange(20, dtype=mx.float32).reshape(1, 5, 4),
        lambda group, start: group + start,
    )
    original = CacheList(kv, qsa)

    restored = scheduler._reconstruct_cache_from_states(
        [
            {
                "state": original.state,
                "meta_state": original.meta_state,
                "class_ref": CacheList,
            }
        ]
    )

    assert restored is not None
    restored_qsa = restored[0].caches[1]
    assert isinstance(restored_qsa, QSAIndexCache)
    assert restored_qsa.offset == qsa.offset
    np.testing.assert_array_equal(
        np.array(restored_qsa.state[1]), np.array(qsa.state[1])
    )


def test_qsa_sparse_scores_use_one_reference_batched_matmul(monkeypatch):
    args = _args(
        indexer_budget=2,
        indexer_compress_ratio=2,
        rope_parameters={"rope_theta": 10_000_000, "partial_rotary_factor": 0.5},
    )
    indexer = QSAIndexer(args)
    cache = QSAIndexCache(compress_ratio=2)
    original = qwen4_exp.mx.matmul
    shapes = []

    def record_matmul(left, right):
        shapes.append((left.shape, right.shape))
        return original(left, right)

    monkeypatch.setattr(qwen4_exp.mx, "matmul", record_matmul)
    selected = indexer(
        mx.zeros((1, 6, args.hidden_size), dtype=mx.bfloat16),
        cache,
        physical_kv_length=6,
    )
    mx.eval(selected)
    assert shapes == [
        (
            (args.indexer_n_heads, 6, args.indexer_head_dim),
            (args.indexer_head_dim, 3),
        )
    ]


def test_qsa_cache_uses_standard_batch_lifecycle_without_rebuilding_history():
    def transform(group, start):
        return group + start

    first = QSAIndexCache(compress_ratio=2)
    second = QSAIndexCache(compress_ratio=2)
    first.update(mx.array([[[1.0], [3.0], [5.0]]]), transform)
    second.update(mx.array([[[2.0], [4.0], [6.0], [8.0], [10.0]]]), transform)

    batch = QSAIndexCache.merge([first, second])
    assert isinstance(batch, ArraysCache)
    assert batch._offsets == [3, 5]
    assert batch._compressed_counts == [1, 2]
    np.testing.assert_array_equal(np.array(batch.left_padding), np.array([2, 0]))

    batch.update(mx.array([[[7.0]], [[12.0]]]), transform)
    mx.eval(batch.state)
    assert batch._offsets == [4, 6]
    assert batch._compressed_counts == [2, 3]

    extracted = batch.extract(0)
    assert extracted.offset == 4
    np.testing.assert_array_equal(
        np.array(extracted.compressed_keys[:, :2]), np.array([[[2.0], [8.0]]])
    )

    batch.filter(mx.array([1]))
    assert batch._offsets == [6]
    np.testing.assert_array_equal(np.array(batch.left_padding), np.array([0]))
    batch.extend(QSAIndexCache.merge([extracted]))
    assert batch._offsets == [6, 4]
    np.testing.assert_array_equal(np.array(batch.left_padding), np.array([0, 2]))


def test_qsa_cache_skips_right_padding_and_preserves_physical_alignment():
    cache = QSAIndexCache(compress_ratio=2)
    # This is the same adoption seam used by mlx-lm's _make_cache for an
    # ArraysCache subclass.
    cache.left_padding = mx.array([0, 0])
    cache.prepare(lengths=[3, 1], right_padding=[0, 2])
    cache.update(
        mx.array(
            [
                [[1.0], [3.0], [5.0]],
                [[7.0], [99.0], [99.0]],
            ]
        ),
        lambda group, start: group + start,
    )
    assert cache._offsets == [3, 1]
    assert cache._compressed_counts == [1, 0]
    cache.finalize()
    np.testing.assert_array_equal(np.array(cache.left_padding), np.array([0, 2]))


def test_qsa_cache_skips_fresh_batch_left_padding():
    cache = QSAIndexCache(compress_ratio=2)
    cache.left_padding = mx.array([2, 0])
    cache.update(
        mx.array(
            [
                [[99.0], [99.0], [1.0], [3.0], [5.0]],
                [[2.0], [4.0], [6.0], [8.0], [10.0]],
            ]
        ),
        lambda group, start: group + start,
    )
    mx.eval(cache.state)
    assert cache._offsets == [3, 5]
    assert cache._compressed_counts == [1, 2]
    np.testing.assert_array_equal(
        np.array(cache.compressed_keys[0, :1]), np.array([[2.0]])
    )


def test_qsa_attention_continuous_batch_decode_matches_cache_lengths():
    args = _args(
        indexer_budget=2,
        indexer_compress_ratio=2,
        rope_parameters={"rope_theta": 10_000_000, "partial_rotary_factor": 0.5},
    )
    attention = QSAAttention(args)
    caches = []
    for length in (3, 5):
        cache = CacheList(KVCache(), QSAIndexCache(compress_ratio=2))
        output = attention(mx.zeros((1, length, args.hidden_size)), cache)
        mx.eval(output, cache.state)
        caches.append(cache)

    batch_cache = CacheList.merge(caches)
    output = attention(mx.zeros((2, 1, args.hidden_size)), batch_cache)
    mx.eval(output, batch_cache.state)
    assert output.shape == (2, 1, args.hidden_size)
    np.testing.assert_array_equal(np.array(batch_cache[0].offset), np.array([4, 6]))
    np.testing.assert_array_equal(np.array(batch_cache[1].offset), np.array([4, 6]))


def test_qsa_attention_fresh_left_padded_batch_aligns_with_main_kv():
    args = _args(
        indexer_budget=2,
        indexer_compress_ratio=2,
        rope_parameters={"rope_theta": 10_000_000, "partial_rotary_factor": 0.5},
    )
    attention = QSAAttention(args)
    qsa = QSAIndexCache(compress_ratio=2)
    # mlx-lm adopts an ArraysCache subclass by assigning this metadata.
    qsa.left_padding = mx.array([2, 0])
    cache = CacheList(BatchKVCache([2, 0]), qsa)
    output = attention(mx.zeros((2, 5, args.hidden_size)), cache)
    mx.eval(output, cache.state)
    assert output.shape == (2, 5, args.hidden_size)
    np.testing.assert_array_equal(np.array(cache[0].offset), np.array([3, 5]))
    np.testing.assert_array_equal(np.array(cache[1].offset), np.array([3, 5]))


def test_complete_synthetic_text_model_prefill_and_decode():
    args = _ple_args()
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=asdict(args)))
    cache = model.make_cache()
    prompt = mx.array([[1, 2, 3]])
    logits = model(prompt, cache=cache)
    mx.eval(logits, [layer.state for layer in cache])
    assert logits.shape == (1, 3, args.vocab_size)
    assert cache[0][0].shape == (1, 2, 20)
    assert cache[0][2].shape == (1, 9, args.hc_count * args.hidden_size)
    assert cache[1][0].offset == 3
    assert cache[1][1].offset == 3

    next_logits = model(mx.array([[4]]), cache=cache)
    mx.eval(next_logits, [layer.state for layer in cache])
    assert next_logits.shape == (1, 1, args.vocab_size)
    assert cache[1][0].offset == 4
    assert cache[1][1].offset == 4


def test_sanitize_preserves_ple_shards_and_maps_experts_without_concat():
    args = _ple_args()
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=asdict(args)))
    weights = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": mx.zeros((4, 8, 8)),
        "model.language_model.layers.0.mlp.experts.down_proj": mx.zeros((4, 8, 4)),
        "model.language_model.layers.0.mlp.experts.gate_up_proj.scales": mx.zeros(
            (4, 8, 1)
        ),
        "model.language_model.layers.0.mlp.experts.gate_up_proj.biases": mx.zeros(
            (4, 8, 1)
        ),
        "model.language_model.layers.0.mlp.experts.down_proj.scales": mx.zeros(
            (4, 8, 1)
        ),
        "model.language_model.layers.0.mlp.experts.down_proj.biases": mx.zeros(
            (4, 8, 1)
        ),
        "model.language_model.layers.0.linear_attn.conv1d.weight": mx.zeros((20, 1, 3)),
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_3.weight": mx.zeros(
            (32, 1)
        ),
        "model.visual.blocks.0.weight": mx.zeros((1,)),
        "mtp.layers.0.weight": mx.zeros((1,)),
    }
    sanitized = model.sanitize(weights)
    assert "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight" in sanitized
    assert "language_model.model.layers.0.mlp.switch_mlp.up_proj.weight" in sanitized
    assert "language_model.model.layers.0.mlp.switch_mlp.down_proj.weight" in sanitized
    for projection in ("gate_proj", "up_proj", "down_proj"):
        for auxiliary in ("scales", "biases"):
            assert (
                f"language_model.model.layers.0.mlp.switch_mlp."
                f"{projection}.{auxiliary}" in sanitized
            )
    conv = sanitized["language_model.model.layers.0.linear_attn.conv1d.weight"]
    assert conv.shape == (20, 3, 1)
    assert (
        "language_model.model.layers.0.ple.ple_embedding.ngram_embedding.shards.3.weight"
        in sanitized
    )
    assert all("visual" not in key and not key.startswith("mtp") for key in sanitized)


def test_converter_quantized_keys_match_loader_sanitizer_contract():
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=asdict(_ple_args())))
    ordinary = "model.language_model.layers.0.self_attn.q_proj.weight"
    assert quantized_tensor_names(ordinary) == (
        ordinary,
        "model.language_model.layers.0.self_attn.q_proj.scales",
        "model.language_model.layers.0.self_attn.q_proj.biases",
    )

    emitted = {}
    for name, shape in (
        (ordinary, (8, 8)),
        ("model.language_model.layers.0.mlp.experts.gate_up_proj", (4, 8, 8)),
        ("model.language_model.layers.0.mlp.experts.down_proj", (4, 8, 4)),
    ):
        weight, scales, biases = quantized_tensor_names(name)
        emitted[weight] = mx.zeros(shape)
        aux_shape = (*shape[:-2], shape[-2], 1)
        emitted[scales] = mx.zeros(aux_shape)
        emitted[biases] = mx.zeros(aux_shape)

    sanitized = model.sanitize(emitted)
    expected = {
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.q_proj.scales",
        "language_model.model.layers.0.self_attn.q_proj.biases",
    }
    for projection in ("gate_proj", "up_proj", "down_proj"):
        expected.update(
            {
                f"language_model.model.layers.0.mlp.switch_mlp.{projection}.weight",
                f"language_model.model.layers.0.mlp.switch_mlp.{projection}.scales",
                f"language_model.model.layers.0.mlp.switch_mlp.{projection}.biases",
            }
        )
    assert set(sanitized) == expected


def test_quantized_aux_repair_preflights_cross_shard_collision(tmp_path):
    output = _repair_fixture(
        tmp_path,
        {
            "model-00001-of-00002.safetensors": {
                "layer.weight.scales": mx.array([1.0])
            },
            "model-00002-of-00002.safetensors": {"layer.scales": mx.array([2.0])},
        },
        {
            "layer.weight.scales": "model-00001-of-00002.safetensors",
            "layer.scales": "model-00002-of-00002.safetensors",
        },
    )
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    with pytest.raises(RuntimeError, match="output index rename collision"):
        converter.repair_quantized_aux_names(output)

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_quantized_aux_repair_commits_validated_plan(tmp_path):
    shard_name = "model-00001-of-00001.safetensors"
    output = _repair_fixture(
        tmp_path,
        {shard_name: {"layer.weight.scales": mx.array([1.0])}},
        {"layer.weight.scales": shard_name},
    )

    result = converter.repair_quantized_aux_names(output)

    assert result["changed_keys"] == result["changed_shards"] == 1
    tensors = mx.load(str(output / shard_name))
    assert set(tensors) == {"layer.scales"}
    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert index["weight_map"] == {"layer.scales": shard_name}


def test_quantized_aux_repair_rolls_back_commit_failure(tmp_path, monkeypatch):
    output = _repair_fixture(
        tmp_path,
        {"model-00001-of-00001.safetensors": {"layer.weight.scales": mx.array([1.0])}},
        {"layer.weight.scales": "model-00001-of-00001.safetensors"},
    )
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    monkeypatch.setattr(
        converter,
        "_write_sha256sums",
        lambda _output: (_ for _ in ()).throw(OSError("injected checksum failure")),
    )

    with pytest.raises(OSError, match="injected checksum failure"):
        converter.repair_quantized_aux_names(output)

    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_quantization_contract_uses_shape_exact_ple_groups_and_q8_routing():
    model = Model(ModelArgs(model_type="qwen4_exp", text_config=_ple_args().__dict__))
    predicate = model.quant_predicate

    assert predicate(
        "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.3",
        object(),
    ) == {"group_size": 32, "bits": 4}
    assert predicate("language_model.model.layers.1.mlp.gate", object()) == {
        "group_size": 64,
        "bits": 8,
    }
    assert predicate(
        "language_model.model.layers.1.mlp.shared_expert_gate", object()
    ) == {"group_size": 64, "bits": 8}
    assert predicate("language_model.model.layers.1.self_attn.q_proj", object()) is True
