#!/usr/bin/env python3
"""Paired draftless n-gram experiment for Qwen3.8 Flash-Next.

This is deliberately a direct, single-request harness rather than a serving
integration.  It answers whether prompt/history lookup is worth production
engineering for Qwen4's hybrid GDN/QSA cache.  Greedy output must be token
identical to baseline on every run.

On a rejected block the target cache is restored to its pre-verify boundary
and the accepted prefix is replayed.  This is slower than a kernel-provided
partial-state checkpoint, but it is lossless and preserves the all-accepted
fast path we want to measure on copy/edit workloads.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

from vllm_mlx.models.qwen4_exp import Qwen4ExpStateCache
from vllm_mlx.models.qwen4_exp_cache import QSAIndexCache
from vllm_mlx.speculative.suffix_decoding import SuffixDecodingDrafter
from vllm_mlx.utils.tokenizer import _register_vendored_archs


def _fixture_python() -> str:
    rows = [
        "from dataclasses import dataclass",
        "",
        "@dataclass(frozen=True)",
        "class Event:",
        "    sequence: int",
        "    topic: str",
        "    payload: str",
        "",
        "EVENTS = [",
    ]
    for index in range(48):
        rows.append(
            f'    Event(sequence={index}, topic="build.step.{index % 6}", '
            f'payload="artifact-{index:03d}"),'
        )
    rows.extend(
        [
            "]",
            "",
            "def by_topic(events: list[Event]) -> dict[str, list[Event]]:",
            "    result: dict[str, list[Event]] = {}",
            "    for event in events:",
            "        result.setdefault(event.topic, []).append(event)",
            "    return result",
        ]
    )
    return "\n".join(rows)


def _fixture_json() -> str:
    rows = []
    for index in range(36):
        rows.append(
            {
                "id": f"task-{index:03d}",
                "owner": f"worker-{index % 4}",
                "state": ["queued", "running", "passed"][index % 3],
                "attempt": index % 2 + 1,
            }
        )
    return json.dumps(rows, indent=2, separators=(",", ": "))


def _cases() -> dict[str, tuple[str, int]]:
    python_fixture = _fixture_python()
    json_fixture = _fixture_json()
    return {
        "copy_python": (
            "Return exactly the text between BEGIN and END. Do not add fences, "
            "commentary, or the markers.\n"
            f"BEGIN\n{python_fixture}\nEND",
            1800,
        ),
        "copy_json": (
            "Return exactly the JSON between BEGIN and END. Do not add fences, "
            "commentary, or the markers.\n"
            f"BEGIN\n{json_fixture}\nEND",
            2200,
        ),
        "structured_transform": (
            "Using the task records below, return a JSON array containing only "
            "records whose state is passed. Preserve id, owner, state, and attempt "
            "exactly. Return JSON only.\n" + json_fixture,
            700,
        ),
        "novel_code": (
            "Write a Python function schedule_jobs(durations, workers) that assigns "
            "each positive integer duration to the currently least-loaded worker. "
            "Preserve input order for ties, reject workers below one, and do not "
            "mutate durations. Return only valid Python source with no code fence.",
            500,
        ),
    }


@dataclass
class Run:
    case: str
    mode: str
    repetition: int
    prompt_tokens: int
    completion_tokens: int
    decode_seconds: float
    decode_tokens_per_second: float
    output_ids: list[int]
    output_text: str
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    proposal_count: int = 0
    rejection_count: int = 0
    replay_tokens: int = 0


@dataclass
class _QsaSnapshot:
    cache: QSAIndexCache
    raw_ring: mx.array | None
    offsets: list[int]
    compressed_counts: list[int]


@dataclass
class _VerifySnapshot:
    recurrent: list[tuple[Qwen4ExpStateCache, list[mx.array | None]]]
    qsa: list[_QsaSnapshot]
    kv: list[Any]


def _snapshot_verify_boundary(cache: list[Any]) -> _VerifySnapshot:
    """Capture only state that cannot be restored by logical KV trimming."""

    recurrent = []
    qsa = []
    kv = []
    snapshots_to_eval: list[mx.array] = []
    for layer_cache in cache:
        if isinstance(layer_cache, Qwen4ExpStateCache):
            recurrent.append((layer_cache, list(layer_cache.state)))
            continue
        if len(layer_cache.caches) != 2:
            raise AssertionError("unexpected Qwen4 attention cache shape")
        kv_cache, qsa_cache = layer_cache.caches
        if not isinstance(qsa_cache, QSAIndexCache):
            raise AssertionError("unexpected Qwen4 QSA cache type")
        raw_copy = None
        if qsa_cache.raw_ring is not None:
            # QSA writes its small circular raw-key buffer in place.  Force a
            # distinct materialized value before verification mutates it.
            raw_copy = qsa_cache.raw_ring + mx.zeros_like(qsa_cache.raw_ring)
            snapshots_to_eval.append(raw_copy)
        qsa.append(
            _QsaSnapshot(
                qsa_cache,
                raw_copy,
                list(qsa_cache._offsets),
                list(qsa_cache._compressed_counts),
            )
        )
        kv.append(kv_cache)
    if snapshots_to_eval:
        mx.eval(snapshots_to_eval)
    return _VerifySnapshot(recurrent, qsa, kv)


def _restore_before_verify(snapshot: _VerifySnapshot, verify_tokens: int) -> None:
    for cache, state in snapshot.recurrent:
        cache.state = list(state)
        cache.rollback_state = None
    for item in snapshot.qsa:
        item.cache.raw_ring = item.raw_ring
        item.cache._offsets = list(item.offsets)
        item.cache._compressed_counts = list(item.compressed_counts)
    for cache in snapshot.kv:
        trimmed = cache.trim(verify_tokens)
        if trimmed != verify_tokens:
            raise AssertionError(
                f"Qwen4 KV rollback trimmed {trimmed}, expected {verify_tokens}"
            )


def _clear_rollback(cache: list[Any]) -> None:
    for layer_cache in cache:
        if isinstance(layer_cache, Qwen4ExpStateCache):
            layer_cache.rollback_state = None


def _render(tokenizer, prompt: str) -> list[int]:
    return list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def _eos_ids(tokenizer) -> set[int]:
    ids = getattr(tokenizer, "eos_token_ids", None)
    if ids is not None:
        return {int(token) for token in ids}
    return {int(tokenizer.eos_token_id)}


def _prefill(model, prompt_ids: list[int]) -> tuple[list[Any], int]:
    cache = model.make_cache()
    logits = model(mx.array([prompt_ids], dtype=mx.int32), cache=cache)
    token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
    return cache, token


def _run_baseline(
    model, tokenizer, case: str, prompt: str, max_tokens: int, rep: int
) -> Run:
    prompt_ids = _render(tokenizer, prompt)
    cache, next_token = _prefill(model, prompt_ids)
    output = [next_token]
    eos = _eos_ids(tokenizer)
    started = time.perf_counter()
    while len(output) < max_tokens and next_token not in eos:
        logits = model(mx.array([[next_token]], dtype=mx.int32), cache=cache)
        next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        output.append(next_token)
    elapsed = time.perf_counter() - started
    measured = max(0, len(output) - 1)
    return Run(
        case,
        "baseline",
        rep,
        len(prompt_ids),
        len(output),
        elapsed,
        measured / elapsed if elapsed else 0.0,
        output,
        tokenizer.decode(output),
    )


def _run_ngram(
    model,
    tokenizer,
    case: str,
    prompt: str,
    max_tokens: int,
    rep: int,
    *,
    max_draft: int,
    min_suffix: int,
    max_suffix: int,
    min_confidence: float,
) -> Run:
    prompt_ids = _render(tokenizer, prompt)
    cache, next_token = _prefill(model, prompt_ids)
    drafter = SuffixDecodingDrafter(
        max_draft_tokens=max_draft,
        min_suffix_len=min_suffix,
        max_suffix_len=max_suffix,
        min_confidence=min_confidence,
        max_history=None,
    )
    drafter.add_prompt_tokens(prompt_ids)
    drafter.add_generated_token(next_token)
    output = [next_token]
    eos = _eos_ids(tokenizer)
    rejection_count = 0
    replay_tokens = 0
    started = time.perf_counter()
    while len(output) < max_tokens and next_token not in eos:
        remaining = max_tokens - len(output)
        draft = drafter.get_draft()[:remaining]
        if not draft:
            logits = model(mx.array([[next_token]], dtype=mx.int32), cache=cache)
            next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            output.append(next_token)
            drafter.add_generated_token(next_token)
            continue

        verify_ids = [next_token, *draft]
        snapshot = _snapshot_verify_boundary(cache)
        verify_logits = model(mx.array([verify_ids], dtype=mx.int32), cache=cache)
        predictions = mx.argmax(verify_logits[0], axis=-1)
        mx.eval(predictions)
        predicted = [int(token) for token in predictions.tolist()]

        accepted = 0
        stopped = False
        for index, draft_token in enumerate(draft):
            if predicted[index] != draft_token:
                break
            accepted += 1
            output.append(draft_token)
            drafter.add_generated_token(draft_token)
            if draft_token in eos or len(output) >= max_tokens:
                stopped = True
                break
        drafter.record_acceptance(accepted)
        if stopped:
            break

        if accepted < len(draft):
            rejection_count += 1
            _restore_before_verify(snapshot, len(verify_ids))
            kept = [next_token, *draft[:accepted]]
            replay_logits = model(mx.array([kept], dtype=mx.int32), cache=cache)
            mx.eval(replay_logits)
            replay_tokens += len(kept)
        else:
            _clear_rollback(cache)

        next_token = predicted[accepted]
        output.append(next_token)
        drafter.add_generated_token(next_token)

    elapsed = time.perf_counter() - started
    measured = max(0, len(output) - 1)
    stats = drafter.stats_dict()
    return Run(
        case,
        "ngram",
        rep,
        len(prompt_ids),
        len(output),
        elapsed,
        measured / elapsed if elapsed else 0.0,
        output,
        tokenizer.decode(output),
        proposed_tokens=int(stats["total_draft_tokens_proposed"]),
        accepted_tokens=int(stats["total_draft_tokens_accepted"]),
        proposal_count=int(stats["n_drafts_returned"]),
        rejection_count=rejection_count,
        replay_tokens=replay_tokens,
    )


def _load(checkpoint: Path):
    from mlx_lm.utils import load_model, load_tokenizer

    _register_vendored_archs()
    model, _ = load_model(checkpoint, lazy=True, strict=True)
    tokenizer = load_tokenizer(checkpoint)
    mx.eval(model.parameters())
    return model, tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-draft", type=int, default=64)
    parser.add_argument("--min-suffix", type=int, default=24)
    parser.add_argument("--max-suffix", type=int, default=24)
    parser.add_argument("--min-confidence", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument(
        "--cases", nargs="+", choices=list(_cases()), default=list(_cases())
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    model, tokenizer = _load(checkpoint)
    print(
        json.dumps(
            {
                "event": "model_loaded",
                "checkpoint": str(checkpoint),
                "active_gib": round(mx.get_active_memory() / 2**30, 3),
                "peak_gib": round(mx.get_peak_memory() / 2**30, 3),
            }
        ),
        flush=True,
    )

    rows: list[Run] = []
    cases = _cases()
    for repetition in range(1, args.runs + 1):
        for name in args.cases:
            prompt, case_max = cases[name]
            max_tokens = min(case_max, args.max_tokens) if args.max_tokens else case_max
            for mode in ("baseline", "ngram"):
                gc.collect()
                mx.clear_cache()
                if mode == "baseline":
                    row = _run_baseline(
                        model, tokenizer, name, prompt, max_tokens, repetition
                    )
                else:
                    row = _run_ngram(
                        model,
                        tokenizer,
                        name,
                        prompt,
                        max_tokens,
                        repetition,
                        max_draft=args.max_draft,
                        min_suffix=args.min_suffix,
                        max_suffix=args.max_suffix,
                        min_confidence=args.min_confidence,
                    )
                    baseline = rows[-1]
                    if row.output_ids != baseline.output_ids:
                        common = min(len(row.output_ids), len(baseline.output_ids))
                        mismatch = next(
                            (
                                index
                                for index in range(common)
                                if row.output_ids[index] != baseline.output_ids[index]
                            ),
                            common,
                        )
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        args.output.write_text(
                            json.dumps(
                                {
                                    "failure": {
                                        "case": name,
                                        "repetition": repetition,
                                        "mismatch": mismatch,
                                        "baseline_window": baseline.output_ids[
                                            max(0, mismatch - 8) : mismatch + 9
                                        ],
                                        "ngram_window": row.output_ids[
                                            max(0, mismatch - 8) : mismatch + 9
                                        ],
                                    },
                                    "baseline": asdict(baseline),
                                    "ngram": asdict(row),
                                },
                                indent=2,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        raise AssertionError(
                            f"{name} repetition {repetition} token mismatch at {mismatch}"
                        )
                rows.append(row)
                print(
                    json.dumps(
                        {
                            "case": name,
                            "mode": mode,
                            "repetition": repetition,
                            "completion_tokens": row.completion_tokens,
                            "decode_tps": round(row.decode_tokens_per_second, 3),
                            "proposed": row.proposed_tokens,
                            "accepted": row.accepted_tokens,
                            "replays": row.replay_tokens,
                        }
                    ),
                    flush=True,
                )

    summary = {}
    for name in args.cases:
        baseline = [
            row.decode_tokens_per_second
            for row in rows
            if row.case == name and row.mode == "baseline"
        ]
        ngram = [
            row.decode_tokens_per_second
            for row in rows
            if row.case == name and row.mode == "ngram"
        ]
        baseline_median = statistics.median(baseline)
        ngram_median = statistics.median(ngram)
        proposed = sum(
            row.proposed_tokens
            for row in rows
            if row.case == name and row.mode == "ngram"
        )
        accepted = sum(
            row.accepted_tokens
            for row in rows
            if row.case == name and row.mode == "ngram"
        )
        summary[name] = {
            "baseline_median_tps": baseline_median,
            "ngram_median_tps": ngram_median,
            "speedup": ngram_median / baseline_median,
            "accepted_over_proposed": accepted / proposed if proposed else None,
            "all_outputs_token_exact": True,
        }

    payload = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "config": {
            "runs": args.runs,
            "max_draft": args.max_draft,
            "min_suffix": args.min_suffix,
            "max_suffix": args.max_suffix,
            "min_confidence": args.min_confidence,
            "temperature": 0,
            "thinking": False,
        },
        "mlx_memory": {
            "active_gib": mx.get_active_memory() / 2**30,
            "peak_gib": mx.get_peak_memory() / 2**30,
        },
        "summary": summary,
        "runs": [asdict(row) for row in rows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"summary": summary}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
