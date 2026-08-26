# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``scripts/microbench_parsers.py``.

The microbench itself does timing — we don't reliably-test timing here
(unit tests run on shared hardware too). What we DO test is the gate
logic: threshold compare, sample wiring, exit codes, --report mode.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "microbench_parsers.py"


def _load_module():
    # Register in sys.modules BEFORE exec_module — the script's
    # dataclass declaration calls sys.modules.get(cls.__module__),
    # which returns None for a module that hasn't been registered,
    # and dataclasses then crashes on .__dict__ access.
    import sys

    spec = importlib.util.spec_from_file_location("microbench_parsers", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["microbench_parsers"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mb():
    return _load_module()


# ---------- threshold compare logic ----------------------------------


def test_bench_under_threshold_passes(mb):
    """``bench_one`` with a fast no-op callable should pass."""
    result = mb.bench_one("hermes", lambda _t: None, "irrelevant", iters=100)
    assert result.passed
    assert result.iters == 100
    assert result.us_per_call < result.threshold_us


def test_bench_over_threshold_fails(mb):
    """``bench_one`` with an artificially slow callable should fail."""
    import time

    def slow(_t):
        # Sleep ~1ms = 1000 μs, well over any parser threshold.
        time.sleep(0.001)

    result = mb.bench_one("hermes", slow, "irrelevant", iters=5)
    assert not result.passed
    assert result.us_per_call > result.threshold_us


def test_unknown_parser_gets_default_threshold(mb):
    """Adding a new parser without a threshold entry should still run
    (with a generous default), not crash with KeyError."""
    result = mb.bench_one("brand_new", lambda _t: None, "x", iters=10)
    assert result.threshold_us > 0  # has a default
    assert result.passed


# ---------- sample / parser wiring -----------------------------------


def test_each_base_us_has_a_sample(mb):
    """Every parser in BASE_US must have a SAMPLES entry.
    Otherwise the bench silently skips it without complaining, which
    would let a regression slip through unbenched."""
    missing = sorted(set(mb.BASE_US) - set(mb.SAMPLES))
    assert not missing, (
        f"parsers in BASE_US but missing in SAMPLES: {missing}. "
        "Add a realistic sample input to SAMPLES so it actually benches."
    )


def test_each_sample_has_a_base_us(mb):
    """And vice versa — every SAMPLES entry should have a base cost so
    the gate is enforced, not just a printed timing."""
    missing = sorted(set(mb.SAMPLES) - set(mb.BASE_US))
    assert not missing, (
        f"parsers in SAMPLES but missing in BASE_US: {missing}. "
        "Either add a base cost or remove from SAMPLES."
    )


def test_base_us_are_positive(mb):
    """Catch the 'paste-bug' where someone sets a base cost to 0 or
    negative — that would make every measurement fail."""
    for name, val in mb.BASE_US.items():
        assert val > 0, f"{name}: base cost must be positive, got {val}"


def test_regression_limit_sane(mb):
    """The relative gate must be a positive multiple > 1 so it actually
    catches an order-of-magnitude regression rather than being inverted
    or a no-op."""
    assert mb.REGRESSION_LIMIT > 1.0


# The two runner-speed tests use a CPU-bound synthetic parser whose work
# scales linearly with its iteration count. Because the work is CPU-bound on
# the same hardware, a parser asked to do N ops takes ~2× the wall-time of
# N/2 ops — and, critically, the SAME N on a 5×-slower runner takes ~5× the
# wall-time. We pick N so the parser costs ``BASE_US * eps * speedup`` μs, so
# its measured μs/call grows with ``runner_speedup`` exactly like the
# calibration baseline does. The verdict then reduces to ``eps <= LIMIT``
# independent of runner speed — which is the property that kills the #2344
# shared-runner flake.
_CAL_TEXT = (
    "<tool_call>get_weather city San Francisco "
    "<arg_key>city</arg_key><arg_value>San Francisco</arg_value></tool_call>"
)


def _cpu_iter_cost_us() -> float:
    """Measure the per-iteration μs of the CPU-bound op on THIS machine."""
    import time

    n = 30_000
    t0 = time.perf_counter()
    for _ in range(n):
        _CAL_TEXT.count("x")
    dt = time.perf_counter() - t0
    return (dt / n) * 1_000_000


def _prop_parser():
    """A synthetic parser whose per-call wall-time ∝ its iteration count."""

    def _make(us_per_call: float):
        k = _cpu_iter_cost_us()
        n = max(1, int(us_per_call / k))

        def fn(_t):
            c = 0
            for _ in range(n):
                c += _CAL_TEXT.count("x")
            return c

        return fn

    return _make


def _bench_with_cal(mb, make, eps_mult, speedup, monkeypatch):
    """Exercise ``bench_one`` through the REAL calibration path by mocking the
    runner-speed measurement (we cannot make the shared test machine 5× slower).

    The parser workload is CPU-bound and sized to cost ``BASE_US*ε*speedup`` μs
    (each μs of work is one calibration-op's worth), so it reproduces what a
    real parser experiences on a ``speedup``× slower runner, and the mocked
    calibration returns exactly that ``speedup``. The verdict must then depend
    only on ``ε`` vs ``REGRESSION_LIMIT`` — proving runner-speed independence
    (#2344).
    """
    base_us = mb.BASE_US["hermes"]
    limit = mb.REGRESSION_LIMIT
    target = base_us * limit * eps_mult * speedup
    monkeypatch.setattr(mb, "_measure_runner_speedup", lambda: float(speedup))
    return mb.bench_one("hermes", make(target), "x", iters=_CAL_MIN_ITERS_FOR_TEST)


# Shared by the two speed tests: enough iters to trigger multi-round
# interleaving but still fast.
_CAL_MIN_ITERS_FOR_TEST = 50


def test_slow_runner_scales_threshold_up(mb, monkeypatch):
    """A slower runner produces a proportionally higher effective threshold,
    and a speed-proportional parser keeps the SAME verdict on both (the
    relative gate does not flake on shared-runner speed, #2344)."""
    make = _prop_parser()
    # A healthy parser, well under REGRESSION_LIMIT (eps=0.3×LIMIT), so
    # timing noise can't tip it over the boundary.
    on_m3 = _bench_with_cal(mb, make, 0.3, 1.0, monkeypatch)
    on_slow = _bench_with_cal(mb, make, 0.3, 5.0, monkeypatch)
    assert on_slow.threshold_us > on_m3.threshold_us
    assert on_m3.passed and on_slow.passed


def test_relative_budget_catches_regression_across_runner_speeds(mb, monkeypatch):
    """The gate is a ratio: a parser at REGRESSION_LIMIT × its base fails at
    EVERY runner speed (fast M3 and 5×-slower shared runner alike), and one
    just under the limit passes at every speed (#2344)."""
    make = _prop_parser()
    # Under the limit (ε=0.8×LIMIT): passes on 1x and 5x runners.
    assert _bench_with_cal(mb, make, 0.8, 1.0, monkeypatch).passed
    assert _bench_with_cal(mb, make, 0.8, 5.0, monkeypatch).passed
    # Over the limit (ε=1.2×LIMIT): fails on 1x and 5x runners alike.
    assert not _bench_with_cal(mb, make, 1.2, 1.0, monkeypatch).passed
    assert not _bench_with_cal(mb, make, 1.2, 5.0, monkeypatch).passed


def test_speedup_override_passthrough(mb, monkeypatch):
    """The explicit ``runner_speedup`` override path still works for tests that
    hand a scalar (backward-compat with pre-interleave unit semantics)."""
    make = _prop_parser()
    base_us = mb.BASE_US["hermes"]
    r = mb.bench_one("hermes", make(base_us * 5.0), "x", iters=20, runner_speedup=1.0)
    assert r.passed
    assert r.threshold_us == pytest.approx(mb.BASE_US["hermes"] * mb.REGRESSION_LIMIT)


def test_calibration_returns_positive_finite(mb):
    """``_measure_runner_speedup`` must return a positive, finite scalar."""
    import math

    speedup = mb._measure_runner_speedup()
    assert math.isfinite(speedup)
    assert speedup >= mb._RUNNER_SPEEDUP_FLOOR


# ---------- entry point ----------------------------------------------


def test_main_with_no_args_runs_and_exits_cleanly(mb):
    """End-to-end smoke: load real parsers, run a tiny iter count."""
    # Small iter count so the test is fast; threshold gates are still
    # generous enough to handle CI variance at this iter count.
    rc = mb.main(["--iters", "100"])
    assert rc == 0


def test_report_mode_returns_zero_even_with_failures(mb):
    """``--report`` should suppress the non-zero exit so it can be used
    as an info-only step on PR-validation runs."""
    # Run with --iters 1 just to ensure execution finishes fast; even
    # if perf is degenerate, --report should still exit 0.
    rc = mb.main(["--iters", "1", "--report"])
    assert rc == 0
