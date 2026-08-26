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


def test_slow_runner_scales_threshold_up(mb):
    """A slower runner must produce a proportionally higher effective
    threshold, so an unchanged parser doesn't flake (issue #2344)."""
    base = mb.bench_one("hermes", lambda _t: None, "x", iters=10, runner_speedup=1.0)
    slow = mb.bench_one("hermes", lambda _t: None, "x", iters=10, runner_speedup=5.0)
    # 5x slower runner -> threshold scales in kind; the same fast callable
    # must still pass on BOTH (relative budget, not absolute wall-clock).
    assert slow.threshold_us > base.threshold_us
    assert slow.passed and base.passed


def test_relative_budget_catches_regression_across_runner_speeds(mb):
    """The gate is a ratio (parser / calibrated baseline), so the SAME
    relative slowdown fails regardless of how slow the runner is. This is
    what makes the gate runner-speed-independent (#2344)."""

    def slow(_t):
        # ~1ms per call, i.e. far more than REGRESSION_LIMIT × any base.
        import time

        time.sleep(0.001)

    on_m3 = mb.bench_one("hermes", slow, "x", iters=3, runner_speedup=1.0)
    on_slow_runner = mb.bench_one("hermes", slow, "x", iters=3, runner_speedup=5.0)
    assert not on_m3.passed
    # On a 5x slower runner the effective threshold scales 5x too, so the
    # ~1ms absolute slowdown is still (first-order) a hard FAIL on both.
    assert not on_slow_runner.passed


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
