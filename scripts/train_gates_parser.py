#!/usr/bin/env python3
"""Single source of truth: parse the exact gate definitions out of the hosted
CI workflows so that ``scripts/train_gates.sh`` reproduces, locally, the same
gates the GitHub Actions matrix runs.

This module is imported by the drift test
``tests/test_train_gates_matches_ci.py`` and executed as a subprocess by
``scripts/train_gates.sh`` (via ``python -m scripts.train_gates_parser``). It
must parse the workflows the same way in both cases; if the workflow layout
changes such that the parser can no longer find a gate, the drift test fails.

Gate surface parsed here (see ``scripts/train_gates.sh`` for the full gate
list):
  * Linux no-MLX pytest roster + --deselect + -k filter (ci.yml test-matrix,
    "Run unit tests (no MLX required)" step)
  * Apple-MLX pytest roster (ci.yml test-apple-silicon, "Run MLX-dependent
    tests" step)
  * mypy budget invocation (ci.yml type-check job)
  * diff-cover invocation (ci.yml changed-lines-coverage job)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - yaml is a dev-only dependency
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
MAC_CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "rapid-mac-ci.yml"
MYPY_BUDGET_SCRIPT = "scripts/check_mypy_error_budget.py"

# A single test path token: `tests/test_*.py` with any number of `::X` segments
# (e.g. ``tests/test_responses_route.py::TestResponsesNonStream::test_...``).
_TEST_PATH = re.compile(r"tests/[A-Za-z0-9_]+\.py(::[A-Za-z0-9_]+)*")

_DESELECT = re.compile(r"--deselect=([^ \t\\]+)")

_K_FILTER = re.compile(r'-k\s+"([^"]+)"')


def _load_workflow() -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to parse the CI workflows")
    return yaml.safe_load(CI_WORKFLOW.read_text())


def parse_linux_pytest_args() -> dict[str, Any]:
    """Parse the Linux no-MLX pytest block from ci.yml.

    Returns ``{paths, deselect, marker}``:
      * ``paths``  - ordered list of ``tests/test_*.py[::...]`` tokens
      * ``deselect`` - list of ``tests/...`` paths passed via ``--deselect=``
      * ``marker`` - the ``-k "..."`` filter string (or None)
    """
    workflow = _load_workflow()
    job = workflow["jobs"]["test-matrix"]
    step = _find_step_by_name(job, "Run unit tests (no MLX required)")
    run_text = step["run"]

    paths: list[str] = []
    deselect: list[str] = []
    marker: str | None = None

    for line in run_text.splitlines():
        path_match = _TEST_PATH.search(line)
        if path_match and "--deselect=" not in line:
            paths.append(path_match.group(0))

        deselect_match = _DESELECT.search(line)
        if deselect_match:
            deselect.append(deselect_match.group(1))

        k_match = _K_FILTER.search(line)
        if k_match:
            marker = k_match.group(1)

    if not paths:
        raise ValueError(
            "could not find the Linux no-MLX pytest roster in ci.yml; "
            "the test-matrix 'Run unit tests' step layout may have changed"
        )
    return {"paths": paths, "deselect": deselect, "marker": marker}


def parse_apple_pytest_args() -> dict[str, Any]:
    """Parse the Apple-MLX pytest block from ci.yml."""
    workflow = _load_workflow()
    job = workflow["jobs"]["test-apple-silicon"]
    step = _find_step_by_name(job, "Run MLX-dependent tests")
    run_text = step["run"]

    paths: list[str] = []
    for line in run_text.splitlines():
        path_match = _TEST_PATH.search(line)
        if path_match and "--deselect=" not in line:
            paths.append(path_match.group(0))

    if not paths:
        raise ValueError(
            "could not find the Apple-MLX pytest roster in ci.yml; "
            "the test-apple-silicon 'Run MLX-dependent tests' step may have "
            "changed"
        )
    return {"paths": paths}


def parse_mypy_invocation() -> dict[str, Any]:
    """Parse the mypy budget invocation from ci.yml (type-check job)."""
    workflow = _load_workflow()
    job = workflow["jobs"]["type-check"]
    step = _find_step_by_name(job, "Enforce shrink-only mypy error budget")
    script = step["run"].strip()
    if script != f"python {MYPY_BUDGET_SCRIPT}":
        raise ValueError(
            f"mypy budget invocation drifted; expected 'python "
            f"{MYPY_BUDGET_SCRIPT}', found {script!r}"
        )
    return {"script": MYPY_BUDGET_SCRIPT}


def parse_diff_cover_invocation() -> dict[str, Any]:
    """Parse the diff-cover invocation from ci.yml (changed-lines job)."""
    workflow = _load_workflow()
    job = workflow["jobs"]["changed-lines-coverage"]
    step = _find_step_by_name(job, "Combine coverage and enforce changed lines")
    run_text = step["run"]

    linux = "coverage-data/linux/coverage-linux-3.11.data"
    apple = "coverage-data/apple/coverage-apple.data"
    if linux not in run_text or apple not in run_text:
        raise _drift(
            f"diff-cover combine inputs drifted; expected {linux!r} and {apple!r}"
        )
    if "--compare-branch" not in run_text:
        raise _drift("diff-cover --compare-branch missing")
    if "--fail-under 100" not in run_text:
        raise _drift("diff-cover --fail-under 100 missing")
    return {
        "linux": "coverage-linux-3.11.data",
        "apple": "coverage-apple.data",
        "fail_under": 100,
    }


def parse_swift_test_invocation() -> dict[str, Any]:
    """Parse the Desktop ``swift test`` invocation from rapid-mac-ci.yml."""
    parsed = yaml.safe_load(MAC_CI_WORKFLOW.read_text())
    for job in parsed["jobs"].values():
        for step in job.get("steps", []):
            if isinstance(step, dict) and step.get("name") == "swift test":
                run = step["run"].strip()
                if run != "swift test --no-parallel":
                    raise _drift(
                        f"Desktop swift test drifted: expected "
                        f"'swift test --no-parallel', found {run!r}"
                    )
                return {"cmd": run}
    raise _drift("could not find the Desktop 'swift test' step")


def _find_step_by_name(job: dict[str, Any], name: str) -> dict[str, Any]:
    for step in job["steps"]:
        if isinstance(step, dict) and step.get("name") == name:
            return step
    raise _drift(f"step {name!r} not found in job")


def _drift(message: str) -> RuntimeError:
    return RuntimeError(f"[train-gates drift] {message}")


def main() -> int:  # pragma: no cover - exercised via train_gates.sh
    """CLI entry: ``python -m scripts.train_gates_parser PARSE_NAME``."""
    if len(sys.argv) != 2:
        print(
            "usage: python -m scripts.train_gates_parser "
            "<linux|apple|mypy|diff_cover|swift_test|all>",
            file=sys.stderr,
        )
        return 2
    what = sys.argv[1]
    try:
        if what == "linux":
            result = parse_linux_pytest_args()
        elif what == "apple":
            result = parse_apple_pytest_args()
        elif what == "mypy":
            result = parse_mypy_invocation()
        elif what == "diff_cover":
            result = parse_diff_cover_invocation()
        elif what == "swift_test":
            result = parse_swift_test_invocation()
        elif what == "all":
            result = {
                "linux": parse_linux_pytest_args(),
                "apple": parse_apple_pytest_args(),
                "mypy": parse_mypy_invocation(),
                "diff_cover": parse_diff_cover_invocation(),
                "swift_test": parse_swift_test_invocation(),
            }
        else:
            print(f"unknown parse target {what!r}", file=sys.stderr)
            return 2
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - surface drift as non-zero
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
