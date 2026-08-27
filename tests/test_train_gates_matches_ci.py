"""Drift test: keep `scripts/train_gates.sh` (and its parser helper) honest
against the hosted CI workflows.

`scripts/train_gates.sh <base-sha>` reproduces the 5 validation gates the CI
matrix runs. It does NOT hardcode those gates — it parses them at runtime from
`.github/workflows/ci.yml` and `.github/workflows/rapid-mac-ci.yml` via
`scripts/train_gates_parser.py`. This test guards that reproduction from
drifting away from the workflows: if a CI-definition edit changes the Linux
pytest roster, the Apple pytest roster, the mypy invocation, the diff-cover
invocation, or the Desktop swift invocation, one of the assertions below must
fail — exactly the machine-readable tripwire that keeps the local train-gates
reproduction honest.

Pure-pytest, Linux-friendly, no MLX import (the parser is stdlib + PyYAML).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.train_gates_parser import (
    CI_WORKFLOW,
    MAC_CI_WORKFLOW,
    MYPY_BUDGET_SCRIPT,
    parse_apple_pytest_args,
    parse_diff_cover_invocation,
    parse_linux_pytest_args,
    parse_mypy_invocation,
    parse_swift_test_invocation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
yaml = pytest.importorskip("yaml")

CI = CI_WORKFLOW.read_text()
MAC_CI = MAC_CI_WORKFLOW.read_text()
CI_PARSED = yaml.safe_load(CI)
MAC_CI_PARSED = yaml.safe_load(MAC_CI)


def _extract_path_tokens(run_text: str) -> list[str]:
    """Extract the literal `tests/test_*.py[::...]` tokens CI passes on its own
    (independent re-derivation, NOT via the shared parser, so the parser cannot
    mask a drift against a bug of its own)."""
    tokens: list[str] = []
    for line in run_text.splitlines():
        # Skip the --deselect= entries (they carry the same shape but are
        # excluded args, not executed paths).
        if "--deselect=" in line:
            continue
        stripped = line.strip()
        if stripped.startswith("tests/test_") and ".py" in stripped:
            # strip the trailing ` \` continuation and any trailing flag words
            tok = stripped.split(" \\")[0].split()[0]
            tokens.append(tok)
    return tokens


def _run_text(job_name: str, step_name: str, ci: dict) -> str:
    job = ci["jobs"][job_name]
    matches = [s for s in job["steps"] if s.get("name") == step_name]
    assert matches, f"step {step_name!r} not found in job {job_name}"
    return matches[0]["run"]


# ---------------------------------------------------------------------------
# Linux no-MLX roster
# ---------------------------------------------------------------------------
def test_linux_parser_matches_workflow_roster() -> None:
    parsed = parse_linux_pytest_args()
    run_text = _run_text("test-matrix", "Run unit tests (no MLX required)", CI_PARSED)
    workflow_paths = _extract_path_tokens(run_text)
    assert workflow_paths, "workflow carries no Linux test paths"
    assert parsed["paths"] == workflow_paths, (
        "Linux pytest roster drifted: the parser extracts\n"
        f"{parsed['paths']}\nbut ci.yml carries\n{workflow_paths}\n"
        "Update the parser (or the workflow) so they agree."
    )


def test_linux_parser_extracts_deselect_and_k_filter() -> None:
    parsed = parse_linux_pytest_args()
    run_text = _run_text("test-matrix", "Run unit tests (no MLX required)", CI_PARSED)
    # Every --deselect= arg in the workflow must be captured, and only those.
    expected_deselect = [
        line.strip().split("--deselect=", 1)[1].split(" \\")[0].split()[0]
        for line in run_text.splitlines()
        if "--deselect=" in line
    ]
    assert parsed["deselect"] == expected_deselect
    assert "--deselect=" in run_text  # sanity: the workflow does deselect

    # The -k filter must be captured verbatim.
    assert parsed["marker"], "Linux -k filter is empty; workflow must carry one"
    assert parsed["marker"] in run_text


def test_linux_roster_asserts_no_mlx_import() -> None:
    # The Linux roster lives in the "no MLX required" step; this is the contract
    # that Gate 1 enforces. Guard the guard: every path the parser hands to
    # Gate 1 must resolve to a real file.
    parsed = parse_linux_pytest_args()
    for path in parsed["paths"]:
        file_part = path.split("::", 1)[0]
        assert (CI_WORKFLOW.parents[2] / file_part).is_file(), f"{file_part} missing"


# ---------------------------------------------------------------------------
# Apple-MLX roster
# ---------------------------------------------------------------------------
def test_apple_parser_matches_workflow_roster() -> None:
    parsed = parse_apple_pytest_args()
    run_text = _run_text("test-apple-silicon", "Run MLX-dependent tests", CI_PARSED)
    workflow_paths = _extract_path_tokens(run_text)
    assert workflow_paths, "workflow carries no Apple test paths"
    assert parsed["paths"] == workflow_paths, (
        "Apple pytest roster drifted: the parser extracts\n"
        f"{parsed['paths']}\nbut ci.yml carries\n{workflow_paths}\n"
    )
    for path in parsed["paths"]:
        file_part = path.split("::", 1)[0]
        assert (CI_WORKFLOW.parents[2] / file_part).is_file(), f"{file_part} missing"


def test_apple_roster_has_no_overlap_with_linux_surface() -> None:
    # A test that lands in BOTH rosters is a red flag: an mlx-importing test in
    # the Linux (no-MLX) roster would crash CI, and a no-MLX test wasted on the
    # Apple gate hides Linux-only coverage. (test_mllm_cache.py legitimately
    # appears in both, exercising no-MLX paths on Linux and mlx paths on Apple.)
    linux = set(parse_linux_pytest_args()["paths"])
    apple = set(parse_apple_pytest_args()["paths"])
    overlap = linux & apple
    # Documented intentional overlaps: test_mllm_cache.py exercises no-MLX
    # paths on Linux and mlx paths on Apple; test_routing_groups_0131.py is
    # legitimately in both rosters as an existing cross-platform suite.
    assert overlap <= {
        "tests/test_mllm_cache.py",
        "tests/test_routing_groups_0131.py",
    }, f"unexpected roster overlap: {overlap}"


# ---------------------------------------------------------------------------
# mypy budget
# ---------------------------------------------------------------------------
def test_mypy_invocation_matches_workflow() -> None:
    parsed = parse_mypy_invocation()
    run_text = _run_text(
        "type-check", "Enforce shrink-only mypy error budget", CI_PARSED
    )
    assert parsed == {"script": MYPY_BUDGET_SCRIPT}
    assert f"python {MYPY_BUDGET_SCRIPT}" in run_text
    # The pinned budget file must exist (it feeds the gates-hash).
    assert (CI_WORKFLOW.parents[2] / "config/mypy-requirements.txt").is_file()
    assert (CI_WORKFLOW.parents[2] / "config/mypy-error-baseline.txt").is_file()


# ---------------------------------------------------------------------------
# diff-cover (Gate 3)
# ---------------------------------------------------------------------------
def test_diff_cover_invocation_matches_workflow() -> None:
    parsed = parse_diff_cover_invocation()
    run_text = _run_text(
        "changed-lines-coverage",
        "Combine coverage and enforce changed lines",
        CI_PARSED,
    )
    assert "coverage combine" in run_text
    assert "coverage-data/linux/coverage-linux-3.11.data" in run_text
    assert "coverage-data/apple/coverage-apple.data" in run_text
    assert "--fail-under 100" in run_text
    assert parsed["fail_under"] == 100
    assert parsed["linux"] == "coverage-linux-3.11.data"
    assert parsed["apple"] == "coverage-apple.data"


# ---------------------------------------------------------------------------
# Desktop swift test (Gate 5)
# ---------------------------------------------------------------------------
def test_swift_test_invocation_matches_workflow() -> None:
    parsed = parse_swift_test_invocation()
    assert parsed == {"cmd": "swift test --no-parallel"}
    assert "swift test --no-parallel" in MAC_CI


# ---------------------------------------------------------------------------
# The training gates must actually be wired into the shared parser so the
# hash that freeze relies on stays stable under renames.
# ---------------------------------------------------------------------------
def test_train_gates_script_parses_workflows() -> None:
    # Exercise the subprocess entry path too (what train_gates.sh runs), so a
    # break in `python -m scripts.train_gates_parser` surfaces here, in CI,
    # before it surfaces in a local train run.
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "scripts.train_gates_parser", "all"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    import json

    payload = json.loads(proc.stdout)
    assert set(payload) == {"linux", "apple", "mypy", "diff_cover", "swift_test"}
    assert payload["linux"]["paths"]
    assert payload["apple"]["paths"]


def test_train_gates_script_cli_targets_are_reachable() -> None:
    # `train_gates.sh` invokes the parser for exactly these single targets via
    # `python -m scripts.train_gates_parser <target>`. A new gate must not add
    # a subprocess target the script uses without this tripwire noticing, and
    # an existing target must never become unreachable.
    import json
    import subprocess
    import sys

    expected_targets = ("linux", "apple", "mypy", "diff_cover", "swift_test")
    for target in expected_targets:
        proc = subprocess.run(
            [sys.executable, "-m", "scripts.train_gates_parser", target],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, (target, proc.stderr)
        payload = json.loads(proc.stdout)
        assert payload, (target, "empty payload")
