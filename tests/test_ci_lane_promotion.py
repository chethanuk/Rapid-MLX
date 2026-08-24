# SPDX-License-Identifier: Apache-2.0
"""Contracts for lane-scoped full-CI promotion.

These tests intentionally inspect the workflows: a future cleanup must not
restore the expensive behavior where applying ``full-ci`` changed an
engine-only or Desktop-only PR into an all-product run.
"""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ENGINE_WORKFLOW = ROOT / ".github/workflows/ci.yml"
DESKTOP_WORKFLOW = ROOT / ".github/workflows/rapid-mac-ci.yml"
COVERAGE_CONFIG = ROOT / ".coveragerc"


def _step_run(workflow: Path, job: str, step_name: str) -> str:
    steps = yaml.safe_load(workflow.read_text())["jobs"][job]["steps"]
    (step,) = [candidate for candidate in steps if candidate.get("name") == step_name]
    return str(step["run"])


def _job(workflow: Path, job: str) -> dict[str, object]:
    return yaml.safe_load(workflow.read_text())["jobs"][job]


def test_engine_full_ci_still_classifies_the_pr_diff():
    run = _step_run(ENGINE_WORKFLOW, "changes", "Classify validation lanes")
    assert 'git diff --no-renames --name-only "$PR_BASE_SHA" "$GITHUB_SHA"' in run
    assert 'full_gate="$FULL_CI"' in run
    assert 'if [ "$FULL_CI" = true ]' not in run


def test_desktop_full_ci_still_classifies_the_pr_diff():
    run = _step_run(DESKTOP_WORKFLOW, "changes", "Classify desktop lane")
    assert 'git diff --no-renames --name-only "$PR_BASE_SHA" "$GITHUB_SHA"' in run
    assert 'echo "full_gate=$FULL_CI"' in run
    assert '|| [ "$FULL_CI" = true ]' not in run


def test_non_engine_change_exits_before_full_ci_requirement():
    run = _step_run(ENGINE_WORKFLOW, "tests", "Check test results")
    classifier_gate = run.index("needs.changes.result")
    common_gate = run.index("needs.lint.result")
    no_lane = run.index('if [ "$expected" != "true" ]')
    engine_gate = run.index("needs.engine-contracts.result")
    promotion = run.index("needs.changes.outputs.full_gate")
    assert classifier_gate < common_gate < no_lane < engine_gate < promotion


def test_non_desktop_change_exits_before_full_ci_requirement():
    run = _step_run(DESKTOP_WORKFLOW, "desktop-tests", "Check desktop results")
    classifier_gate = run.index("needs.changes.result")
    no_lane = run.index('if [ "$DESKTOP_EXPECTED" != true ]')
    promotion = run.index('if [ "${{ github.event_name }}" = pull_request ]')
    assert classifier_gate < no_lane < promotion


def test_gui_golden_job_requires_both_desktop_lane_and_full_promotion():
    condition = str(_job(DESKTOP_WORKFLOW, "gui-golden-flows")["if"])
    assert "needs.changes.outputs.desktop == 'true'" in condition
    assert "needs.changes.outputs.full_gate == 'true'" in condition


def test_engine_only_contracts_are_not_universal_pr_guards():
    universal_steps = {
        step.get("name") for step in _job(ENGINE_WORKFLOW, "lint")["steps"]
    }
    engine_steps = {
        step.get("name") for step in _job(ENGINE_WORKFLOW, "engine-contracts")["steps"]
    }
    assert {
        "GitHub Actions SHA pinning",
        "Workflow expression sanity",
        "Model-management architecture SSOT",
        "Run ruff lint",
        "Run ruff format check",
        "Engine ↔ desktop app version sync",
    } <= universal_steps
    assert {
        "CLI ↔ Config fidelity audit",
        "Release-script offline tests",
        "Installer offline tests",
        "Parser microbench",
    } <= engine_steps
    assert not universal_steps & {
        "CLI ↔ Config fidelity audit",
        "Release-script offline tests",
        "Installer offline tests",
        "Parser microbench",
    }


def test_engine_jobs_follow_fail_closed_engine_classification():
    for job_name in ("engine-contracts", "type-check"):
        job = _job(ENGINE_WORKFLOW, job_name)
        assert job["needs"] == "changes"
        assert str(job["if"]) == "needs.changes.outputs.engine == 'true'"

    bound_guard = _job(ENGINE_WORKFLOW, "mlx-bound-guard")
    assert bound_guard["needs"] == "changes"
    condition = str(bound_guard["if"])
    assert "github.event_name == 'pull_request'" in condition
    assert "needs.changes.outputs.engine == 'true'" in condition


def test_type_check_enforces_shrink_only_error_budget():
    type_check = _job(ENGINE_WORKFLOW, "type-check")
    steps = type_check["steps"]
    ratchet = next(
        step
        for step in steps
        if step.get("name") == "Enforce shrink-only mypy error budget"
    )

    assert "continue-on-error" not in ratchet
    assert ratchet["run"] == "python scripts/check_mypy_error_budget.py"
    install = next(step for step in steps if step.get("name") == "Install dependencies")
    assert "pip install --requirement config/mypy-requirements.txt" in install["run"]
    requirements = (ROOT / "config/mypy-requirements.txt").read_text().splitlines()
    pins = [line for line in requirements if line and not line.startswith("#")]
    assert pins
    assert all("==" in pin for pin in pins)
    assert {pin.split("==", maxsplit=1)[0] for pin in pins} >= {
        "mypy",
        "pydantic",
        "pydantic_core",
        "fastapi",
        "starlette",
        "typing_extensions",
    }
    unit_roster = _step_run(
        ENGINE_WORKFLOW, "test-matrix", "Run unit tests (no MLX required)"
    )
    assert "tests/test_check_mypy_error_budget.py" in unit_roster


def test_changed_lines_coverage_combines_linux_and_apple_evidence():
    assert "relative_files = true" in COVERAGE_CONFIG.read_text()
    test_matrix = _job(ENGINE_WORKFLOW, "test-matrix")
    linux_upload = next(
        step
        for step in test_matrix["steps"]
        if step.get("name") == "Upload Linux coverage evidence"
    )
    assert linux_upload["if"] == "matrix.python-version == '3.11'"
    assert linux_upload["with"]["retention-days"] == 1

    apple = _job(ENGINE_WORKFLOW, "test-apple-silicon")
    apple_run = next(
        step for step in apple["steps"] if step.get("name") == "Run MLX-dependent tests"
    )
    assert "--cov=vllm_mlx" in apple_run["run"]
    assert "tests/test_audio*.py" in apple_run["run"]

    coverage = _job(ENGINE_WORKFLOW, "changed-lines-coverage")
    assert set(coverage["needs"]) == {
        "changes",
        "test-matrix",
        "test-apple-silicon",
    }
    gate = next(
        step
        for step in coverage["steps"]
        if step.get("name") == "Enforce changed-lines coverage"
    )
    assert gate["env"] == {"PR_BASE_SHA": "${{ github.event.pull_request.base.sha }}"}
    assert "continue-on-error" not in gate
    assert "coverage combine coverage-linux coverage-apple" in gate["run"]
    assert "coverage.xml" in gate["run"]
    assert '--compare-branch "$PR_BASE_SHA"' in gate["run"]
    assert "--show-uncovered" in gate["run"]
    assert "--fail-under 100" in gate["run"]
    assert "--cov-fail-under" not in gate["run"]

    aggregate = _step_run(ENGINE_WORKFLOW, "tests", "Check test results")
    coverage_gate = aggregate.index("needs.changed-lines-coverage.result")
    assert (
        aggregate.rfind('github.event_name }}" = "pull_request"', 0, coverage_gate) >= 0
    )
