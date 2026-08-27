#!/usr/bin/env bash
#
# train_gates.sh <base-sha>
#
# Reproduce, LOCALLY on one machine, the same 5 validation gates the hosted CI
# matrix runs for an engine train, so a train can be "frozen" without waiting
# for (or burning) hosted runners. On full success it prints exactly:
#
#     GATES OK <sha> <gates-hash>
#
# and exits 0. If any non-skippable gate fails it prints a per-gate FAILURE
# block and exits non-zero.
#
# The gate definitions are NOT hardcoded here: they are parsed at runtime from
# .github/workflows/ci.yml and .github/workflows/rapid-mac-ci.yml by
# scripts/train_gates_parser.py (the single source of truth). The drift test
# tests/test_train_gates_matches_ci.py guards that this parser stays in sync
# with the workflows.
#
# Environment (all optional):
#   TRAIN_GATES_PYTHON           control interpreter: must be able to import
#                                yaml, coverage, diff_cover, pytest. Defaults
#                                to python3, then falls back to a repo .venv.
#   TRAIN_GATES_APPLE_VENV       path to an existing Apple-Silicon venv that
#                                already has the package installed (with mlx),
#                                to reuse for Gate 4 instead of reinstalling.
#   TRAIN_GATES_ALLOW_APPLE_INSTALL=1
#                                if no apple venv is provided, create a fresh
#                                one and `pip install -e ".[vision]"` (slow).
#   TRAIN_GATES_SKIP_APPLE=1     skip Gate 4 with a clear SKIPPED message.
#   TRAIN_GATES_SKIP_SWIFT=1     skip Gate 5 with a clear SKIPPED message.
#
# The 5 gates (hosted equivalents in parens):
#   1. Linux no-MLX pytest  (ci.yml test-matrix)   — fresh venv, `pip install
#      . --no-deps`, assert `import mlx` FAILS, run the parsed Linux pytest
#      roster with the parsed -k filter.
#   2. mypy error budget     (ci.yml type-check)    — `pip install -r
#      config/mypy-requirements.txt` then `python scripts/check_mypy_error_budget.py`.
#   3. coverage union        (ci.yml changed-lines-coverage) — combine the
#      Linux+Apple coverage .data produced by gates 1+4, emit coverage.xml,
#      then diff-cover --compare-branch <base-sha> --fail-under 100.
#   4. Apple-MLX pytest      (ci.yml test-apple-silicon) — run the parsed Apple
#      pytest roster in an mlx-capable venv.
#   5. Desktop swift test    (rapid-mac-ci.yml build) — `swift test
#      --no-parallel` in apps/ when Desktop sources changed vs <base-sha>.
#
# gates-hash: a deterministic hash over the exact gate definitions (the parsed
# Linux/Apple pytest args, the mypy script + budget files, the diff-cover
# inputs/flags, the swift invocation, and the relevant workflow step text). A
# CI-definition edit (a test added to a pytest roster, a mypy budget pin
# change, a diff_cover knob) changes the hash. Host state (venv paths,
# timestamps, machine id) is never included.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARSER_MODULE="scripts.train_gates_parser"

usage() {
  echo "usage: $0 <base-sha>" >&2
  echo "  <base-sha>  merge-base (or base commit) to diff/validate against" >&2
  exit 2
}

# ---------------------------------------------------------------------------
# Control interpreter
# ---------------------------------------------------------------------------
resolve_python() {
  if [[ -n "${TRAIN_GATES_PYTHON:-}" ]]; then
    echo "${TRAIN_GATES_PYTHON}"
    return
  fi
  if python3 -c 'import yaml, coverage, diff_cover, pytest' >/dev/null 2>&1; then
    echo "python3"
    return
  fi
  local venv_candidate="/Users/raullenstudio/work/rapid-mlx/.venv/bin/python"
  if [[ -x "$venv_candidate" ]] \
    && "$venv_candidate" -c 'import yaml, coverage, diff_cover, pytest' >/dev/null 2>&1; then
    echo "$venv_candidate"
    return
  fi
  echo "ERROR: no interpreter with yaml+coverage+diff_cover+pytest found; set TRAIN_GATES_PYTHON" >&2
  return 1
}

PYTHON_BIN="$(resolve_python)"
echo "train-gates: control interpreter: $PYTHON_BIN"

# ---------------------------------------------------------------------------
# Parse the gate surface from the workflows (single source of truth).
# ---------------------------------------------------------------------------
json_parse() {
  # $PYTHON_BIN -m scripts.train_gates_parser <target>
  ( cd "$ROOT" && "$PYTHON_BIN" -m "$PARSER_MODULE" "$1" )
}

LINUX_JSON="$(json_parse linux)"
APPLE_JSON="$(json_parse apple)"
DIFF_JSON="$(json_parse diff_cover)"
SWIFT_JSON="$(json_parse swift_test)"

# ---------------------------------------------------------------------------
# gates-hash (deterministic; no host state).
# ---------------------------------------------------------------------------
compute_gates_hash() {
  local mypy_script_sha
  mypy_script_sha="$(git -C "$ROOT" hash-object "$ROOT/scripts/check_mypy_error_budget.py")"
  # (a)+(b) parsed Linux + Apple pytest args; (c) mypy script committed hash;
  # (d) the workflow step text (captures ANY knob change). Feed the whole
  # deterministic payload through `git hash-object --stdin`.
  {
    echo "linux-pytest"
    "$PYTHON_BIN" -c 'import sys,json; print(json.dumps(json.loads(sys.argv[1]), sort_keys=True))' "$LINUX_JSON"
    echo "apple-pytest"
    "$PYTHON_BIN" -c 'import sys,json; print(json.dumps(json.loads(sys.argv[1]), sort_keys=True))' "$APPLE_JSON"
    echo "mypy-script-sha"
    echo "$mypy_script_sha"
    echo "mypy-requirements"
    git -C "$ROOT" hash-object "$ROOT/config/mypy-requirements.txt"
    echo "mypy-baseline"
    git -C "$ROOT" hash-object "$ROOT/config/mypy-error-baseline.txt"
    echo "diff-cover"
    "$PYTHON_BIN" -c 'import sys,json; print(json.dumps(json.loads(sys.argv[1]), sort_keys=True))' "$DIFF_JSON"
    echo "swift-test"
    "$PYTHON_BIN" -c 'import sys,json; print(json.dumps(json.loads(sys.argv[1]), sort_keys=True))' "$SWIFT_JSON"
    echo "workflow-step-text"
    git -C "$ROOT" hash-object .github/workflows/ci.yml
    git -C "$ROOT" hash-object .github/workflows/rapid-mac-ci.yml
  } | git -C "$ROOT" hash-object --stdin
}

GATES_HASH="$(compute_gates_hash)"

# ---------------------------------------------------------------------------
# Gate helpers
# ---------------------------------------------------------------------------
PASSED=()
SKIPPED=()
FAILED=()

note()  { printf '    %s\n' "$*"; }
passed(){ PASSED+=("$1"); }
skip()  { SKIPPED+=("$1"); printf 'GATE %s: SKIPPED — %s\n' "$1" "$2"; }
fail()  { FAILED+=("$1"); printf 'GATE %s: FAILURE — %s\n' "$1" "$2"; }

# ---------------------------------------------------------------------------
# Gate 1 — Linux no-MLX pytest (fresh venv, no mlx).
# ---------------------------------------------------------------------------
gate1_linux() {
  echo
  echo "== Gate 1: Linux no-MLX pytest =="
  local venv
  venv="$(mktemp -d /private/tmp/rapid-train-gates-venv-XXXXXX)"
  local py="$venv/bin/python"
  echo "  fresh venv: $venv"

  "$PYTHON_BIN" -m venv "$venv"
  # --no-deps mirrors the hosted Linux install precisely (linux CI installs
  # pytest bits separately; here the fresh venv gets a bootstrap pytest + the
  # package with default extras, exactly the "no MLX" contract).
  "$py" -m pip install --quiet --upgrade pip
  "$py" -m pip install --quiet "$ROOT" --no-deps
  # pytest bits needed to actually run the selected tests.
  "$py" -m pip install --quiet pytest pytest-asyncio pytest-cov pydantic fastapi jsonschema httpx psutil transformers requests pyyaml python-multipart uvicorn websockets jinja2

  if "$py" -c "import mlx" >/dev/null 2>&1; then
    fail 1 "import mlx unexpectedly SUCCEEDED in the fresh --no-deps venv; the hosted Linux gate expects no MLX"
    return 1
  fi
  note "mlx correctly absent from fresh --no-deps venv"

  # Build the pytest command from the parsed args: <paths> + deselect + -k.
  local paths_str deselect_str marker_str
  paths_str="$("$PYTHON_BIN" - "$LINUX_JSON" <<'PYEOF'
import sys, json
print(" ".join(json.loads(sys.argv[1])["paths"]))
PYEOF
)"
  deselect_str="$("$PYTHON_BIN" - "$LINUX_JSON" <<'PYEOF'
import sys, json
args = json.loads(sys.argv[1])
print(" ".join("--deselect=%s" % d for d in args["deselect"]))
PYEOF
)"
  marker_str="$("$PYTHON_BIN" - "$LINUX_JSON" <<'PYEOF'
import sys, json
m = json.loads(sys.argv[1]).get("marker")
print(m if m else "")
PYEOF
)"

  echo "  running Linux pytest roster ($("$PYTHON_BIN" -c 'import sys,json; print(len(json.loads(sys.argv[1])["paths"]))' "$LINUX_JSON") test files)"
  local covfile="${venv}/coverage-linux-3.11.data"
  # shellcheck disable=SC2086 - intended word-splitting (paths/deselect/marker)
  if ! ( cd "$ROOT" \
      && COVERAGE_FILE="$covfile" \
      "$py" -m pytest \
        $paths_str \
        $deselect_str \
        -v --tb=short \
        -k "$marker_str" \
        --cov=vllm_mlx \
        --cov-report=term-missing \
        --cov-report=json:"${venv}/cov-linux.json" ); then
    fail 1 "Linux no-MLX pytest failed (see output above)"
    return 1
  fi
  cp "$covfile" "$ROOT/coverage-linux-3.11.data"
  note "Linux coverage data written to coverage-linux-3.11.data"
  passed 1
}

# ---------------------------------------------------------------------------
# Gate 2 — pinned mypy error budget.
# ---------------------------------------------------------------------------
gate2_mypy() {
  echo
  echo "== Gate 2: mypy error budget =="
  if ! "$PYTHON_BIN" -c "import yaml" >/dev/null 2>&1; then
    fail 2 "control interpreter lacks pyyaml; cannot parse the workflow"
    return 1
  fi
  if [[ ! -f "$ROOT/config/mypy-requirements.txt" ]]; then
    fail 2 "config/mypy-requirements.txt missing"
    return 1
  fi
  local venv
  venv="$(mktemp -d /private/tmp/rapid-train-gates-mypy-XXXXXX)"
  local py="$venv/bin/python"
  echo "  mypy venv: $venv"
  "$PYTHON_BIN" -m venv "$venv"
  "$py" -m pip install --quiet --upgrade pip
  "$py" -m pip install --quiet --requirement "$ROOT/config/mypy-requirements.txt"
  note "installed pinned mypy environment"
  if ! ( cd "$ROOT" && "$py" "$ROOT/scripts/check_mypy_error_budget.py" ); then
    fail 2 "mypy error budget overrun (see output above); freeze blocked"
    return 1
  fi
  passed 2
}

# ---------------------------------------------------------------------------
# Gate 3 — coverage union + diff-cover.
# ---------------------------------------------------------------------------
gate3_diffcover() {
  echo
  echo "== Gate 3: coverage union + diff-cover =="
  local base_sha="$1"
  local linux_data="$ROOT/coverage-linux-3.11.data"
  local apple_data="$ROOT/coverage-apple.data"
  if [[ ! -f "$linux_data" ]]; then
    fail 3 "missing $linux_data (Gate 1 must produce it)"
    return 1
  fi
  if [[ ! -f "$apple_data" ]]; then
    fail 3 "missing $apple_data (Gate 4 must produce it)"
    return 1
  fi
  if ! "$PYTHON_BIN" -c "import coverage, diff_cover" >/dev/null 2>&1; then
    fail 3 "control interpreter lacks coverage/diff_cover"
    return 1
  fi
  # Reproduce the hosted changed-lines-coverage job exactly: combine + xml +
  # diff-cover all run FROM the repo root (so `.coveragerc` applies, the
  # relative_files coverage paths resolve, and `--compare-branch <base-sha>`
  # resolves against `.git`). combine merges the two .data files passed by
  # absolute path into `./.coverage`; we clean the transient artifacts after.
  local work_xml="${ROOT}/coverage.xml"
  local combined="${ROOT}/.coverage"
  rm -f "$combined" "$work_xml"
  if ! ( cd "$ROOT" \
      && "$PYTHON_BIN" -m coverage combine "$linux_data" "$apple_data" \
      && "$PYTHON_BIN" -m coverage xml -o "$work_xml" ); then
    fail 3 "coverage combine/xml failed (see output above)"
    rm -f "$combined" "$work_xml"
    return 1
  fi
  if ! ( cd "$ROOT" \
      && "$PYTHON_BIN" -m diff_cover.diff_cover_tool \
          "$work_xml" \
          --compare-branch "$base_sha" \
          --show-uncovered \
          --fail-under 100 ); then
    rm -f "$combined" "$work_xml"
    fail 3 "diff-cover --compare-branch $base_sha failed (see output above)"
    return 1
  fi
  rm -f "$combined" "$work_xml"
  passed 3
}

# ---------------------------------------------------------------------------
# Gate 4 — Apple-MLX pytest.
# ---------------------------------------------------------------------------
gate4_apple() {
  echo
  echo "== Gate 4: Apple-MLX pytest =="
  if [[ "${TRAIN_GATES_SKIP_APPLE:-0}" == "1" ]]; then
    skip 4 "TRAIN_GATES_SKIP_APPLE=1"
    return 0
  fi

  local apple_py=""
  if [[ -n "${TRAIN_GATES_APPLE_VENV:-}" ]]; then
    apple_py="$TRAIN_GATES_APPLE_VENV/bin/python"
    if [[ ! -x "$apple_py" ]]; then
      fail 4 "TRAIN_GATES_APPLE_VENV=$TRAIN_GATES_APPLE_VENV has no bin/python"
      return 1
    fi
  elif [[ "${TRAIN_GATES_ALLOW_APPLE_INSTALL:-0}" == "1" ]]; then
    local venv
    venv="$(mktemp -d /private/tmp/rapid-train-gates-apple-XXXXXX)"
    apple_py="$venv/bin/python"
    echo "  creating Apple venv: $venv"
    "$PYTHON_BIN" -m venv "$venv"
    "$apple_py" -m pip install --quiet --upgrade pip
    "$apple_py" -m pip install --quiet -e ".[vision]"
    "$apple_py" -m pip install --quiet pytest pytest-asyncio pytest-cov
  else
    fail 4 "no Apple venv provided; set TRAIN_GATES_APPLE_VENV=<venv with the package + mlx>, TRAIN_GATES_ALLOW_APPLE_INSTALL=1 to create one, or TRAIN_GATES_SKIP_APPLE=1"
    return 1
  fi

  if ! "$apple_py" -c "import mlx.core as mx" >/dev/null 2>&1; then
    fail 4 "Apple venv cannot import mlx (not an Apple-Silicon runtime?)"
    return 1
  fi

  local paths_str
  paths_str="$("$PYTHON_BIN" - "$APPLE_JSON" <<'PYEOF'
import sys, json
print(" ".join(json.loads(sys.argv[1])["paths"]))
PYEOF
)"

  echo "  running Apple-MLX pytest roster ($("$PYTHON_BIN" -c 'import sys,json; print(len(json.loads(sys.argv[1])["paths"]))' "$APPLE_JSON") test files)"
  if ! ( cd "$ROOT" \
      && COVERAGE_FILE="$ROOT/coverage-apple.data" \
      "$apple_py" -m pytest \
        $paths_str \
        -v --tb=short \
        -m "not slow" \
        -k "not Integration" \
        --cov=vllm_mlx \
        --cov-report=term-missing ); then
    fail 4 "Apple-MLX pytest failed (see output above)"
    return 1
  fi
  note "Apple coverage data written to coverage-apple.data"
  passed 4
}

# ---------------------------------------------------------------------------
# Gate 5 — Desktop swift test.
# ---------------------------------------------------------------------------
gate5_swift() {
  echo
  echo "== Gate 5: Desktop swift test =="
  if [[ "${TRAIN_GATES_SKIP_SWIFT:-0}" == "1" ]]; then
    skip 5 "TRAIN_GATES_SKIP_SWIFT=1"
    return 0
  fi
  local base_sha="$1"
  local desktop_dir="$ROOT/apps/rapid-mac"
  if [[ ! -d "$desktop_dir" ]]; then
    skip 5 "Desktop app dir $desktop_dir absent"
    return 0
  fi
  if ! command -v swift >/dev/null 2>&1; then
    skip 5 "swift toolchain not on PATH"
    return 0
  fi
  if ! git -C "$ROOT" diff --quiet "$base_sha" -- apps/; then
    note "apps/ changed vs $base_sha — running Desktop tests"
  else
    skip 5 "apps/ unchanged vs $base_sha"
    return 0
  fi
  ( cd "$desktop_dir" && swift test --no-parallel ) || {
    fail 5 "swift test --no-parallel failed (see output above)"
    return 1
  }
  passed 5
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if [[ $# -lt 1 ]]; then
  usage
fi
BASE_SHA="${1}"
BASE_SHA_RESOLVED="$(git -C "$ROOT" rev-parse --verify "${BASE_SHA}^{commit}")"
echo "train-gates: base-sha resolved -> $BASE_SHA_RESOLVED"
echo "train-gates: gates-hash        -> $GATES_HASH"
if [[ -n "$(git -C "$ROOT" status --porcelain)" ]]; then
  echo "train-gates: NOTE: working tree is dirty; hash covers committed gate defs only."
fi

# Design: even if one gate fails we keep going so the operator sees every
# failure in one pass. Each gate writes its own PASSED/FAILED/SKIPPED entry
# and returns non-zero on failure; we ignore that code here (set -e would
# otherwise abort the run) and decide the exit status from the FAILED array.
gate1_linux "$BASE_SHA_RESOLVED" || true
gate2_mypy || true
gate3_diffcover "$BASE_SHA_RESOLVED" || true
gate4_apple || true
gate5_swift "$BASE_SHA_RESOLVED" || true

echo
printf 'PASSED (%d): %s\n' "${#PASSED[@]}" "${PASSED[*]:-none}"
if [[ "${#SKIPPED[@]}" -gt 0 ]]; then
  printf 'SKIPPED (%d): %s\n' "${#SKIPPED[@]}" "${SKIPPED[*]}"
fi
if [[ "${#FAILED[@]}" -gt 0 ]]; then
  printf 'FAILED (%d): %s\n' "${#FAILED[@]}" "${FAILED[*]}"
  echo
  echo "GATES FAILED — a gate below blocked the freeze:"
  for g in "${FAILED[@]}"; do
    echo "  FAILURE gate $g"
  done
  exit 1
fi

# The freeze contract requires all 5 gates to have PASSED (not merely skipped).
if [[ "${#PASSED[@]}" -ne 5 ]]; then
  echo "GATES INCOMPLETE — ${#PASSED[@]}/5 passed (${#SKIPPED[@]} skipped). A skip is not a pass; rerun to exercise all gates." >&2
  exit 1
fi

echo
echo "GATES OK ${BASE_SHA_RESOLVED} ${GATES_HASH}"
