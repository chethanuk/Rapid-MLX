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
#                                one and `pip install -e "$ROOT[vision]"` (slow).
#   TRAIN_GATES_SKIP_APPLE=1     skip Gate 4 with a clear SKIPPED message
#                                (a skip is NOT a pass).
#   TRAIN_GATES_SKIP_SWIFT=1     skip Gate 5 with a clear SKIPPED message
#                                (a skip is NOT a pass).
#
# The 5 gates (hosted equivalents in parens):
#   1. Linux no-MLX pytest  (ci.yml test-matrix)   — fresh venv, `pip install
#      . --no-deps`, assert `import mlx` FAILS, then run the parsed Linux pytest
#      invocations (one process per `pytest` block in ci.yml — the broad roster
#      and the engine-lifecycle seam set run in separate processes, mirroring
#      the hosted split, with the second `--cov-append`ing into the same data).
#   2. mypy error budget     (ci.yml type-check)    — `pip install -r
#      config/mypy-requirements.txt` then `python scripts/check_mypy_error_budget.py`.
#   3. coverage union        (ci.yml changed-lines-coverage) — combine the
#      Linux+Apple coverage .data produced by gates 1+4, emit coverage.xml,
#      then diff-cover --compare-branch <base-sha> --fail-under 100.
#   4. Apple-MLX pytest      (ci.yml test-apple-silicon) — run the parsed Apple
#      pytest roster (with ci.yml's -m / -k filters) in an mlx-capable venv.
#   5. Desktop swift test    (rapid-mac-ci.yml build, the `swift test` step
#      only) — `swift test --no-parallel` in apps/ when Desktop sources changed
#      vs <base-sha>; when apps/ is unchanged the gate is PASS-BY-N/A (counts
#      toward the 5-pass contract).
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
# Scratch dir (honors TMPDIR) + exit cleanup. All transient coverage artifacts
# live here — NEVER the repo root — so a second run on a new head cannot
# silently union stale coverage from the previous head.
# ---------------------------------------------------------------------------
RUN_TMP="$(mktemp -d "${TMPDIR:-/tmp}/rapid-train-gates-run-XXXXXX")"
COV_DIR="$RUN_TMP/cov"
mkdir -p "$COV_DIR"
cleanup() { rm -rf "$RUN_TMP"; }
trap cleanup EXIT

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
  local venv_candidate="$ROOT/.venv/bin/python"
  if [[ -x "$venv_candidate" ]] \
    && "$venv_candidate" -c 'import yaml, coverage, diff_cover, pytest' >/dev/null 2>&1; then
    echo "$venv_candidate"
    return
  fi
  echo "ERROR: no interpreter with yaml+coverage+diff_cover+pytest found; set TRAIN_GATES_PYTHON" >&2
  return 1
}

PYTHON_BIN="$(resolve_python)"
PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
echo "train-gates: control interpreter: $PYTHON_BIN (python $PY_VERSION)"

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
passed(){ PASSED+=("$1"); if [[ $# -ge 2 ]]; then printf 'GATE %s: PASS — %s\n' "$1" "$2"; else printf 'GATE %s: PASS\n' "$1"; fi; }
passed_na(){ PASSED+=("$1"); if [[ $# -ge 2 ]]; then printf 'GATE %s: PASS (N/A) — %s\n' "$1" "$2"; else printf 'GATE %s: PASS (N/A)\n' "$1"; fi; }
skip()  { SKIPPED+=("$1"); printf 'GATE %s: SKIPPED — %s\n' "$1" "${2:-}"; }
fail()  { FAILED+=("$1"); if [[ $# -ge 2 ]]; then printf 'GATE %s: FAILURE — %s\n' "$1" "$2"; else printf 'GATE %s: FAILURE\n' "$1"; fi; }

# ---------------------------------------------------------------------------
# Gate 1 — Linux no-MLX pytest (fresh venv, no mlx), one process per ci.yml
# pytest block.
# ---------------------------------------------------------------------------
gate1_linux() {
  echo
  echo "== Gate 1: Linux no-MLX pytest =="
  local venv
  venv="$(mktemp -d "${TMPDIR:-/tmp}/rapid-train-gates-venv-XXXXXX")"
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

  # ci.yml runs TWO separate pytest processes in this step (the broad roster
  # and the engine-lifecycle seam set — see scripts/train_gates_parser.py).
  # Reproduce that split: run each parsed invocation in its OWN pytest process,
  # in ci.yml's order, each writing into the SAME coverage file with the
  # declared --cov-append semantics so the union equals the hosted combined
  # Linux coverage.
  local n_inv
  n_inv="$("$PYTHON_BIN" -c 'import sys,json; print(len(json.loads(sys.argv[1])))' "$LINUX_JSON")"
  echo "  running $n_inv Linux pytest process(es) parsed from ci.yml"
  local covfile="$COV_DIR/coverage-linux-3.11.data"
  for (( i=0; i<n_inv; i++ )); do
    local n_files cov_append paths_str deselect_str marker_str
    n_files="$("$PYTHON_BIN" - "$LINUX_JSON" "$i" <<'PY'
import sys, json
inv = json.loads(sys.argv[1])[int(sys.argv[2])]
print(len(inv["paths"]))
PY
)"
    cov_append="$("$PYTHON_BIN" - "$LINUX_JSON" "$i" <<'PY'
import sys, json
inv = json.loads(sys.argv[1])[int(sys.argv[2])]
print("1" if inv["cov_declaration"]["cov_append"] else "0")
PY
)"
    echo "    pytest process $((i+1))/$n_inv ($n_files test file tokens, cov_append=$cov_append)"

    # Split the parsed space-joined token string into an array so the paths
    # reach pytest verbatim (no accidental globbing/word-splitting).
    paths_str="$("$PYTHON_BIN" - "$LINUX_JSON" "$i" <<'PY'
import sys, json
inv = json.loads(sys.argv[1])[int(sys.argv[2])]
print(" ".join(inv["paths"]))
PY
)"
    deselect_str="$("$PYTHON_BIN" - "$LINUX_JSON" "$i" <<'PY'
import sys, json
inv = json.loads(sys.argv[1])[int(sys.argv[2])]
print(" ".join("--deselect=%s" % d for d in inv["deselect"]))
PY
)"
    marker_str="$("$PYTHON_BIN" - "$LINUX_JSON" "$i" <<'PY'
import sys, json
inv = json.loads(sys.argv[1])[int(sys.argv[2])]
print(inv["marker"] or "")
PY
)"

    # Split the parsed space-joined strings into arrays so no accidental
    # word-splitting/globbing ever occurs when they are passed to pytest.
    # NOTE (bash-3.2 + set -u): `read -a` on an EMPTY string leaves the array
    # unbound, and `${arr[@]}` on an unbound array errors under `set -u`. So
    # every array expansion below uses the `${arr[@]+"${arr[@]}"}` guard, which
    # is a no-op when the array holds no elements.
    local -a pytest_args=() deselect_args=()
    read -r -a pytest_args <<<"$paths_str"
    read -r -a deselect_args <<<"$deselect_str"
    local -a aux_args=()
    if [[ "$cov_append" == "1" ]]; then aux_args+=(--cov-append); fi
    if [[ -n "$marker_str" ]]; then aux_args+=(-k "$marker_str"); fi

    if ! ( cd "$ROOT" \
        && COVERAGE_FILE="$covfile" \
        "$py" -m pytest \
          ${pytest_args[@]+"${pytest_args[@]}"} \
          ${deselect_args[@]+"${deselect_args[@]}"} \
          ${aux_args[@]+"${aux_args[@]}"} \
          -v --tb=short \
          --cov=vllm_mlx \
          --cov-report=term-missing ); then
      fail 1 "Linux no-MLX pytest process $((i+1)) failed (see output above)"
      return 1
    fi
  done
  note "Linux coverage data written to $covfile"
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
  venv="$(mktemp -d "${TMPDIR:-/tmp}/rapid-train-gates-mypy-XXXXXX")"
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
  local linux_data="$COV_DIR/coverage-linux-3.11.data"
  local apple_data="$COV_DIR/coverage-apple.data"
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
  # resolves against `.git`). The .data inputs live under COV_DIR (never the
  # repo root), and the transient combined `./.coverage` + `coverage.xml` are
  # pointed into COV_DIR too so no coverage artifact ever lands in the repo
  # root (a stale one there would silently union on the next run's new head).
  local combined="$COV_DIR/.coverage"
  local work_xml="$COV_DIR/coverage.xml"
  rm -f "$combined" "$work_xml"
  if ! ( cd "$ROOT" \
      && COVERAGE_FILE="$combined" "$PYTHON_BIN" -m coverage combine "$linux_data" "$apple_data" \
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
    venv="$(mktemp -d "${TMPDIR:-/tmp}/rapid-train-gates-apple-XXXXXX")"
    apple_py="$venv/bin/python"
    echo "  creating Apple venv: $venv"
    "$PYTHON_BIN" -m venv "$venv"
    "$apple_py" -m pip install --quiet --upgrade pip
    "$apple_py" -m pip install --quiet -e "${ROOT}[vision]"
    "$apple_py" -m pip install --quiet pytest pytest-asyncio pytest-cov
  else
    fail 4 "no Apple venv provided; set TRAIN_GATES_APPLE_VENV=<venv with the package + mlx>, TRAIN_GATES_ALLOW_APPLE_INSTALL=1 to create one, or TRAIN_GATES_SKIP_APPLE=1"
    return 1
  fi

  if ! "$apple_py" -c "import mlx.core as mx" >/dev/null 2>&1; then
    fail 4 "Apple venv cannot import mlx (not an Apple-Silicon runtime?)"
    return 1
  fi

  # ci.yml's Apple -m / -k filters, parsed (not hardcoded) — if ci.yml changes
  # them, Gate 4 follows.
  local paths_str m_str k_str
  paths_str="$("$PYTHON_BIN" - "$APPLE_JSON" <<'PY'
import sys, json
print(" ".join(json.loads(sys.argv[1])["paths"]))
PY
)"
  m_str="$("$PYTHON_BIN" - "$APPLE_JSON" <<'PY'
import sys, json
print(json.loads(sys.argv[1]).get("m") or "")
PY
)"
  k_str="$("$PYTHON_BIN" - "$APPLE_JSON" <<'PY'
import sys, json
print(json.loads(sys.argv[1]).get("k") or "")
PY
)"

  echo "  running Apple-MLX pytest roster ($("$PYTHON_BIN" -c 'import sys,json; print(len(json.loads(sys.argv[1])["paths"]))' "$APPLE_JSON") test files)"
  local -a apple_args=() m_args=() k_args=()
  read -r -a apple_args <<<"$paths_str"
  if [[ -n "$m_str" ]]; then m_args=(-m "$m_str"); fi
  if [[ -n "$k_str" ]]; then k_args=(-k "$k_str"); fi
  if ! ( cd "$ROOT" \
      && COVERAGE_FILE="$COV_DIR/coverage-apple.data" \
      "$apple_py" -m pytest \
        ${apple_args[@]+"${apple_args[@]}"} \
        -v --tb=short \
        ${m_args[@]+"${m_args[@]}"} \
        ${k_args[@]+"${k_args[@]}"} \
        --cov=vllm_mlx \
        --cov-report=term-missing ); then
    fail 4 "Apple-MLX pytest failed (see output above)"
    return 1
  fi
  note "Apple coverage data written to $COV_DIR/coverage-apple.data"
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
    # apps/ unchanged => there is nothing for the build job's swift test step
    # to exercise. Count as a PASS-BY-N/A (it satisfies the "all five must
    # pass" contract) rather than a skip: unlike an environment skip (no swift
    # toolchain, no apps/ dir, TRAIN_GATES_SKIP_*), this is a *deterministic*
    # property of the diff and always yields the same true outcome.
    passed_na 5 "apps/ unchanged vs $base_sha; the build's swift test step has nothing to run"
    return 0
  fi
  ( cd "$desktop_dir" && swift test --no-parallel ) || {
    fail 5 "swift test --no-parallel failed (see output above)"
    return 1
  }
  passed 5 "Desktop swift test passed"
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

# B4: reject a wrong/forward base. diff-cover compares against <base-sha>; it
# must be a real ancestor of HEAD, else a forward/foreign base yields a trivial
# (often empty) diff and a guaranteed 100% pass — a false green.
if ! git -C "$ROOT" merge-base --is-ancestor "$BASE_SHA_RESOLVED" HEAD; then
  echo "ERROR: base $BASE_SHA_RESOLVED is NOT an ancestor of HEAD." >&2
  echo "       A forward/wrong base would make diff-cover compare an empty/trivial diff and pass 100% by construction." >&2
  exit 2
fi

# B2: never tolerate stale coverage artifacts from a previous run in the repo
# root — the next run against a NEW head would silently union old coverage into
# a false green. (All current-run artifacts live under $COV_DIR.)
for stale in "$ROOT"/coverage-*.data "$ROOT"/coverage.xml "$ROOT"/.coverage; do
  [[ -e "$stale" ]] && rm -f "$stale" && echo "train-gates: removed stale $stale"
done

# The validated head is the current checkout HEAD (what we're actually testing).
HEAD_RESOLVED="$(git -C "$ROOT" rev-parse HEAD)"
GIT_DIRTY=""
if [[ -n "$(git -C "$ROOT" status --porcelain)" ]]; then
  GIT_DIRTY=1
  echo "train-gates: WARNING: working tree is DIRTY; the hash below covers committed gate defs only."
fi

# Design: even if one gate fails we keep going so the operator sees every
# failure in one pass. Each gate writes its own PASSED/FAILED/SKIPPED entry
# and returns non-zero on failure; we ignore that code here (set -e would
# otherwise abort the run) and decide the exit status from the FAILED array.
#
# Order matters: Gate 4 (Apple) must produce coverage-apple.data BEFORE Gate 3
# (coverage union + diff-cover) combines it, so the run order is 1 -> 2 -> 4 ->
# 3 -> 5.
gate1_linux "$BASE_SHA_RESOLVED" || true
gate2_mypy || true
gate4_apple || true
gate3_diffcover "$BASE_SHA_RESOLVED" || true
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

# The freeze contract requires all 5 gates to have PASSED (skips are NOT
# passes; PASS-BY-N/A counts).
if [[ "${#PASSED[@]}" -ne 5 ]]; then
  echo "GATES INCOMPLETE — ${#PASSED[@]}/5 passed (${#SKIPPED[@]} skipped). A skip is not a pass; rerun to exercise all gates." >&2
  exit 1
fi

if [[ -n "$GIT_DIRTY" ]]; then
  echo
  echo "^^MARK^^ GATES OK ${HEAD_RESOLVED} ${GATES_HASH}  (python ${PY_VERSION})  [DIRTY TREE — freeze not trustworthy]"
else
  echo
  echo "GATES OK ${HEAD_RESOLVED} ${GATES_HASH}  (python ${PY_VERSION})"
fi
