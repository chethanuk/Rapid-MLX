#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORKFLOW="$REPO_ROOT/.github/workflows/post-dmg-engine-recovery.yml"
PASS=0
FAIL=0

ok() { PASS=$((PASS + 1)); printf '  PASS %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf '  FAIL %s\n' "$1"; }
contains() { if grep -qF -- "$2" <<<"$1"; then ok "$3"; else bad "$3"; fi; }
lacks() { if grep -qF -- "$2" <<<"$1"; then bad "$3"; else ok "$3"; fi; }

ALL=$(cat "$WORKFLOW")
VERIFY=$(sed -n '/^  verify-published-desktop:/,/^  recover-engine-release:/p' "$WORKFLOW")
RECOVER=$(sed -n '/^  recover-engine-release:/,$p' "$WORKFLOW")

contains "$ALL" "workflow_dispatch:" "recovery is explicit, never automatic"
lacks "$ALL" "push:" "recovery cannot run from an ordinary main push"
contains "$VERIFY" 'EVENT_REF: ${{ github.ref }}' "preflight binds dispatch to main"
contains "$VERIFY" 'refs/tags/${APP_TAG}^{commit}' "Desktop tag is the recovery identity SSOT"
contains "$VERIFY" 'git merge-base --is-ancestor "$RELEASE_SHA" origin/main' "candidate must belong to main history"
contains "$VERIFY" 'git show "${RELEASE_SHA}:pyproject.toml"' "version is read from the recovered candidate"
contains "$VERIFY" 'if [ "$ENGINE_SHA" != "$RELEASE_SHA" ]' "a mismatched existing engine tag fails before approval"
contains "$VERIFY" 'scripts/check_desktop_publish.py' "exact Desktop publication is verified before approval"
contains "$VERIFY" 'scripts/check_release_blockers.py' "release blockers are checked before approval"
contains "$VERIFY" 'scripts/build_release_notes.sh' "notes are generated for the immutable candidate"
contains "$VERIFY" 'git worktree add --detach "$RECOVERY_SOURCE" "$RELEASE_SHA"' "curated notes are read from the immutable candidate tree"
contains "$VERIFY" 'path: recovery-evidence/' "evidence upload has one stable artifact root"

ENV_LINE=$(grep -n 'environment: rapid-mac-tag' "$WORKFLOW" | cut -d: -f1)
RECHECK_LINE=$(grep -n 'Re-verify blockers and immutable Desktop publication' "$WORKFLOW" | cut -d: -f1)
CREATE_LINE=$(grep -n 'Recover the engine tag and Release at the Desktop candidate SHA' "$WORKFLOW" | cut -d: -f1)
if [ "$ENV_LINE" -lt "$RECHECK_LINE" ] && [ "$RECHECK_LINE" -lt "$CREATE_LINE" ]; then
  ok "protected approval precedes live rechecks and engine publication"
else
  bad "protected approval/recheck/publication ordering"
fi
contains "$RECOVER" 'scripts/check_release_environment.py' "live environment protection is re-read after approval"
contains "$RECOVER" "LIVE_HAVE_PAT: \${{ secrets.RELEASE_PAT != '' }}" "PAT presence is re-evaluated inside the protected job"
lacks "$RECOVER" 'needs.verify-published-desktop.outputs.have_pat' "protected job never trusts a stale pre-approval PAT output"
contains "$RECOVER" 'scripts/check_desktop_publish.py' "Desktop publication is re-verified after approval"
contains "$RECOVER" 'grep -Fx "RELEASE_SHA=$RELEASE_SHA" evidence/recovery-identity.txt' "downloaded evidence is bound to the approved SHA"
contains "$RECOVER" '--expected-open-ids "$EXPECTED_OPEN_IDS"' "blocker set drift fails closed"
contains "$RECOVER" 'bash scripts/create_release.sh' "existing idempotent engine release helper is reused"
lacks "$ALL" 'tag_desktop_app.sh' "recovery never creates or moves the Desktop tag"
lacks "$ALL" 'rapid-mac-release.yml --ref' "recovery never dispatches another DMG build"

echo "passed: $PASS failed: $FAIL"
[ "$FAIL" -eq 0 ]
