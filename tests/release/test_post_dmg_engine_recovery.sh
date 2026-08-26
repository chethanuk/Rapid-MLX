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
contains "$VERIFY" 'recovery reason must be a single line' "multiline recovery reasons fail before evidence"
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
contains "$RECOVER" 'git ls-remote --exit-code --refs origin "refs/tags/${ENGINE_TAG}"' "engine tag is re-read after approval"
contains "$RECOVER" 'if [ "$ENGINE_SHA" != "$RELEASE_SHA" ]' "post-approval engine tag mismatch fails closed"
contains "$RECOVER" 'bash scripts/recover_engine_release.sh' "testable recovery publication unit is executed"
contains "$RECOVER" 'VERSION: ${{ needs.verify-published-desktop.outputs.version }}' "publication receives the verified version"
contains "$RECOVER" 'RELEASE_SHA: ${{ needs.verify-published-desktop.outputs.release_sha }}' "publication receives the verified SHA"
contains "$RECOVER" 'NOTES_FILE: ${{ github.workspace }}/evidence/release-notes.md' "publication receives immutable notes"
contains "$RECOVER" 'REASON: ${{ inputs.reason }}' "publication receives the operator reason"
lacks "$ALL" 'tag_desktop_app.sh' "recovery never creates or moves the Desktop tag"
lacks "$ALL" 'rapid-mac-release.yml --ref' "recovery never dispatches another DMG build"

# Execute the actual post-approval run block with controlled command doubles.
# A textual ordering assertion alone cannot prove that `set -e` propagates each
# live-check failure before the following publication step becomes reachable.
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/bin" "$TMP/evidence"
sed -n '/- name: Re-verify blockers and immutable Desktop publication/,/- name: Recover the engine tag and Release at the Desktop candidate SHA/p' "$WORKFLOW" \
  | sed '1,/run: |/d; $d; s/^          //' > "$TMP/reverify.sh"
chmod +x "$TMP/reverify.sh"
printf '%s\n' \
  'VERSION=0.13.0' \
  'RELEASE_SHA=0000000000000000000000000000000000000001' \
  > "$TMP/evidence/recovery-identity.txt"

cat > "$TMP/bin/python3" <<'SH'
#!/usr/bin/env bash
echo "python3 $*" >> "$CALLS"
case "$1" in
  *check_release_blockers.py) exit "${BLOCKERS_RC:-0}" ;;
  *check_desktop_publish.py) exit "${DESKTOP_RC:-0}" ;;
esac
exit 0
SH
cat > "$TMP/bin/git" <<'SH'
#!/usr/bin/env bash
echo "git $*" >> "$CALLS"
case "$1" in
  ls-remote) exit "${TAG_QUERY_RC:-2}" ;;
  fetch) exit "${TAG_FETCH_RC:-0}" ;;
  rev-parse) printf '%s\n' "${ENGINE_SHA:-0000000000000000000000000000000000000001}" ;;
esac
exit 0
SH
chmod +x "$TMP/bin/python3" "$TMP/bin/git"

run_transaction() {
  : > "$TMP/calls"
  rm -f "$TMP/published"
  if (cd "$TMP" && env \
      PATH="$TMP/bin:/usr/bin:/bin" \
      CALLS="$TMP/calls" \
      RUNNER_TEMP="$TMP" \
      LIVE_HAVE_PAT="${LIVE_HAVE_PAT:-true}" \
      VERSION=0.13.0 \
      RELEASE_SHA=0000000000000000000000000000000000000001 \
      EXPECTED_OPEN_IDS= \
      REPO=raullenchai/Rapid-MLX \
      BLOCKERS_RC="${BLOCKERS_RC:-0}" \
      DESKTOP_RC="${DESKTOP_RC:-0}" \
      TAG_QUERY_RC="${TAG_QUERY_RC:-2}" \
      ENGINE_SHA="${ENGINE_SHA:-0000000000000000000000000000000000000001}" \
      bash "$TMP/reverify.sh"); then
    touch "$TMP/published"
    return 0
  fi
  return 1
}

BLOCKERS_RC=1 DESKTOP_RC=0 TAG_QUERY_RC=2 run_transaction || true
[ ! -e "$TMP/published" ] && ok "live blocker failure makes publication unreachable" || bad "blocker failure reached publication"
LIVE_HAVE_PAT=false BLOCKERS_RC=0 DESKTOP_RC=0 TAG_QUERY_RC=2 run_transaction || true
[ ! -e "$TMP/published" ] && ok "post-approval PAT loss makes publication unreachable" || bad "PAT loss reached publication"
BLOCKERS_RC=0 DESKTOP_RC=1 TAG_QUERY_RC=2 run_transaction || true
[ ! -e "$TMP/published" ] && ok "Desktop evidence failure makes publication unreachable" || bad "Desktop failure reached publication"
BLOCKERS_RC=0 DESKTOP_RC=0 TAG_QUERY_RC=1 run_transaction || true
[ ! -e "$TMP/published" ] && ok "engine-tag API failure makes publication unreachable" || bad "tag query failure reached publication"
BLOCKERS_RC=0 DESKTOP_RC=0 TAG_QUERY_RC=0 ENGINE_SHA=ffffffffffffffffffffffffffffffffffffffff run_transaction || true
[ ! -e "$TMP/published" ] && ok "post-approval engine-tag mismatch makes publication unreachable" || bad "tag mismatch reached publication"
BLOCKERS_RC=0 DESKTOP_RC=0 TAG_QUERY_RC=0 ENGINE_SHA=0000000000000000000000000000000000000001 run_transaction
[ -e "$TMP/published" ] && ok "same-SHA engine tag permits idempotent recovery" || bad "same-SHA tag blocked recovery"
BLOCKERS_RC=0 DESKTOP_RC=0 TAG_QUERY_RC=2 run_transaction
[ -e "$TMP/published" ] && ok "absent engine tag permits recovery publication" || bad "valid recovery did not reach publication"

# Execute the real publication unit with a create_release.sh double. This
# proves the final workflow step constructs TAG from the verified version and
# preserves the exact RELEASE_SHA / NOTES_FILE values passed by the workflow.
mkdir -p "$TMP/publication/scripts"
cp "$REPO_ROOT/scripts/recover_engine_release.sh" "$TMP/publication/scripts/"
cat > "$TMP/publication/scripts/create_release.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
{
  printf 'TAG=%s\n' "$TAG"
  printf 'RELEASE_SHA=%s\n' "$RELEASE_SHA"
  printf 'NOTES_FILE=%s\n' "$NOTES_FILE"
  printf 'REASON=%s\n' "$REASON"
} > "$PUBLICATION_CALL"
SH
chmod +x "$TMP/publication/scripts/create_release.sh"
printf 'release notes\n' > "$TMP/publication/notes.md"

run_publication() {
  rm -f "$TMP/publication/call"
  env \
    VERSION=0.13.0 \
    RELEASE_SHA=0000000000000000000000000000000000000001 \
    NOTES_FILE="$TMP/publication/notes.md" \
    REASON="$1" \
    PUBLICATION_CALL="$TMP/publication/call" \
    bash "$TMP/publication/scripts/recover_engine_release.sh"
}

run_publication 'recover missing engine half'
contains "$(cat "$TMP/publication/call")" 'TAG=v0.13.0' "publication constructs the exact engine tag"
contains "$(cat "$TMP/publication/call")" 'RELEASE_SHA=0000000000000000000000000000000000000001' "publication preserves the approved SHA"
contains "$(cat "$TMP/publication/call")" "NOTES_FILE=$TMP/publication/notes.md" "publication preserves the immutable notes path"
contains "$(cat "$TMP/publication/call")" 'REASON=recover missing engine half' "publication preserves the audited reason"

if run_publication $'forged reason\nSECOND_RECORD'; then
  bad "multiline LF reason reached create_release"
else
  [ ! -e "$TMP/publication/call" ] && ok "LF reason is rejected before create_release" || bad "LF reason reached create_release"
fi
if run_publication $'forged reason\rSECOND_RECORD'; then
  bad "multiline CR reason reached create_release"
else
  [ ! -e "$TMP/publication/call" ] && ok "CR reason is rejected before create_release" || bad "CR reason reached create_release"
fi

echo "passed: $PASS failed: $FAIL"
[ "$FAIL" -eq 0 ]
