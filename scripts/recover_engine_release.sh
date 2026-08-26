#!/usr/bin/env bash
# Publish the engine half of a verified post-DMG recovery transaction.
#
# Live blocker, environment, Desktop artifact and tag checks belong to the
# protected workflow job. This unit owns the final workflow-to-create_release
# wiring so it can be executed offline with a create_release.sh double.

set -euo pipefail

: "${VERSION:?recover_engine_release.sh: VERSION is required}"
: "${RELEASE_SHA:?recover_engine_release.sh: RELEASE_SHA is required}"
: "${NOTES_FILE:?recover_engine_release.sh: NOTES_FILE is required}"
: "${REASON:?recover_engine_release.sh: REASON is required}"

case "$REASON" in
  *$'\r'*|*$'\n'*)
    echo "recover_engine_release.sh: REASON must be a single line" >&2
    exit 2
    ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TAG="v${VERSION}"
export RELEASE_SHA NOTES_FILE

bash "$SCRIPT_DIR/create_release.sh"
