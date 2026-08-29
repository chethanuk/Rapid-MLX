#!/usr/bin/env bash
# Run the first native XCUITest journey against the production-shaped app.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT="${RAPID_XCUI_PROJECT:-$ROOT/Tests/RapidUITests/RapidUITests.xcodeproj}"
APP="${RAPID_XCUI_APP:-$ROOT/build/Rapid-MLX Desktop.app}"
RESULT_BUNDLE="${RAPID_XCUI_RESULT_BUNDLE:-$ROOT/build/RapidUITests-$(date +%s)-$$.xcresult}"
DERIVED_DATA="${RAPID_XCUI_DERIVED_DATA:-${RESULT_BUNDLE%.xcresult}-DerivedData}"
ONLY_TESTING="${RAPID_XCUI_ONLY_TESTING:-}"
XCODEBUILD="${RAPID_XCUI_XCODEBUILD:-xcodebuild}"
STARTUP_TIMEOUT="${RAPID_XCUI_STARTUP_TIMEOUT_SECONDS:-90}"
TERMINATION_GRACE="${RAPID_XCUI_TERMINATION_GRACE_SECONDS:-10}"
STARTUP_SENTINEL="${RAPID_XCUI_STARTUP_SENTINEL:-$RESULT_BUNDLE.startup-ready}"
DIAGNOSTICS_DIR="${RAPID_XCUI_DIAGNOSTICS_DIR:-$RESULT_BUNDLE.startup-diagnostics}"

case "$STARTUP_TIMEOUT:$TERMINATION_GRACE" in
    *[!0-9:]*|:*|*:)
        echo "error: XCUITest startup and termination timeouts must be whole seconds" >&2
        exit 2
        ;;
esac
(( STARTUP_TIMEOUT > 0 )) || {
    echo "error: RAPID_XCUI_STARTUP_TIMEOUT_SECONDS must be greater than zero" >&2
    exit 2
}

[[ -d "$APP" ]] || { echo "error: build the app first: $APP" >&2; exit 1; }
"$XCODEBUILD" -version >/dev/null 2>&1 || {
    echo "error: full Xcode is required for XCUITest (Command Line Tools are insufficient)" >&2
    exit 1
}
[[ -d "$PROJECT" ]] || {
    echo "error: generated Xcode project missing: $PROJECT" >&2
    exit 1
}

# XCUIApplication(bundleIdentifier:) resolves through LaunchServices.
LSREGISTER="${RAPID_XCUI_LSREGISTER:-/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister}"
"$LSREGISTER" -f "$APP"

# A native UI test requires both Developer Mode and an interactive Aqua
# session. These checks turn two otherwise opaque pre-launch hangs into an
# immediate, operator-actionable failure on local/self-hosted runners.
if [[ "$(uname -s)" == "Darwin" && "${RAPID_XCUI_SKIP_HOST_PREFLIGHT:-0}" != "1" ]]; then
    developer_status="$(/usr/sbin/DevToolsSecurity -status 2>&1 || true)"
    if [[ "$developer_status" != *"enabled"* ]]; then
        echo "error: XCUITest requires Developer Mode; run 'sudo DevToolsSecurity -enable' on this host" >&2
        echo "DevToolsSecurity: $developer_status" >&2
        exit 1
    fi
    if ! launchctl print "gui/$(id -u)" >/dev/null 2>&1; then
        echo "error: XCUITest requires a logged-in GUI session for uid $(id -u)" >&2
        exit 1
    fi
    automation_status="$(/usr/bin/automationmodetool 2>&1 || true)"
    if [[ "$automation_status" == *"requires user authentication"* ]]; then
        echo "error: this CI host requires interactive authentication before every XCUITest session" >&2
        echo "error: run 'sudo automationmodetool enable-automationmode-without-authentication' once on the host" >&2
        echo "$automation_status" >&2
        exit 1
    fi
fi

test_selection=()
if [[ -n "$ONLY_TESTING" ]]; then
    test_selection+=("-only-testing:$ONLY_TESTING")
fi

xcode_options=(
    -project "$PROJECT"
    -scheme RapidUITests
    -destination 'platform=macOS'
    -derivedDataPath "$DERIVED_DATA"
    -parallel-testing-enabled NO
)
signing_options=(
    CODE_SIGN_STYLE=Manual
    CODE_SIGNING_ALLOWED=YES
    CODE_SIGNING_REQUIRED=YES
    CODE_SIGN_IDENTITY=-
    DEVELOPMENT_TEAM=
)

# Keep compilation outside the startup timer. The issue is Xcode's local test
# worker/launcher materialization after a successful build, not slow Swift
# compilation. test-without-building then exercises exactly that bounded phase.
build_command=("$XCODEBUILD" build-for-testing "${xcode_options[@]}")
if (( ${#test_selection[@]} )); then
    build_command+=("${test_selection[@]}")
fi
build_command+=("${signing_options[@]}")
"${build_command[@]}"

rm -f "$STARTUP_SENTINEL"
mkdir -p "$DIAGNOSTICS_DIR"
export RAPID_XCUI_STARTUP_SENTINEL="$STARTUP_SENTINEL"
test_log="$DIAGNOSTICS_DIR/xcodebuild-test.log"

test_command=(
    "$XCODEBUILD" test-without-building
    "${xcode_options[@]}"
    -resultBundlePath "$RESULT_BUNDLE"
)
if (( ${#test_selection[@]} )); then
    test_command+=("${test_selection[@]}")
fi
test_command+=("${signing_options[@]}")
"${test_command[@]}" > >(tee "$test_log") 2>&1 &
xcode_pid=$!

stop_owned_xcodebuild() {
    local pid="$1"
    kill -INT "$pid" 2>/dev/null || true
    local deadline=$((SECONDS + TERMINATION_GRACE))
    while kill -0 "$pid" 2>/dev/null && (( SECONDS < deadline )); do
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
        sleep 1
    fi
    if kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$pid" 2>/dev/null || true
    fi
}

cleanup_on_signal() {
    stop_owned_xcodebuild "$xcode_pid"
    exit 130
}
trap cleanup_on_signal INT TERM

startup_deadline=$((SECONDS + STARTUP_TIMEOUT))
while kill -0 "$xcode_pid" 2>/dev/null; do
    if [[ -s "$STARTUP_SENTINEL" ]]; then
        break
    fi
    if (( SECONDS >= startup_deadline )); then
        echo "::error::XCUITest target app did not finish launching within ${STARTUP_TIMEOUT}s" >&2
        echo "error: Xcode's UI-test worker/launcher is stalled before the Desktop journey started" >&2
        echo "result bundle: $RESULT_BUNDLE" >&2
        echo "startup diagnostics: $DIAGNOSTICS_DIR" >&2
        # Do not dump the host-wide process table into a public CI artifact:
        # unrelated command lines may contain credentials. Keep only our
        # xcodebuild, its DerivedData-launched test processes, and the target
        # app path.
        {
            echo "owned xcodebuild"
            ps -p "$xcode_pid" -o pid=,ppid=,state=,etime=,command=
            echo "processes launched from this DerivedData"
            ps -axo pid=,ppid=,state=,etime=,command= | \
                awk -v root="$DERIVED_DATA" 'index($0, root)'
            echo "target app processes"
            ps -axo pid=,ppid=,state=,etime=,command= | \
                awk -v root="$APP/Contents/MacOS/" 'index($0, root)'
        } > "$DIAGNOSTICS_DIR/process-tree.txt" 2>&1 || true
        if [[ "${RAPID_XCUI_CAPTURE_DIAGNOSTICS:-1}" == "1" ]] && \
           command -v sample >/dev/null 2>&1; then
            sample "$xcode_pid" 3 1 \
                -file "$DIAGNOSTICS_DIR/xcodebuild.sample.txt" >/dev/null 2>&1 || true
        fi
        if [[ "${RAPID_XCUI_CAPTURE_DIAGNOSTICS:-1}" == "1" ]] && \
           command -v log >/dev/null 2>&1; then
            log show --last 3m --style compact \
                --predicate 'process == "xcodebuild" OR process == "xctest" OR process == "testmanagerd"' \
                > "$DIAGNOSTICS_DIR/unified.log" 2>&1 || true
        fi
        stop_owned_xcodebuild "$xcode_pid"
        wait "$xcode_pid" 2>/dev/null || true
        trap - INT TERM
        exit 124
    fi
    sleep 1
done

set +e
wait "$xcode_pid"
status=$?
set -e
trap - INT TERM

if [[ ! -s "$STARTUP_SENTINEL" && "$status" -eq 0 ]]; then
    echo "error: XCUITest completed without proving that the target app launched" >&2
    exit 1
fi
exit "$status"
