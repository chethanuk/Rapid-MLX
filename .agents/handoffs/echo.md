# Echo handoff

- Status: first Mac mini server + GUI dogfood pass complete; one suite-level
  timing flake and one automation-permission gap remain
- Active task: dogfood latest `origin/main` across the real server and Rapid-MLX
  Desktop GUI
- Owner: Echo
- Branch: `echo/mini-server-gui-dogfood-20260824`
- Host: Mac mini `Mac14,12`, Apple M2 Pro, 32 GB unified memory, macOS 26.5.2
- Source: `origin/main` at `c23f9cca`
- Worktree: `~/worktrees/rapid-mlx-echo-dogfood-20260824`
- Explicit non-goals: no release, deployment, tag, production install, or
  changes to the dirty `~/work/Rapid-MLX` checkout
- Coordination note: the required PR-start FYIs could not be sent because the
  current coordinator was not attached to a stable Orca pane; this handoff
  carries the same scope and evidence for Atlas, Pixel, Vector, and Harbor.
- Verified facts:
  - The main mini checkout was 578 commits behind and carried unrelated
    uncommitted engine/MTP work. `origin` was fetched and a clean task worktree
    was created instead of pulling across that state.
  - A clean Python 3.12.13 environment resolved MLX 0.32.1, mlx-lm 0.31.3,
    transformers 5.12.1, and mlx-vlm 0.6.3.
  - A real `qwen3.5-4b-4bit` server on `127.0.0.1:8124` passed health/models,
    non-streaming Chat Completions, Responses, SSE with the usage trailer,
    a forced `weather` tool call, four concurrent sentinel requests, client
    disconnect accounting, post-cancel recovery, and graceful shutdown.
  - The full Desktop build completed with its real bundled sidecar. Strict deep
    codesign verification passed. `release-smoke.sh` passed after a 15-second
    life check with a 1200x820 main window.
  - A repository-provided isolated dogfood persona used bundle ID
    `com.rapidmlx.rapid.dogfood-3f875862`, throwaway HOME under `/private/tmp`,
    and port 55394. Telemetry decline and onboarding skip were reachable by
    semantic AX identifiers.
  - The GUI started the bundled sidecar with cached `lfm2.5-1b-4bit`, sent two
    real chat turns, displayed replies and performance captions, exposed Launch
    commands and Settings categories, and kept the conversation after a clean
    restart.
  - The GUI switched from the resident text model to cached
    `flux2-klein-4b`, loaded the image runtime, and generated one 512x512 starter
    image. The result exposed the stage, selected gallery thumbnail, Edit, and
    Download actions. This single cold-path observation took roughly three
    minutes and is not a benchmark claim.
  - Both clean quits reaped the sidecar and left zero crash markers.
  - `scripts/smoke.sh` passed. The full `swift test` run executed 2,763 tests;
    three `Declined tool diagnosis` approval tests failed with
    `ApprovalNeverArrived()`. The focused 26-test suite passed immediately on
    rerun, so the evidence currently indicates full-suite timing sensitivity,
    not a deterministic product regression.
- Environment limitation:
  - Accessibility permission is granted on the mini, but Screen Recording is
    not. The stock `gui-ax-smoke.sh` therefore cannot pass its permission gate.
    The first pass used direct macOS AX APIs without screenshots. Do not count
    screenshots or the complete Peekaboo smoke as covered.
- Non-blocking observations:
  - Settings showed a global storage summary of `78.5 GB · 20 models` while the
    current text-model footer showed `70.25 GB across 16 models`. This may be an
    all-capability versus current-filter distinction, but the labels do not make
    the scope difference obvious.
  - Build/test output repeats the existing unhandled snapshot-resource warning
    and an unused `baseURL` warning in `MenuBarStatus.swift`.
- Risks:
  - The image timing is one cold observation with AX polling overhead; route it
    to Vector only if repeatable measurement is desired.
  - The full-suite approval failures should be treated as a Pixel-owned test
    reliability follow-up if they reproduce again under full concurrency.
- Receiving roles:
  - Pixel: assess the `Declined tool diagnosis` full-suite timing sensitivity
    and the model-storage scope copy.
  - Vector: only if a controlled image cold/warm timing run is requested.
  - Harbor: enable Screen Recording for the mini automation account or document
    that the canonical Peekaboo GUI gate must run on another authorized host.
  - Atlas: no blocker found in the tested slice; retain normal release ownership.
- Next action: rerun the complete Peekaboo smoke on a Screen Recording-enabled
  host, then repeat the full Swift suite once to decide whether the three
  approval tests need a scoped Pixel fix.
