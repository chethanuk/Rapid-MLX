# SPDX-License-Identifier: Apache-2.0
"""GUI journeys prove control *outcomes*, not just button presses.

These contract tests replaced brittle source-string greps that re-read the
harness and CI workflow and asserted on exact error-message copy. Renaming a
display string or rewording a `die` message used to fail the suite for no
behavioural reason (#2494). Each guard below now asserts the structural
*behaviour* that actually matters:

* a journey performs an action and then evaluates a post-condition on the
  resulting UI/fake-event state (counts of outcome assertions vs. presses);
* the fake-event and fixture contracts the journey depends on are intact
  (`pull` lifecycle, machine aliases, watchdog events);
* the flow is gated in CI and its failures leave usable evidence (parsed
  structurally from the workflow YAML, mirroring ``test_gui_golden_ci_coverage``);
* the only deliberately retained single-anchor source checks are the two sides
  of a cross-language request-body contract (Swift default <-> shell request),
  where no behavioural proxy exists — see ``test_image_generation_...``.

The guarantees here are behaviour; the anchors are precise and loud about why
they exist when they break.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "apps/rapid-mac/scripts/gui-golden-flows.sh"
WORKFLOW = ROOT / ".github/workflows/rapid-mac-ci.yml"
IMAGE_VIEW_MODEL = ROOT / "apps/rapid-mac/Sources/Rapid/Images/ImageGenViewModel.swift"
SNAPSHOTS = ROOT / "apps/rapid-mac/Tests/GUIGoldenFlows/__Snapshots__"


def _harness_flow_body(flow_function: str) -> str:
    """Return one named ``flow_*`` function's body from the golden harness.

    Splitting on the function signature and the first closing brace is stable
    across copy edits inside the body; only the function's name and structure
    are load-bearing.
    """
    source = HARNESS.read_text()
    return source.split(f"{flow_function}() {{", 1)[1].split("\n}", 1)[0]


def _assertion_count(body: str) -> int:
    """Outcome-assertion helper calls: `die`, `wait_fake_event`, `wait_identifier`."""
    return (
        body.count("die ")
        + body.count("wait_fake_event")
        + body.count("wait_identifier")
    )


def _action_count(body: str) -> int:
    """Gesture helper calls that drive a control: `press`, `set-value`, `increment`, `decrement`."""
    return (
        body.count('press "$OUT/')
        + body.count('"$AX_DRIVER" set-value')
        + body.count('"$AX_DRIVER" increment')
        + body.count('"$AX_DRIVER" decrement')
    )


def _golden_flow_steps() -> list[dict[str, Any]]:
    """All 'Golden flow: <name>' steps in the GUI CI job, parsed structurally."""
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["gui-golden-flows"]["steps"]
    return [
        step for step in steps if str(step.get("name", "")).startswith("Golden flow:")
    ]


def _golden_flow_step(name: str) -> dict[str, Any]:
    return next(
        step
        for step in _golden_flow_steps()
        if step.get("name") == f"Golden flow: {name}"
    )


def _diagnostic_flow_list() -> list[str]:
    """Names in the 'Regenerate baselines' step's ``for flow in ...`` list."""
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["gui-golden-flows"]["steps"]
    (diagnostic,) = [
        step
        for step in steps
        if step.get("name") == "Regenerate baselines on this runner (diagnostic)"
    ]
    run = str(diagnostic.get("run", ""))
    return run.split("for flow in ", 1)[1].split("; do", 1)[0].split()


def _owns_committed_baseline(flow: str) -> bool:
    return any(p.name.startswith(flow) for p in SNAPSHOTS.glob("*.txt"))


def test_audio_readiness_asserts_outcomes_not_just_presses():
    """The audio-readiness journey proves model lifecycle outcomes.

    Every gesture (press / set-value) is followed by an outcome assertion
    (`die`/`wait_fake_event`/`wait_identifier`) that verifies the resulting
    state, and the total number of outcome assertions dwarfs the number of
    gestures. A journey that merely pressed buttons and read one final tree
    would fail here.
    """
    flow = _harness_flow_body("flow_audio_readiness")

    # Proves control outcomes, not just presses: assertions > gestures.
    assert _assertion_count(flow) > _action_count(flow)
    # The lifecycle is two real downloads (Download -> Start) for model start,
    # keyed on the fake's recorded `pull` command event.
    assert flow.count('.subcommand == "pull"') == 2
    # The journey watches both a TTS and a transcription model start/stop.
    assert '.alias == "fake-qwen3-tts"' in flow
    assert '.alias == "fake-whisper-small"' in flow


def test_audio_readiness_never_auto_starts_a_model():
    """Download-only actions must not start a model; only Start may.

    Previously pinned by grepping five exact `die` message strings. The
    behavioural contract is structural: the flow contains a negative guard
    (die when a `server_started` fake-event is observed prematurely) for
    EACH model path — the Speech/TTS alias and the Dictation/transcription
    alias — and a positive `wait_fake_event` that asserts `server_started`
    only after the explicit Start gesture.
    """
    flow = _harness_flow_body("flow_audio_readiness")

    # Negative auto-load guards: `die` when a model started without an action.
    # Both the Speech/TTS and the Dictation/transcription model paths must
    # refuse premature `server_started`, so dropping either path fails here.
    for alias in ("fake-qwen3-tts", "fake-whisper-small"):
        assert (
            f'any(.[]; .event == "server_started" and .alias == "{alias}")' in flow
        ), f"audio-readiness no longer refutes premature auto-start of {alias}"
    # The one allowed start is asserted as a waited post-condition after Start.
    assert '"server_started" and .alias == "fake-qwen3-tts"' in flow


def test_audio_control_journey_is_blocking_gui_ci_and_has_failure_evidence():
    """audio-readiness is CI-gated, and a baseline failure regenerates it."""
    step = _golden_flow_step("audio-readiness")
    assert "--flow audio-readiness" in str(step.get("run", ""))
    assert step.get("env", {}).get("RAPID_GUI_GOLDEN_OUT") == (
        "${{ runner.temp }}/golden/audio-readiness"
    )
    # audio-readiness owns a committed baseline, so regeneration is the right
    # failure evidence for it (and image-generation, the other new-Images
    # journey, is regenerated alongside in the same diagnostic pass).
    assert _owns_committed_baseline("audio-readiness")
    diagnostic = _diagnostic_flow_list()
    assert "audio-readiness" in diagnostic
    assert "image-generation" in diagnostic


def test_dictation_journey_proves_loading_before_ready():
    """The dictation journey asserts a cold-loading phase, then Listening.

    The fixture keeps a fake STT probe open long enough to observe both
    state transitions. Previously the two `die` message strings were pinned;
    the stable guarantee is that the flow asserts *two* independent
    `Dictation.Status` outcome predicates — a Loading phase and a
    Listening/ready phase — via the shared ``.description // .value // .label``
    status filter, plus the behavioural warmup probe.
    """
    flow = _harness_flow_body("flow_dictation")

    assert "RAPID_GUI_DICTATION_READINESS_FIXTURE=1" in flow
    assert "FAKE_AUDIO_TRANSCRIPTION_DELAY_MS=1800" in flow
    # Warmup probe is waited on as a fake event.
    assert '.event == "audio_transcription"' in flow
    # The status filter used to read readiness text off a control description.
    assert '(.description // .value // .label // "")' in flow
    # Two distinct outcome-state predicates on Dictation.Status: the loading
    # phase (before readiness) and the Listening phase (after warmup).
    assert flow.count('select(.identifier == "Dictation.Status"') == 2


def test_image_generation_shell_request_matches_the_swift_default():
    """Keep the shell E2E contract aligned with the Swift default.

    The view-model test catches a wrong UI default, while the golden journey
    catches a wrong request body. Pinning both sides here prevents changing
    one literal and leaving the other to fail only in the 20-minute GUI job.
    These two single-anchor source checks are the deliberate exception to the
    behaviour-test rule: a cross-language request-body default has no
    behavioural proxy from Python, so we keep the precise anchors that make a
    rename or drift fail loudly and explain why.
    """
    view_model = IMAGE_VIEW_MODEL.read_text()
    flow = _harness_flow_body("flow_image_generation")

    assert "var resolution: Resolution = .compact" in view_model, (
        "ImageGenViewModel default drift: expected the compact (square) resolution every new canvas starts on"
    )
    assert '.size == "512x512"' in flow, (
        "image-generation journey no longer requests the 512x512 default — it must match ImageGenViewModel's .compact default (see var resolution: Resolution = .compact)"
    )


SNAP_AUDIT_FLOWS = [
    "no-dead-controls",
    "catalog-integrity",
    "update-state",
    "launch-integrations",
]
BASELINED_AUDIT_FLOWS = ["update-state", "launch-integrations"]
# Semantic audits carry no committed AX snapshots; their failure evidence is
# the flow's own output directory, which must sit inside the artifact uploaded
# on failure.
SNAPSHOT_LESS_AUDIT_FLOWS = ["no-dead-controls", "catalog-integrity"]


@pytest.mark.parametrize("flow", SNAP_AUDIT_FLOWS)
def test_semantic_control_audits_are_blocking_gui_ci(flow: str):
    """Each semantic audit is gated, and its failure leaves usable evidence.

    Parsed structurally from the workflow YAML. What "evidence" means depends
    on whether the flow owns committed AX baselines: update-state and
    launch-integrations do, so they belong in the regenerate-on-failure
    diagnostic loop; no-dead-controls and catalog-integrity carry no
    snapshots, so their evidence is the per-flow output directory that the
    upload-on-failure step ships.
    """
    step = _golden_flow_step(flow)
    assert f"--flow {flow}" in str(step.get("run", ""))

    if _owns_committed_baseline(flow):
        assert flow in _diagnostic_flow_list()
        # Fix the guards above if these preconditions become stale.
        assert flow in BASELINED_AUDIT_FLOWS
    else:
        assert flow in SNAPSHOT_LESS_AUDIT_FLOWS
        assert step.get("env", {}).get("RAPID_GUI_GOLDEN_OUT") == (
            f"${{{{ runner.temp }}}}/golden/{flow}"
        )
        upload = _upload_ax_evidence_step()
        assert upload.get("if") == "failure()"
        paths = {
            ln.strip()
            for ln in str(upload.get("with", {}).get("path", "")).splitlines()
            if ln.strip()
        }
        assert "${{ runner.temp }}/golden" in paths


def _upload_ax_evidence_step() -> dict[str, Any]:
    steps: list[dict[str, Any]] = yaml.safe_load(WORKFLOW.read_text())["jobs"][
        "gui-golden-flows"
    ]["steps"]
    (upload,) = [step for step in steps if step.get("name") == "Upload AX evidence"]
    return upload
