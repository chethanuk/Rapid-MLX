# SPDX-License-Identifier: Apache-2.0
"""The engine and the Desktop app must agree on serving-lane reason strings.

The engine reports *why* a model is serving text-only in
``serving_lane_reason``; the Desktop app turns that into the sentence a user
reads when the photo button is disabled. Nothing connects the two sides — the
engine emits bare string literals, the app matches them in a Swift ``switch`` —
so they drifted:

* the app matched ``operator_forced_text``, which the engine has never emitted
  (it emits ``text_lane_forced``), leaving that copy permanently unreachable;
* ``vision_memory_insufficient`` had no case at all, so a vision-capable model
  that merely did not fit in memory told the user to "choose a vision-capable
  model" — a remedy they had already applied.

Both survived because the Swift fixtures asserting this area used the ghost
strings too, so the tests agreed with the app instead of with the engine.

The engine now has a single source of truth: ``SERVING_LANE_REASONS`` /
``AUTO_TEXT_FALLBACK_REASONS`` in ``vllm_mlx/api/utils.py``, enforced at
construction time by ``ServingLaneDecision.__post_init__``. This test imports
that SSOT and, rather than trusting it to be right, re-scans the whole engine
tree for the literals actually emitted and proves the SSOT is both complete (no
emitted reason can fall outside it) and exactly matches the Swift ``case``
labels the user actually reads.

mlx-free: the ``vllm_mlx.api.utils`` import chain pulls in no MLX (verified —
the Linux CI leg runs this with no MLX installed), and reason collection is
pure text parsing.
"""

from __future__ import annotations

import re
from pathlib import Path

from vllm_mlx.api.utils import (
    AUTO_TEXT_FALLBACK_REASONS,
    SERVING_LANE_REASONS,
)

REPO = Path(__file__).resolve().parents[1]

# Scan the whole engine tree rather than naming files, so a reason introduced
# in a fourth module (or a literal seeded outside the decision function) cannot
# slip past this contract.
ENGINE = REPO / "vllm_mlx"

PROFILE_SWIFT = REPO / "apps/rapid-mac/Sources/Rapid/Server/ServerModelProfile.swift"

# ``ServingLaneDecision(False, "x", auto_text_fallback=True)`` — tolerant of
# line breaks and spacing, strict about the keyword actually being True.
AUTO_FALLBACK_RE = re.compile(r"auto_text_fallback\s*=\s*True")


def _balanced_call(text: str, open_paren: int) -> str:
    """Source of the call whose ``(`` sits at ``open_paren``."""
    depth = 0
    for i in range(open_paren, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren : i + 1]
    raise AssertionError(f"unbalanced ServingLaneDecision call at offset {open_paren}")


def _decision_reasons() -> dict[str, bool]:
    """``{reason: auto_text_fallback}`` for every ``ServingLaneDecision`` in the tree."""
    found: dict[str, bool] = {}
    for path in ENGINE.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"ServingLaneDecision\(", text):
            call = _balanced_call(text, match.end() - 1)
            reason = re.search(r'"([^"]+)"', call)
            assert reason, f"ServingLaneDecision without a literal reason: {call!r}"
            auto_fallback = bool(AUTO_FALLBACK_RE.search(call))
            # A reason emitted from several call sites is auto-fallback if ANY of
            # them sets the flag — the user can reach the copy through that path.
            found[reason.group(1)] = found.get(reason.group(1), False) or auto_fallback
    assert found, "no ServingLaneDecision constructions found — parser is stale"
    return found


def _literal_assignments(attribute: str) -> set[str]:
    """Reason strings assigned to ``attribute`` anywhere in the engine tree."""
    pattern = re.compile(rf'{re.escape(attribute)}\s*=\s*"([^"]+)"')
    found: set[str] = set()
    for path in ENGINE.rglob("*.py"):
        found.update(pattern.findall(path.read_text(encoding="utf-8")))
    return found


def _engine_reasons() -> dict[str, bool]:
    """``{reason: auto_text_fallback}`` across the whole engine tree.

    The flag answers "can a vision-capable model reach the user with this
    reason while serving text-only", which is what makes the generic copy
    wrong. ``ServingLaneDecision`` records that as ``auto_text_fallback``;
    assignments outside it have to be classified here.
    """
    reasons = _decision_reasons()
    # Plain assignments seed the field — startup does this for image-gen /
    # video-gen modalities and passes it straight through. Those models have
    # no vision lane to lose, so the generic copy is not misleading for them.
    for reason in _literal_assignments("serving_lane_reason"):
        reasons.setdefault(reason, False)
    # An assignment to the instance attribute is the engine downgrading
    # itself mid-load: after a failed vision load its own warning calls this
    # "auto-falling back to text-only serving" before starting the text lane.
    # Same silent downgrade the flag marks elsewhere, but it never passes
    # through ServingLaneDecision, so there is no flag to read. Runs after the
    # loop above, whose broader pattern also matches these lines.
    for reason in _literal_assignments("self._serving_lane_reason"):
        reasons[reason] = True
    return reasons


def _swift_case_labels() -> set[str]:
    """Every string matched by ``ImageInputAvailability.message(for:)``."""
    text = PROFILE_SWIFT.read_text()
    marker = "private static func message(for laneReason: String?) -> String"
    start = text.find(marker)
    assert start != -1, f"message(for:) not found in {PROFILE_SWIFT} — parser is stale"

    depth, body_start = 0, text.index("{", start)
    body = ""
    for i in range(body_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                body = text[body_start : i + 1]
                break
    assert body, f"unbalanced message(for:) body in {PROFILE_SWIFT}"

    # Only quoted strings between `case` and its `:` are matched values; the
    # rest of the body is user-facing copy.
    labels: set[str] = set()
    for arm in re.finditer(r"case\s+((?:\"[^\"]+\"\s*,?\s*)+):", body):
        labels.update(re.findall(r'"([^"]+)"', arm.group(1)))
    assert labels, "no case labels parsed out of message(for:) — parser is stale"
    return labels


def test_ssot_is_the_exhaustive_set_of_emitted_reasons():
    """The SSOT must both contain every reason the engine emits and not be dead.

    This is the mechanism-level check behind the issue: a stray ``serving_lane_reason``
    literal introduced anywhere in the engine (now or later) fails the ``<=`` half, and
    a stale SSOT entry that the engine stopped emitting fails the ``>=`` half. Either
    half means engine and copy have been allowed to drift again.
    """
    emitted = set(_engine_reasons())
    assert set(SERVING_LANE_REASONS) == emitted, (
        f"SERVING_LANE_REASONS and the engine tree disagree:\n"
        f"  in SSOT but never emitted by the engine: "
        f"{sorted(set(SERVING_LANE_REASONS) - emitted) or 'none'}\n"
        f"  emitted by the engine but missing from SSOT: "
        f"{sorted(emitted - set(SERVING_LANE_REASONS)) or 'none'}\n"
        f"Emitted (from tree scan): {sorted(emitted)}"
    )


def test_auto_text_fallback_ssot_matches_the_tree():
    """``AUTO_TEXT_FALLBACK_REASONS`` must exactly describe the decision function's flag.

    ``auto_text_fallback=True`` is a field on ``ServingLaneDecision``, so its
    authoritative enumeration is the set of ``ServingLaneDecision`` call sites
    that pass the flag (compared against the imported SSOT). ``__post_init__``
    enforces membership, so a reason that gains or loses the flag without the
    SSOT following is a construction-time error — this test is the belt that
    shows both stayed in lockstep.

    (This is deliberately distinct from the wider "silent downgrade" copy set —
    ``vision_weights_unavailable`` is a mid-load rewrite that never passes
    through ``ServingLaneDecision``, so it is not an ``AUTO_TEXT_FALLBACK``
    member even though it still needs dedicated copy; see
    ``test_every_silent_downgrade_has_dedicated_copy``.)
    """
    decision_auto = {r for r, is_auto in _decision_reasons().items() if is_auto}
    assert decision_auto, "no auto_text_fallback decisions parsed — parser is stale"
    assert set(AUTO_TEXT_FALLBACK_REASONS) == decision_auto, (
        f"AUTO_TEXT_FALLBACK_REASONS and the ServingLaneDecision call sites disagree:\n"
        f"  flagged by SSOT but no decision call site uses it: "
        f"{sorted(set(AUTO_TEXT_FALLBACK_REASONS) - decision_auto) or 'none'}\n"
        f"  decision call site flags it but SSOT misses it: "
        f"{sorted(decision_auto - set(AUTO_TEXT_FALLBACK_REASONS)) or 'none'}"
    )


def test_desktop_matches_only_reasons_the_engine_emits():
    """No ghost cases.

    A ``case`` for a string the engine never sends is dead code that reads as
    handled. This is the exact shape of the ``operator_forced_text`` bug.
    """
    emitted = set(_engine_reasons())
    ghosts = _swift_case_labels() - emitted
    assert not ghosts, (
        f"ServerModelProfile.swift matches serving-lane reasons the engine "
        f"never emits: {sorted(ghosts)}. Either the engine renamed them (update "
        f"the Swift cases) or they were never real. Engine reasons: "
        f"{sorted(emitted)}"
    )


def test_every_silent_downgrade_has_dedicated_copy():
    """Silent downgrades must not fall through to the generic sentence.

    A silent downgrade means the checkpoint IS vision-capable but its vision
    lane was not admitted — recorded either as ``auto_text_fallback=True`` on a
    ``ServingLaneDecision`` or as the mid-load ``vision_weights_unavailable``
    rewrite (which never passes through the dataclass). The default arm tells
    the user to pick a vision-capable model, which is the model they are already
    running — so every silent downgrade needs copy naming the real cause. We use
    the full tree-derived downgrade set so both mechanisms are covered.
    """
    silent_downgrades = {r for r, is_auto in _engine_reasons().items() if is_auto}
    assert silent_downgrades, "no silent-downgrade reasons parsed — parser is stale"
    uncovered = silent_downgrades - _swift_case_labels()
    assert not uncovered, (
        f"these reasons downgrade a vision-capable model to text but have no "
        f"dedicated copy in ServerModelProfile.swift: {sorted(uncovered)}. They "
        f"currently render as 'Photos need a vision-capable model', which "
        f"names a remedy the user has already applied."
    )


def test_text_lane_forced_reason_has_dedicated_copy():
    """The deliberate text-lane pin still needs its own sentence.

    ``text_lane_forced`` is not an auto-fallback: the CLI reaches it through
    ``--no-mllm`` / ``--text-only``. The Desktop app exposes no such switch, so
    in-app it arrives only from an alias pinned ``is_text_only`` in the
    registry — a vision-config checkpoint served through the text lane. Either
    way the checkpoint looks vision-capable while photos are refused, which is
    exactly when the generic copy is at its most confusing.
    """
    assert "text_lane_forced" in SERVING_LANE_REASONS, (
        "the engine no longer emits text_lane_forced; update this test and the "
        "Swift case together"
    )
    assert "text_lane_forced" in _swift_case_labels(), (
        "ServerModelProfile.swift lost its text_lane_forced case — a "
        "registry-pinned text-only alias is back to the generic copy"
    )
