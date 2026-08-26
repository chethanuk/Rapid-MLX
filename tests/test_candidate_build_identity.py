from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "rapid-mac-release.yml"
ACTION = ROOT / ".github" / "actions" / "desktop-releasable" / "action.yml"
BUILD = ROOT / "apps" / "rapid-mac" / "scripts" / "build.sh"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def test_dispatch_candidate_identity_is_sha_bound_and_separate_from_versions() -> None:
    text = WORKFLOW.read_text()
    action = yaml.safe_load(ACTION.read_text())

    assert 'CANDIDATE_IDENTITY="candidate-${GITHUB_SHA:0:8}"' in text
    assert 'CANDIDATE_IDENTITY=""' in text
    assert "candidate_identity" in action["inputs"]
    assert action["inputs"]["candidate_identity"]["default"] == ""
    assert "RAPID_CANDIDATE_IDENTITY" in ACTION.read_text()


def test_candidate_tester_dmg_is_additive_and_dispatch_only() -> None:
    jobs = _workflow()["jobs"]
    build = jobs["build"]
    by_name = {step.get("name"): step for step in build["steps"]}

    canonical = by_name["Upload workflow artifact (DMG + manifest)"]
    stage = by_name["Stage candidate-labelled tester DMG"]
    candidate = by_name["Upload candidate-labelled tester DMG"]

    assert "rapid-mlx-desktop.dmg" in canonical["with"]["path"]
    assert stage["if"] == "steps.appmeta.outputs.is_tag != 'true'"
    assert candidate["if"] == "steps.appmeta.outputs.is_tag != 'true'"
    assert "rapid-mlx-desktop-${CANDIDATE_IDENTITY}.dmg" in stage["run"]
    assert 'cmp -s "$SOURCE" "$TARGET"' in stage["run"]
    assert "candidate_identity" in candidate["with"]["name"]


def test_build_script_validates_and_embeds_separate_candidate_key() -> None:
    text = BUILD.read_text()

    assert "^candidate-[0-9a-f]{8}$" in text
    assert "plutil -insert RapidCandidateIdentity" in text
    candidate_block = text[text.index('if [[ -n "${RAPID_CANDIDATE_IDENTITY') :]
    assert "CFBundleVersion" not in candidate_block.split("fi", 1)[0]
    assert "CFBundleShortVersionString" not in candidate_block.split("fi", 1)[0]
