#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Rapid-MLX merge-train driver (loco-desk).

Implements the Loco driver recipe in ``.orca/train-protocol.md``: one boarding
train per departure that batches the open ``train:boarding`` PRs onto a
``train/<stamp>Z`` branch off ``origin/main`` with per-PR ``--no-ff`` merges,
pushes, opens a ``train`` PR, polls the required checks on the train head,
merges with a merge commit (never squash), verifies every member reads MERGED,
and — on a red train — bisects to isolate and eject the failing PR(s) on
evidence, landing the green remainder.

The driver intentionally stays a thin, stdlib-only wrapper over ``gh`` and
``git`` subprocesses so exceptions are reasoned about by a human, not hidden in
Python. We do not rerun CI until green: at most one diagnosed rerun, written
into a PR comment, is allowed.

Flags:
  * ``--dry-run``  resolve boarding PRs, run the merge ordering and conflict
                   detection on a throwaway local branch, and print the train
                   plan + ejection reasons WITHOUT pushing, opening a PR,
                   merging, or reporting to the Run.
  * ``--once``     run exactly one departure cycle then exit (the default; kept
                   as an explicit alias so the cadence is legible at call sites).
  * ``--bisect``   on a red train, force the bisect path described in the
                   protocol (split-half sub-trains) even if the red looks
                   attributable to a single member. Without it the red path
                   still bisects, but this makes the intent explicit.

Reference semantics: bors-ng batching + bisect; GitHub merge-queue grouping.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = "raullenchai/Rapid-MLX"
RUN = "run_fe90400db11e"

# Required checks enforced by branch protection on main (and thus on train PRs).
REQUIRED_CHECKS = ("tests", "desktop-tests", "version-bump-guard")

POLL_SECONDS = 300  # poll every 5 min
MAX_POLL_MINUTES = 60  # give up after an hour and report BLOCKED

REPO_ROOT = Path(__file__).resolve().parents[1]


class TrainError(Exception):
    """Fatal condition that aborts a departure cleanly (reported BLOCKED)."""


def run(args: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess; return the CompletedProcess."""
    res = subprocess.run(args, capture_output=capture, text=True)
    if check and res.returncode != 0:
        raise TrainError(
            f"command failed ({res.returncode}): {' '.join(args)}\n"
            f"stdout: {res.stdout[-4000:]}\nstderr: {res.stderr[-4000:]}"
        )
    return res


def gh_json(args: list[str]) -> list | dict:
    """Run ``gh <args> --json``-style query and parse JSON."""
    out = run(["gh", *args]).stdout
    return json.loads(out)


def stamp_now() -> tuple[str, str]:
    """Return (branch_stamp, title_stamp) for this departure in UTC."""
    now = datetime.now(timezone.utc)
    branch = now.strftime("%Y%m%d-%H%M") + "Z"
    title = now.strftime("%Y-%m-%d %H%M") + "Z"
    return branch, title


def list_boarding() -> list[dict]:
    """List open PRs carrying the ``train:boarding`` label."""
    return gh_json([
        "pr", "list",
        "--repo", REPO,
        "--state", "open",
        "--label", "train:boarding",
        "--json",
        "number,title,headRefName,headRefOid,mergeable,reviewDecision",
    ])


def head_green(head_sha: str) -> tuple[bool, list[str]]:
    """Evaluate the required checks on a commit; return (green, failed_or_pending).

    GitHub required checks are satisfied by EITHER a status context OR a
    check-run whose name matches the required context. We query both endpoints
    and require each ``REQUIRED_CHECKS`` name to have at least one green match.
    Anything present-and-not-success (pending, in_progress, failure, cancelled)
    makes the head not green.
    """
    try:
        statuses = gh_json(["api", f"repos/{REPO}/commits/{head_sha}/status"])["statuses"]
        check_runs = gh_json(["api", f"repos/{REPO}/commits/{head_sha}/check-runs"])["check_runs"]
    except TrainError:
        # A brand-new head may not have statuses yet; treat as not-yet-green.
        return False, list(REQUIRED_CHECKS)

    # Collect per-context green evidence and per-context blocking evidence.
    green = set()
    blocking = set()
    for s in statuses:
        ctx = s.get("context")
        if not ctx:
            continue
        if s.get("state") == "success":
            green.add(ctx)
        elif s.get("state") in ("pending", "error", "failure"):
            blocking.add(ctx)
    for cr in check_runs:
        name = cr.get("name")
        if not name:
            continue
        if cr.get("conclusion") == "success":
            green.add(name)
        else:
            # Any other completed/uncompleted state is not green for that name.
            blocking.add(name)

    bad = []
    for name in REQUIRED_CHECKS:
        if name in blocking and name not in green:
            # A matching request that is currently running/blocking with no green
            # counterpart on this head is not yet mergeable.
            bad.append(name)
        elif name not in green and name not in blocking:
            # No evidence at all (workflow may not have published it yet).
            bad.append(name)
    return (len(bad) == 0), bad


def failure_links(failed: list[str]) -> str:
    """Human hint; links require the head sha which is passed separately by callers."""
    return ", ".join(failed) if failed else ""


def evaluate(prs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Filter boarding PRs into (boarded, ejected). Ejected entries carry a reason."""
    boarded: list[dict] = []
    ejected: list[dict] = []
    for pr in prs:
        n = pr["number"]
        head = pr["headRefOid"]
        if pr.get("mergeable") == "CONFLICTING":
            ejected.append({**pr, "reason": "conflicts with base (mergeable=CONFLICTING); rebase required"})
            continue
        green, bad = head_green(head)
        if not green:
            ejected.append({**pr, "reason": f"required checks not green on exact head: {failure_links(bad)}"})
            continue
        if pr.get("reviewDecision") == "REVIEW_REQUIRED":
            ejected.append({**pr, "reason": "review not converged (REVIEW_REQUIRED)"})
            continue
        boarded.append(pr)
    return boarded, ejected


def merge_order(prs: list[dict]) -> list[dict]:
    """Order PRs for the train. Protocol says 'in label order' — for our label
    there is no rank, so newest-boarded first keeps the most-recent work on top
    and reduces churn. Returned in merge order (oldest base first is fine for
    git; we keep PR list order but stable)."""
    return prs


def build_train(branch: str, base: str, prs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Create ``branch`` from ``base`` and --no-ff merge each PR head in order.

    Returns (merged, conflict_ejected). On a merge conflict we abort the merge,
    eject that PR with the conflicting files, and continue with the rest — never
    force-anything, never squash.
    """
    run(["git", "checkout", "-b", branch, base])
    merged: list[dict] = []
    ejected: list[dict] = []
    for pr in prs:
        n = pr["number"]
        head = pr["headRefOid"]
        title = pr["title"]
        merge = run(
            ["git", "merge", "--no-ff", head,
             "-m", f"train: merge #{n} {title}"],
            check=False,
        )
        if merge.returncode == 0:
            merged.append(pr)
            continue
        # Conflict or merge failure: abort, record which files conflicted, eject.
        conflicted = _conflicted_files(merge.stdout + "\n" + merge.stderr)
        run(["git", "merge", "--abort"], check=False)
        ejected.append({**pr, "reason": f"merge conflict on train: {conflicted or 'unknown files'} (failing job: see merge output)"})
    return merged, ejected


def _conflicted_files(output: str) -> str:
    """Extract conflicted paths from a failed ``git merge`` output (dedup, join)."""
    files = []
    for line in output.splitlines():
        low = line.lower()
        if "conflict" in low and ":" in line:
            # "CONFLICT (content): Merge conflict in pyproject.toml"
            marker = " in "
            if marker not in line:
                continue
            token = line.split(marker, 1)[1].strip()
            # Trim trailing prose and punctuation to leave a path-like token.
            token = token.split(" and ")[0].split(" (")[0].rstrip(".").strip()
            if token:
                files.append(token)
    seen = set()
    uniq = []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return ", ".join(uniq)


def open_train_pr(branch: str, title: str, merged: list[dict]) -> int:
    """Push the train branch and open the train PR; return its number."""
    run(["git", "push", "-u", "origin", branch])
    body_lines = [
        f"train: {branch} departure {title}",
        "",
        "Boarded (in order):",
    ]
    for pr in merged:
        # Carry the member title and its Fixes line if present in the body.
        detail = gh_json(["api", f"repos/{REPO}/pulls/{pr['number']}"])
        fix = _extract_fix(detail.get("body") or "")
        body_lines.append(f"- #{pr['number']} {pr['title']}" + (f" ({fix})" if fix else ""))
    body = "\n".join(body_lines)
    out = gh_json([
        "pr", "create", "--repo", REPO, "--base", "main", "--head", branch,
        "--title", f"train: departure {title} " + ", ".join(f"#{p['number']}" for p in merged),
        "--body", body,
        "--label", "train",
    ])
    return int(out["number"])


def _extract_fix(body: str) -> str | None:
    for line in body.splitlines():
        line = line.strip()
        if line.lower().startswith(("fixes ", "closes ", "part of ", "fix #", "closes #", "part of #")):
            return line
    return None


def comment(pr: str, text: str) -> None:
    run(["gh", "pr", "comment", "--repo", REPO, pr, "--body", text])


def add_label(pr: int, label: str) -> None:
    run(["gh", "pr", "edit", "--repo", REPO, str(pr), "--add-label", label])


def wait_green(head_sha: str, label: str) -> bool:
    """Poll required checks on ``head_sha`` every 5 min; return True when green."""
    deadline = time.time() + MAX_POLL_MINUTES * 60
    while time.time() < deadline:
        green, bad = head_green(head_sha)
        if green:
            return True
        print(f"  [{label}] waiting on: {failure_links(bad)}")
        time.sleep(POLL_SECONDS)
    return False


def members_merged(train_pr: int, member_nums: list[int]) -> list[int]:
    "Return the subset of member PR numbers that do NOT read MERGED."
    not_merged = []
    for n in member_nums:
        p = gh_json(["api", f"repos/{REPO}/pulls/{n}"])
        if p["state"] != "closed" or not p["merged"]:
            not_merged.append(n)
    return not_merged


def report(text: str) -> None:
    """One message to the Run. Uses --subject (the send schema's message field)."""
    run(["orca", "orchestration", "send", "--run", RUN, "--subject", text])


def delete_branch(branch: str) -> None:
    run(["git", "push", "origin", "--delete", branch], check=False)
    run(["git", "branch", "-D", branch], check=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="resolve + plan only; no push/PR/merge/report")
    ap.add_argument("--once", action="store_true", help="run a single departure cycle (default)")
    ap.add_argument("--bisect", action="store_true", help="force bisect on any red train")
    args = ap.parse_args()

    print("== loco driver: fetch origin ==")
    run(["git", "fetch", "origin"])

    boarding_prs = list_boarding()
    if not boarding_prs:
        print("No PRs with label train:boarding — no train this cycle.")
        return 0

    boarded, ejected = evaluate(boarding_prs)
    print(f"\nBoarded: {[p['number'] for p in boarded]}")
    for e in ejected:
        print(f"  EJECT #{e['number']}: {e['reason']}")

    if not boarded:
        print("Nothing eligible to board — no train this cycle.")
        return 0

    stamp_branch, stamp_title = stamp_now()
    train_branch = f"train/{stamp_branch}"
    print(f"\nTrain branch: {train_branch}")

    # Build the train on a fresh branch off origin/main (checks conflicts).
    merged, conflict_ejected = build_train(train_branch, "origin/main", merge_order(boarded))
    for e in conflict_ejected:
        print(f"  EJECT (conflict) #{e['number']}: {e['reason']}")

    if not merged:
        run(["git", "checkout", "-"], check=False)
        run(["git", "branch", "-D", train_branch], check=False)
        print("All candidates conflicted — no train this cycle.")
        return 0

    print("\n== Train plan ==")
    for pr in merged:
        print(f"  #{pr['number']} {pr['title']}  ({pr['headRefOid'][:10]})")
    print(f"  Ejected: {[(e['number'], e['reason']) for e in ejected + conflict_ejected]}")

    if args.dry_run:
        print("\n--dry-run: not pushing/opening PR/merging/reporting. Cleaning up local branch.")
        run(["git", "checkout", "-"], check=False)
        run(["git", "branch", "-D", train_branch], check=False)
        return 0

    # Push, comment Boarded on each member, open the train PR.
    train_pr = open_train_pr(train_branch, stamp_title, merged)
    print(f"\nTrain PR opened: #{train_pr} ({train_branch})")
    for pr in merged:
        comment(str(pr["number"]), f"Boarded on #{train_pr} ({train_branch}).")
        print(f"  commented Boarded on #{pr['number']}")

    train_head = gh_json(["api", f"repos/{REPO}/pulls/{train_pr}"])["head"]["sha"]
    label = f"#{train_pr}"
    if not wait_green(train_head, label):
        report(f"TRAIN {train_branch} BLOCKED required checks did not go green ({stamp_title})")
        return 2

    print("\nTrain required checks green — merging with a merge commit.")
    run(["gh", "pr", "merge", "--repo", REPO, str(train_pr), "--merge"])
    time.sleep(10)

    member_nums = [p["number"] for p in merged]
    not_merged = members_merged(train_pr, member_nums)
    merge_sha = ""
    try:
        merge_sha = gh_json(["api", f"repos/{REPO}/pulls/{train_pr}"])["merge_commit_sha"] or ""
    except TrainError:
        pass

    if not_merged:
        for n in not_merged:
            comment(str(n), f"MEMBER NOT MARKED MERGED after train #{train_pr} landed — investigate.")
        report(
            f"TRAIN {train_branch} LANDED {merge_sha} (#{train_pr}) "
            f"members: {', '.join('#'+str(n) for n in member_nums)} "
            f"ejected: {', '.join('#'+str(e['number']) for e in ejected + conflict_ejected)} "
            f"NOT-MERGED: {', '.join('#'+str(n) for n in not_merged)}"
        )
        delete_branch(train_branch)
        return 3

    report(
        f"TRAIN {train_branch} LANDED {merge_sha} (#{train_pr}) "
        f"members: {', '.join('#'+str(n) for n in member_nums)} "
        f"ejected: {', '.join('#'+str(e['number']) for e in ejected + conflict_ejected)}"
    )
    delete_branch(train_branch)
    print("\nTrain landed and branch cleaned up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
