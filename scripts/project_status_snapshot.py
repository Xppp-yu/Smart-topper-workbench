"""Print the minimal local-state snapshot used for multi-agent handoffs."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def collect_snapshot(
    repo: Path,
    *,
    active_task: str,
    running_jobs: str,
    relevant_outputs: list[str],
) -> dict[str, str]:
    repo = repo.resolve()
    branch = _git(repo, "branch", "--show-current") or "DETACHED"
    head = _git(repo, "rev-parse", "--short=12", "HEAD")
    porcelain = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    lines = [line for line in porcelain.splitlines() if line]
    untracked = [line[3:] for line in lines if line.startswith("?? ")]
    tracked_changes = [line for line in lines if not line.startswith("?? ")]

    upstream = _git(
        repo,
        "rev-parse",
        "--abbrev-ref",
        "--symbolic-full-name",
        "@{upstream}",
        check=False,
    )
    if upstream:
        counts = _git(repo, "rev-list", "--left-right", "--count", f"{upstream}...HEAD")
        behind, ahead = counts.split()
        ahead_behind = f"ahead {ahead} / behind {behind} ({upstream})"
    else:
        ahead_behind = "NO_UPSTREAM"

    return {
        "Branch": branch,
        "HEAD": head,
        "Dirty": "yes" if tracked_changes else "no",
        "Untracked": ", ".join(untracked) if untracked else "none",
        "Active TASK": active_task,
        "Running jobs": running_jobs,
        "Relevant outputs": ", ".join(relevant_outputs) if relevant_outputs else "none declared",
        "Ahead/behind GitHub": ahead_behind,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--active-task", default="UNSET")
    parser.add_argument("--running-jobs", default="UNSET")
    parser.add_argument("--relevant-output", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    snapshot = collect_snapshot(
        args.repo,
        active_task=args.active_task,
        running_jobs=args.running_jobs,
        relevant_outputs=args.relevant_output,
    )
    for key, value in snapshot.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
