from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "project_status_snapshot.py"
SPEC = importlib.util.spec_from_file_location("project_status_snapshot", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def test_collect_snapshot_separates_tracked_and_untracked_changes(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", "Snapshot Test")
    _git(tmp_path, "config", "user.email", "snapshot@example.invalid")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("v1\n", encoding="utf-8")
    _git(tmp_path, "add", "--", "tracked.txt")
    _git(tmp_path, "commit", "-m", "initial")

    tracked.write_text("v2\n", encoding="utf-8")
    (tmp_path / "new.txt").write_text("new\n", encoding="utf-8")
    snapshot = MODULE.collect_snapshot(
        tmp_path,
        active_task="TASK-TEST-v0.1",
        running_jobs="none",
        relevant_outputs=["outputs/test.json"],
    )

    assert snapshot["Dirty"] == "yes"
    assert snapshot["Untracked"] == "new.txt"
    assert snapshot["Active TASK"] == "TASK-TEST-v0.1"
    assert snapshot["Running jobs"] == "none"
    assert snapshot["Relevant outputs"] == "outputs/test.json"
    assert snapshot["Ahead/behind GitHub"] == "NO_UPSTREAM"
