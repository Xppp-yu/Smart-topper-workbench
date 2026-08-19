"""Artifact writing and environment/Git metadata capture for experiment runs.

All JSON artifacts are written via temp-file + atomic replace so a crash never
leaves a half-written file. Environment and Git metadata are gathered here so
the runner can record a truthful ``manifest.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


def atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Write ``data`` as pretty JSON, atomically, via a temp file + replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def compute_config_hash(config: Mapping[str, Any]) -> str:
    """Return a stable ``sha256:...`` hash of a canonical JSON encoding."""
    canonical = json.dumps(
        config, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _git(project_root: Path, *args: str) -> str | None:
    """Run a read-only ``git`` command; return stripped stdout, or None on failure."""
    try:
        result = subprocess.run(
            ["git", "-C", str(project_root), *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def capture_git_info(project_root: Path) -> dict[str, Any]:
    """Return the repo's Git SHA/branch/dirty status, or ``UNKNOWN`` if not a repo."""
    if not (project_root / ".git").exists():
        return {"repo": False, "sha": None, "branch": None, "dirty": None}
    sha = _git(project_root, "rev-parse", "HEAD")
    branch = _git(project_root, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git(project_root, "status", "--porcelain")
    # Fail closed: if ``git status`` could not run, the dirty state is unknown
    # (None) rather than assumed clean, so the mini/full gate refuses the run.
    dirty = None if status is None else bool(status.strip())
    return {
        "repo": True,
        "sha": sha,
        "branch": branch,
        "dirty": dirty,
    }


def _detect_gpu_info() -> dict[str, Any] | None:
    """Best-effort GPU probe via ``nvidia-smi``; returns None when absent/failing."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    line = result.stdout.strip().splitlines()[0]
    name, _, memory = line.partition(",")
    return {
        "name": name.strip(),
        "memory_mb": int(memory.strip()) if memory.strip().isdigit() else None,
    }


def capture_system_info() -> dict[str, Any]:
    """Report Python/OS/CPU and GPU/CUDA; GPU/CUDA are ``None`` when absent."""
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "os": platform.platform(),
        "cpu": {
            "arch": platform.machine(),
            "logical_cores": os.cpu_count() or 0,
        },
        "gpu": _detect_gpu_info(),
        # torch is not a dependency of this batch, so CUDA runtime is absent.
        "cuda": None,
    }
