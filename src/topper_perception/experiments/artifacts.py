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
                "--query-gpu=name,memory.total,driver_version",
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
    fields = [field.strip() for field in line.split(",")]
    if len(fields) < 2:
        return None
    name, memory = fields[:2]
    return {
        "name": name,
        "memory_mb": int(memory) if memory.isdigit() else None,
        "driver_version": fields[2] if len(fields) >= 3 else None,
    }


def _detect_cuda_info(gpu_info: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Best-effort PyTorch/CUDA runtime metadata without requiring PyTorch."""
    try:
        import torch
    except (ImportError, OSError):
        if gpu_info is None:
            return None
        return {
            "driver_version": gpu_info.get("driver_version"),
            "torch_version": None,
            "torch_cuda_version": None,
            "cudnn_version": None,
            "available": None,
            "device_count": None,
        }

    available = bool(torch.cuda.is_available())
    return {
        "driver_version": gpu_info.get("driver_version") if gpu_info else None,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "available": available,
        "device_count": int(torch.cuda.device_count()) if available else 0,
    }


def capture_system_info() -> dict[str, Any]:
    """Report Python/OS/CPU and GPU/CUDA; GPU/CUDA are ``None`` when absent."""
    gpu_info = _detect_gpu_info()
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "os": platform.platform(),
        "cpu": {
            "arch": platform.machine(),
            "logical_cores": os.cpu_count() or 0,
        },
        "gpu": gpu_info,
        "cuda": _detect_cuda_info(gpu_info),
    }
