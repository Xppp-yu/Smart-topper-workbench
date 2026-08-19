from __future__ import annotations

import json
from pathlib import Path

from topper_perception.experiments import artifacts


def test_atomic_write_json_roundtrips(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "data.json"
    artifacts.atomic_write_json(path, {"a": 1, "b": [1, 2, 3], "c": {"x": "y"}})
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == {"a": 1, "b": [1, 2, 3], "c": {"x": "y"}}


def test_atomic_write_json_leaves_no_tmp_files(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    artifacts.atomic_write_json(path, {"k": "v"})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["data.json"]


def test_config_hash_is_stable_and_order_independent() -> None:
    a = artifacts.compute_config_hash({"x": 1, "y": 2})
    b = artifacts.compute_config_hash({"y": 2, "x": 1})
    assert a == b
    assert a.startswith("sha256:")
    assert len(a) == len("sha256:") + 64


def test_config_hash_differs_when_content_differs() -> None:
    assert artifacts.compute_config_hash({"x": 1}) != artifacts.compute_config_hash({"x": 2})


def test_capture_system_info_reports_python_and_cpu() -> None:
    info = artifacts.capture_system_info()
    assert info["python_version"]
    assert info["os"]
    assert isinstance(info["cpu"]["logical_cores"], int)
    assert "gpu" in info
    assert "cuda" in info


def test_gpu_is_null_when_no_nvidia_smi(monkeypatch) -> None:
    monkeypatch.setattr(artifacts.shutil, "which", lambda name: None)
    assert artifacts._detect_gpu_info() is None


def test_gpu_probe_reports_driver_version(monkeypatch) -> None:
    class Result:
        returncode = 0
        stdout = "NVIDIA GeForce RTX 4090, 24564, 595.58.03\n"

    monkeypatch.setattr(artifacts.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(artifacts.subprocess, "run", lambda *args, **kwargs: Result())

    assert artifacts._detect_gpu_info() == {
        "name": "NVIDIA GeForce RTX 4090",
        "memory_mb": 24564,
        "driver_version": "595.58.03",
    }


def test_capture_system_info_includes_detected_cuda_metadata(monkeypatch) -> None:
    gpu = {"name": "GPU", "memory_mb": 1, "driver_version": "driver"}
    cuda = {
        "driver_version": "driver",
        "torch_version": "torch",
        "torch_cuda_version": "runtime",
        "cudnn_version": 1,
        "available": True,
        "device_count": 1,
    }
    monkeypatch.setattr(artifacts, "_detect_gpu_info", lambda: gpu)
    monkeypatch.setattr(artifacts, "_detect_cuda_info", lambda detected: cuda)

    info = artifacts.capture_system_info()

    assert info["gpu"] == gpu
    assert info["cuda"] == cuda


def test_capture_git_info_reports_repo_fields() -> None:
    info = artifacts.capture_git_info(Path(__file__).resolve().parents[1])
    assert isinstance(info["dirty"], bool)
    assert "sha" in info


def test_capture_git_info_returns_unknown_for_non_repo(tmp_path: Path) -> None:
    info = artifacts.capture_git_info(tmp_path)
    assert info["repo"] is False
    assert info["sha"] is None
    assert info["dirty"] is None
