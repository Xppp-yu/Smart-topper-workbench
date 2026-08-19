from __future__ import annotations

import json
from pathlib import Path

import pytest

from topper_perception.experiments.contracts import DirtyWorktreeError, ExpIdError
from topper_perception.experiments.runner import run_experiment

EXP_ID = "EXP-RUNNER-DUMMY-SMOKE-20260819-R01"


def _config(**overrides):
    config = {
        "schema_version": "experiment-v0.1",
        "exp_id": EXP_ID,
        "task_id": "TASK-EXP-RUNNER-B-v0.1",
        "scope": "smoke",
        "runner_type": "dummy",
        "seed": 42,
        "output_root": "outputs/experiments",
        "parameters": {"n_samples": 256},
    }
    config.update(overrides)
    return config


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _dirty_git(root: Path):
    return {"repo": True, "sha": "d" * 40, "branch": "main", "dirty": True}


def test_success_generates_full_artifacts(tmp_path: Path) -> None:
    result = run_experiment(_config(), output_root=tmp_path)
    exp_dir = tmp_path / EXP_ID

    assert result.state == "SUCCEEDED"
    for name in (
        "status.json",
        "resolved_config.json",
        "manifest.json",
        "metrics.json",
        "DONE.json",
    ):
        assert (exp_dir / name).is_file(), name
    assert (exp_dir / "logs").is_dir()
    assert not (exp_dir / "FAILED.json").exists()

    assert _read_json(exp_dir / "status.json")["state"] == "SUCCEEDED"

    manifest = _read_json(exp_dir / "manifest.json")
    assert manifest["exp_id"] == EXP_ID
    assert manifest["task_id"] == "TASK-EXP-RUNNER-B-v0.1"
    assert manifest["seed"] == 42
    assert manifest["ended_at_utc"] is not None


def test_same_seed_reproduces_metrics(tmp_path: Path) -> None:
    a = run_experiment(_config(), output_root=tmp_path / "a")
    b = run_experiment(_config(), output_root=tmp_path / "b")
    assert _read_json(a.experiment_dir / "metrics.json") == _read_json(
        b.experiment_dir / "metrics.json"
    )


def test_failure_generates_failed_json(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        run_experiment(
            _config(parameters={"n_samples": 256, "fail": True}),
            output_root=tmp_path,
        )
    exp_dir = tmp_path / EXP_ID

    assert (exp_dir / "FAILED.json").is_file()
    assert not (exp_dir / "DONE.json").exists()
    assert _read_json(exp_dir / "status.json")["state"] == "FAILED"
    assert (exp_dir / "logs" / "run.log").is_file()


def test_exp_id_conflict_refuses_overwrite(tmp_path: Path) -> None:
    run_experiment(_config(), output_root=tmp_path)
    with pytest.raises(ExpIdError):
        run_experiment(_config(), output_root=tmp_path)


def test_dirty_git_refuses_mini_full(tmp_path: Path) -> None:
    mini_exp = "EXP-RUNNER-DUMMY-MINI-20260819-R01"
    config = _config(scope="mini", exp_id=mini_exp)
    with pytest.raises(DirtyWorktreeError):
        run_experiment(config, output_root=tmp_path, git_info_provider=_dirty_git)
    assert not (tmp_path / mini_exp).exists()


@pytest.mark.parametrize(
    "git_info",
    [
        {"repo": False, "sha": None, "branch": None, "dirty": None},
        {"repo": True, "sha": None, "branch": None, "dirty": False},
        {"repo": True, "sha": "", "branch": None, "dirty": False},
        {"repo": True, "sha": "d" * 40, "branch": None, "dirty": None},
        {"repo": None, "sha": "d" * 40, "branch": None, "dirty": False},
        {"repo": True, "sha": 12345, "branch": None, "dirty": False},
    ],
)
def test_mini_full_gate_fails_closed(tmp_path: Path, git_info) -> None:
    mini_exp = "EXP-RUNNER-DUMMY-MINI-20260819-R01"
    config = _config(scope="mini", exp_id=mini_exp)
    with pytest.raises(DirtyWorktreeError):
        run_experiment(config, output_root=tmp_path, git_info_provider=lambda _: git_info)
    assert not (tmp_path / mini_exp).exists()


def test_mini_full_gate_passes_when_clean(tmp_path: Path) -> None:
    def clean_git(_: Path):
        return {"repo": True, "sha": "d" * 40, "branch": "main", "dirty": False}

    mini_exp = "EXP-RUNNER-DUMMY-MINI-20260819-R01"
    result = run_experiment(
        _config(scope="mini", exp_id=mini_exp),
        output_root=tmp_path,
        git_info_provider=clean_git,
    )
    assert result.state == "SUCCEEDED"
    assert (tmp_path / mini_exp / "DONE.json").is_file()


def test_smoke_allows_dirty_and_records(tmp_path: Path) -> None:
    result = run_experiment(_config(), output_root=tmp_path, git_info_provider=_dirty_git)
    manifest = _read_json(result.experiment_dir / "manifest.json")
    assert manifest["git"]["dirty"] is True


def test_no_gpu_environment_works(tmp_path: Path) -> None:
    def no_gpu():
        return {
            "python_version": "3.12.0",
            "python_executable": "python",
            "os": "Test-OS",
            "cpu": {"arch": "x86_64", "logical_cores": 8},
            "gpu": None,
            "cuda": None,
        }

    result = run_experiment(
        _config(), output_root=tmp_path, system_info_provider=no_gpu
    )
    manifest = _read_json(result.experiment_dir / "manifest.json")
    assert manifest["gpu"] is None
    assert manifest["cuda"] is None
