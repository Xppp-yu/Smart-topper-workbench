"""End-to-end contract tests for the P5.2-A2 neural smoke runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from topper_perception.experiments import contracts
from topper_perception.experiments.runner import (
    RUNNER_REGISTRY,
    run_experiment,
)
from topper_perception.neural.runner import run_popu_neural_smoke

ROWS, COLS = 64, 27
CELLS = ROWS * COLS


def _write_record(
    data_root: Path,
    subject_id: str,
    filename: str,
    posture: str | None,
    n_snapshots: int = 4,
    seed: int = 0,
) -> Path:
    subject_dir = data_root / "tactilus_data" / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    snapshots: dict[str, dict] = {}
    for i in range(n_snapshots):
        readings = rng.uniform(0.0, 1.0, size=CELLS).astype(float).tolist()
        snapshots[str(i)] = {"id": f"snap_{i}", "tactilus_readings": readings}
    record: dict = {
        "tactilus_rows": ROWS,
        "tactilus_columns": COLS,
        "volunteer_id": subject_id,
        "variation": "v1",
        "snapshots": snapshots,
    }
    if posture is not None:
        record["position"] = posture
    path = subject_dir / filename
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def _make_data(tmp_path: Path, n_subjects: int = 2) -> Path:
    for s in range(1, n_subjects + 1):
        subj = str(s)
        for posture in ("empty", "supine", "prone", "left", "right"):
            _write_record(tmp_path, subj, f"{posture}.json", posture, n_snapshots=4, seed=s)
        _write_record(tmp_path, subj, "others.json", None, n_snapshots=4, seed=s)
    return tmp_path


def _params(tmp_path: Path, **overrides) -> dict:
    params = {
        "subject_ids": ["1", "2"],
        "max_samples": 1000,
        "val_ratio": 0.5,
        "test_ratio": 0.0,
        "batch_size": 4,
        "epochs": 1,
        "device": "cpu",
        "amp_enabled": False,
        "flip_augmentation": True,
        "optimizer": {"lr": 1e-3, "weight_decay": 0.0},
        "model_configs": [{"name": "matrix_mlp", "params": {}}],
        "data_root": str(tmp_path),
    }
    params.update(overrides)
    return params


def _no_gpu() -> dict:
    return {
        "python_version": "3.12.0",
        "python_executable": "python",
        "os": "Test-OS",
        "cpu": {"arch": "x86_64", "logical_cores": 8},
        "gpu": None,
        "cuda": None,
    }


def _clean_git(_: Path) -> dict:
    return {"repo": True, "sha": "d" * 40, "branch": "main", "dirty": False}


def test_popu_neural_registered() -> None:
    assert "popu_neural" in contracts.RUNNER_TYPES
    assert "popu_neural" in RUNNER_REGISTRY
    assert callable(RUNNER_REGISTRY["popu_neural"])


def test_smoke_runs_and_writes_artifacts(tmp_path: Path) -> None:
    _make_data(tmp_path)
    exp = tmp_path / "exp"
    result = run_popu_neural_smoke(_params(tmp_path), seed=42, experiment_dir=exp)

    model = result["models"]["matrix_mlp"]
    assert model["param_count"] > 0
    assert np.isfinite(model["final_train_loss"])
    assert np.isfinite(model["val_loss"])
    assert model["resume_ok"] is True
    assert model["reload_prediction_consistent"] is True
    assert result["reproducible_seed"] is True
    assert result["cuda_available"] is False

    assert (exp / "checkpoints" / "matrix_mlp_latest.pt").is_file()
    assert (exp / "checkpoints" / "matrix_mlp_best.pt").is_file()
    assert (exp / "predictions" / "matrix_mlp.json").is_file()
    assert (exp / "train_log.json").is_file()

    # Predictions carry provenance and valid probabilities.
    preds = json.loads((exp / "predictions" / "matrix_mlp.json").read_text("utf-8"))
    assert preds["frozen_labels"] == ["empty", "supine", "prone", "left", "right"]
    assert preds["n_samples"] == result["models"]["matrix_mlp"]["val_samples"]
    for row in preds["predictions"]:
        assert row["sample_id"] and row["record_id"] and row["subject_id"]
        assert abs(sum(row["probabilities"]) - 1.0) < 1e-4


def test_smoke_runs_three_models(tmp_path: Path) -> None:
    _make_data(tmp_path)
    params = _params(
        tmp_path,
        model_configs=[
            {"name": "matrix_mlp", "params": {}},
            {"name": "tiny_cnn", "params": {}},
            {"name": "small_resnet", "params": {}},
        ],
    )
    result = run_popu_neural_smoke(params, seed=1, experiment_dir=tmp_path / "exp")

    assert set(result["models"]) == {"matrix_mlp", "tiny_cnn", "small_resnet"}
    for model in result["models"].values():
        assert np.isfinite(model["final_train_loss"])
        assert model["resume_ok"] is True
        assert model["reload_prediction_consistent"] is True


def test_smoke_through_governed_runner(tmp_path: Path) -> None:
    _make_data(tmp_path)
    config = {
        "schema_version": "experiment-v0.1",
        "exp_id": "EXP-NEURAL-SMOKE-TEST-20260819-R01",
        "task_id": "TASK-P5.2-A2-TRAINING-CPU-SMOKE-v0.1",
        "scope": "smoke",
        "runner_type": "popu_neural",
        "seed": 42,
        "output_root": "outputs/experiments",
        "parameters": _params(tmp_path),
    }
    result = run_experiment(
        config,
        output_root=tmp_path / "out",
        git_info_provider=_clean_git,
        system_info_provider=_no_gpu,
    )
    exp_dir = result.experiment_dir

    assert result.state == "SUCCEEDED"
    assert (exp_dir / "DONE.json").is_file()
    assert (exp_dir / "metrics.json").is_file()
    assert (exp_dir / "checkpoints" / "matrix_mlp_latest.pt").is_file()

    metrics = json.loads((exp_dir / "metrics.json").read_text("utf-8"))
    assert metrics["frozen_labels"] == ["empty", "supine", "prone", "left", "right"]
    assert metrics["models"]["matrix_mlp"]["resume_ok"] is True
    assert np.isfinite(metrics["models"]["matrix_mlp"]["final_train_loss"])


def test_smoke_requires_labeled_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_popu_neural_smoke(_params(tmp_path), seed=1, experiment_dir=tmp_path / "exp")


def test_smoke_rejects_unknown_model_config(tmp_path: Path) -> None:
    _make_data(tmp_path)
    params = _params(tmp_path, model_configs=[{"name": "resnet152", "params": {}}])
    with pytest.raises(ValueError, match="Unknown model name"):
        run_popu_neural_smoke(params, seed=1, experiment_dir=tmp_path / "exp")
