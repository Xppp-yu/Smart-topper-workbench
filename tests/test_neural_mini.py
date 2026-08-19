"""End-to-end contract tests for the P5.2-B Mini screening runner (mock data only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from topper_perception.experiments import contracts
from topper_perception.experiments.runner import RUNNER_REGISTRY, run_experiment
from topper_perception.neural.mini import run_popu_neural_mini

ROWS, COLS = 64, 27
CELLS = ROWS * COLS

POSTURE_OFFSET = {"empty": 0.0, "supine": 0.2, "prone": 0.4, "left": 0.6, "right": 0.8}


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
    offset = POSTURE_OFFSET.get(posture, 0.0)
    snapshots: dict[str, dict] = {}
    for i in range(n_snapshots):
        readings = (rng.uniform(0.0, 0.05, size=CELLS) + offset).astype(float).tolist()
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


def _make_data(tmp_path: Path, n_subjects: int = 6) -> Path:
    for s in range(1, n_subjects + 1):
        subj = str(s)
        for posture in ("empty", "supine", "prone", "left", "right"):
            _write_record(tmp_path, subj, f"{posture}.json", posture, n_snapshots=4, seed=s)
        _write_record(tmp_path, subj, "others.json", None, n_snapshots=4, seed=s)
    return tmp_path


def _params(tmp_path: Path, **overrides) -> dict:
    params = {
        "subject_selection_rule": "Frozen before results: subjects [1..6] in numeric order.",
        "subject_ids": ["1", "2", "3", "4", "5", "6"],
        "max_samples": 6000,
        "val_ratio": 0.2,
        "test_ratio": 0.0,
        "batch_size": 4,
        "epochs": 5,
        "device": "cpu",
        "amp_enabled": False,
        "flip_augmentation": True,
        "optimizer": {"lr": 1e-3, "weight_decay": 0.0},
        "early_stopping": {
            "monitor": "val_loss",
            "mode": "min",
            "patience": 2,
            "min_delta": 0.0,
            "min_epochs": 3,
        },
        "resource_limits": {"max_train_seconds_per_model": 300, "max_cuda_mb": 8000},
        "viability": {"chance_margin": 0.05},
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


def test_popu_neural_mini_registered() -> None:
    assert "popu_neural_mini" in contracts.RUNNER_TYPES
    assert "popu_neural_mini" in RUNNER_REGISTRY
    assert callable(RUNNER_REGISTRY["popu_neural_mini"])


def test_mini_runs_and_writes_artifacts(tmp_path: Path) -> None:
    _make_data(tmp_path)
    exp = tmp_path / "exp"
    result = run_popu_neural_mini(_params(tmp_path), seed=42, experiment_dir=exp)

    assert result["scope"] == "mini"
    assert result["device"] == "cpu"
    assert result["cuda_available"] is torch.cuda.is_available()
    assert result["test_subjects"] == []
    assert set(result["train_subjects"]).isdisjoint(set(result["val_subjects"]))
    assert len(result["val_subjects"]) >= 1

    model = result["models"]["matrix_mlp"]
    assert model["param_count"] > 0
    assert 3 <= model["actual_epochs"] <= 5  # min_epochs..max_epochs
    assert len(model["epoch_history"]) == model["actual_epochs"]
    for record in model["epoch_history"]:
        assert set(record) == {
            "epoch", "train_loss", "val_loss", "val_accuracy",
            "val_macro_f1", "val_balanced_accuracy", "seconds", "amp_active", "is_best",
        }
        assert np.isfinite(record["train_loss"])
        assert np.isfinite(record["val_loss"])
    # Metrics fields required by the Mini contract.
    for key in ("val_macro_f1", "val_balanced_accuracy", "best_val_balanced_accuracy"):
        assert np.isfinite(model[key])
    assert len(model["per_class"]) == 5
    assert len(model["confusion_matrix"]) == 5
    assert all(len(row) == 5 for row in model["confusion_matrix"])
    assert model["best_epoch"] >= 1
    assert model["best_checkpoint_rule"].startswith("best = argmin(val_loss)")
    # CPU-safe CUDA memory field.
    assert model["peak_cuda_mb"] is None
    assert model["resume_ok"] is True
    assert model["reload_ok"] is True
    assert model["no_leakage"] is True
    assert model["same_split"] is True
    assert result["reproducible_seed"] is True

    # Viability gate ran and produced a valid verdict + reasons shape.
    assert model["viability"]["verdict"] in ("proceed", "exclude", "needs_fix")
    assert result["viability"]["overall_verdict"] in ("proceed", "exclude", "needs_fix")

    # Artifacts on disk.
    assert (exp / "checkpoints" / "matrix_mlp_latest.pt").is_file()
    assert (exp / "checkpoints" / "matrix_mlp_best.pt").is_file()
    assert (exp / "predictions" / "matrix_mlp.json").is_file()
    assert (exp / "predictions" / "matrix_mlp_best.json").is_file()
    assert (exp / "train_log.json").is_file()


def test_mini_three_models_share_split(tmp_path: Path) -> None:
    _make_data(tmp_path)
    params = _params(
        tmp_path,
        model_configs=[
            {"name": "matrix_mlp", "params": {}},
            {"name": "tiny_cnn", "params": {}},
            {"name": "small_resnet", "params": {}},
        ],
    )
    result = run_popu_neural_mini(params, seed=1, experiment_dir=tmp_path / "exp")

    assert set(result["models"]) == {"matrix_mlp", "tiny_cnn", "small_resnet"}
    signatures = {summary["split_signature"] for summary in result["models"].values()}
    assert len(signatures) == 1
    for summary in result["models"].values():
        assert summary["same_split"] is True
        assert summary["resume_ok"] is True
        assert summary["reload_ok"] is True


def test_mini_proceed_with_separable_data(tmp_path: Path) -> None:
    _make_data(tmp_path)
    # Flip off keeps every posture's offset signal intact, so the chance
    # baseline (0.25 balanced accuracy) is decisively cleared.
    params = _params(tmp_path, flip_augmentation=False)
    result = run_popu_neural_mini(params, seed=7, experiment_dir=tmp_path / "exp")

    model = result["models"]["matrix_mlp"]
    assert model["best_val_balanced_accuracy"] > 0.25
    assert model["viability"]["verdict"] == "proceed"
    assert result["viability"]["overall_verdict"] == "proceed"


def test_mini_subject_isolation(tmp_path: Path) -> None:
    _make_data(tmp_path)
    result = run_popu_neural_mini(_params(tmp_path), seed=3, experiment_dir=tmp_path / "exp")

    train = set(result["train_subjects"])
    val = set(result["val_subjects"])
    test = set(result["test_subjects"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert test == set()
    assert len(train) + len(val) == 6


def test_mini_through_governed_runner(tmp_path: Path) -> None:
    _make_data(tmp_path)
    config = {
        "schema_version": "experiment-v0.1",
        "exp_id": "EXP-P5.2-B-MINI-TEST-20260819-R01",
        "task_id": "TASK-P5.2-B-MINI-SCREEN-v0.1",
        "scope": "mini",
        "runner_type": "popu_neural_mini",
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
    metrics = json.loads((exp_dir / "metrics.json").read_text("utf-8"))
    assert metrics["scope"] == "mini"
    assert metrics["models"]["matrix_mlp"]["resume_ok"] is True


def test_mini_exp_id_refuses_overwrite(tmp_path: Path) -> None:
    _make_data(tmp_path)
    config = {
        "schema_version": "experiment-v0.1",
        "exp_id": "EXP-P5.2-B-MINI-TEST-20260819-R01",
        "task_id": "TASK-P5.2-B-MINI-SCREEN-v0.1",
        "scope": "mini",
        "runner_type": "popu_neural_mini",
        "seed": 42,
        "output_root": "outputs/experiments",
        "parameters": _params(tmp_path),
    }
    run_experiment(
        config,
        output_root=tmp_path / "out",
        git_info_provider=_clean_git,
        system_info_provider=_no_gpu,
    )
    from topper_perception.experiments.contracts import ExpIdError

    with pytest.raises(ExpIdError):
        run_experiment(
            config,
            output_root=tmp_path / "out",
            git_info_provider=_clean_git,
            system_info_provider=_no_gpu,
        )


def test_mini_requires_labeled_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_popu_neural_mini(_params(tmp_path), seed=1, experiment_dir=tmp_path / "exp")
