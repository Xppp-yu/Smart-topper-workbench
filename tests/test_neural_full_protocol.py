"""Frozen P5.2-C Full protocol validation + selection-rule tests (no torch)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from topper_perception.experiments import contracts
from topper_perception.experiments.contracts import validate_experiment_config
from topper_perception.experiments.runner import RUNNER_REGISTRY
from topper_perception.neural.full_protocol import (
    FullCandidateResult,
    select_full_winner,
    validate_full_config,
    validate_full_data_boundary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FULL_CONFIG_PATH = REPO_ROOT / "configs" / "experiments" / "popu_neural_full_v0.1.json"


def _load_config() -> dict:
    return json.loads(FULL_CONFIG_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Config / data-boundary validation
# ---------------------------------------------------------------------------


def test_full_config_parses_schema_and_frozen_values() -> None:
    config = _load_config()
    parsed = validate_experiment_config(config)  # schema-level: must not raise
    assert parsed.scope == "full"
    assert parsed.runner_type == "popu_neural_full"
    validate_full_config(config)  # frozen-values: must not raise


def test_full_runner_registered_but_not_implemented() -> None:
    assert "popu_neural_full" in contracts.RUNNER_TYPES
    assert "popu_neural_full" in RUNNER_REGISTRY
    with pytest.raises(NotImplementedError, match="not implemented"):
        RUNNER_REGISTRY["popu_neural_full"]({}, seed=42, experiment_dir=Path("."))


@pytest.mark.parametrize(
    "mutate, match",
    [
        (
            lambda c: c["data_manifests"].__setitem__(0, {**c["data_manifests"][0], "sha256": "f" * 64}),
            "sha256",
        ),
        (
            lambda c: c["parameters"].__setitem__("quality_manifest_sha256", "f" * 64),
            "quality_manifest_sha256",
        ),
        (
            lambda c: c["parameters"]["data_boundary"].__setitem__("n_records", 5007),
            "n_records",
        ),
        (
            lambda c: c["parameters"]["data_boundary"].__setitem__("n_snapshots", 50059),
            "n_snapshots",
        ),
        (
            lambda c: c["parameters"]["candidates"]["neural"].pop(),
            "candidates.neural",
        ),
        (
            lambda c: c["parameters"]["evaluation_protocol"].__setitem__("seeds", [11, 22, 44]),
            "seeds",
        ),
        (
            lambda c: c["parameters"]["inner_epoch_selection"]["optimizer"].__setitem__("lr", 0.01),
            "lr",
        ),
        (
            lambda c: c["parameters"]["inner_epoch_selection"].__setitem__("monitor", "val_accuracy"),
            "monitor",
        ),
        (
            lambda c: c["parameters"]["model_selection"].__setitem__("margin", 0.01),
            "margin",
        ),
        (
            lambda c: c["parameters"]["resources_and_stop"].__setitem__("device", "cpu"),
            "device",
        ),
        (
            lambda c: c["parameters"]["model_configs"].append({"name": "extra_cnn", "params": {}}),
            "model_configs",
        ),
    ],
)
def test_full_config_frozen_value_drift_rejected(mutate, match: str) -> None:
    config = copy.deepcopy(_load_config())
    mutate(config)
    with pytest.raises(ValueError, match=match):
        validate_full_config(config)


def test_full_data_boundary_accepts_frozen_values() -> None:
    validate_full_data_boundary(
        n_subjects=60, n_records=5006, n_snapshots=50060, snapshots_per_record=10
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_subjects": 59, "n_records": 5006, "n_snapshots": 50060, "snapshots_per_record": 10}, "n_subjects"),
        ({"n_subjects": 60, "n_records": 5006, "n_snapshots": 50060, "snapshots_per_record": 9}, "snapshots_per_record"),
    ],
)
def test_full_data_boundary_fails_closed(kwargs, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        validate_full_data_boundary(**kwargs)


def test_full_data_boundary_rejects_inconsistent_snapshots() -> None:
    with pytest.raises(ValueError, match="snapshots"):
        validate_full_data_boundary(
            n_subjects=60, n_records=5006, n_snapshots=50061, snapshots_per_record=10
        )


# ---------------------------------------------------------------------------
# Selection rule
# ---------------------------------------------------------------------------


def _candidate(
    *,
    model: str,
    passed_gate: bool = True,
    is_frozen_svm: bool = False,
    macro_f1: float = 0.0,
    bal_acc: float = 0.0,
    worst: float = 0.0,
    std: float = 0.0,
    weakest: float = 0.0,
) -> FullCandidateResult:
    return FullCandidateResult(
        model=model,
        passed_gate=passed_gate,
        is_frozen_svm=is_frozen_svm,
        record_macro_f1_mean=macro_f1,
        record_balanced_acc_mean=bal_acc,
        worst_subject_macro_f1_mean=worst,
        record_macro_f1_std=std,
        weakest_class_record_f1=weakest,
    )


def _svm(macro_f1: float, **kwargs) -> FullCandidateResult:
    return _candidate(model="calibrated_linear_svm", is_frozen_svm=True, macro_f1=macro_f1, **kwargs)


def test_selection_neural_beats_svm_by_more_than_margin() -> None:
    svm = _svm(0.9450, bal_acc=0.945, worst=0.90, std=0.01, weakest=0.90)
    nn = _candidate(model="matrix_mlp", macro_f1=0.9510, bal_acc=0.95, worst=0.90, std=0.01, weakest=0.90)
    assert select_full_winner([svm, nn]) == "matrix_mlp"


def test_selection_near_tie_prefers_svm_without_substantive_improvement() -> None:
    svm = _svm(0.9500, bal_acc=0.950, worst=0.900, std=0.010, weakest=0.900)
    # NN is within margin and even has a higher balanced accuracy, but shows no
    # pre-registered substantive improvement, so the SVM is preferred.
    nn = _candidate(model="tiny_cnn", macro_f1=0.9520, bal_acc=0.960, worst=0.900, std=0.010, weakest=0.900)
    assert select_full_winner([svm, nn]) == "calibrated_linear_svm"


def test_selection_near_tie_neural_wins_on_worst_subject_improvement() -> None:
    svm = _svm(0.9500, bal_acc=0.950, worst=0.900, std=0.010, weakest=0.900)
    nn = _candidate(model="small_resnet", macro_f1=0.9510, bal_acc=0.951, worst=0.925, std=0.010, weakest=0.900)
    assert select_full_winner([svm, nn]) == "small_resnet"


def test_selection_near_tie_neural_wins_on_weakest_class_improvement() -> None:
    svm = _svm(0.9500, bal_acc=0.950, worst=0.900, std=0.010, weakest=0.880)
    nn = _candidate(model="matrix_mlp", macro_f1=0.9505, bal_acc=0.951, worst=0.900, std=0.010, weakest=0.891)
    assert select_full_winner([svm, nn]) == "matrix_mlp"


def test_selection_near_tie_neural_wins_on_std_reduction() -> None:
    svm = _svm(0.9500, bal_acc=0.950, worst=0.900, std=0.010, weakest=0.900)
    nn = _candidate(model="tiny_cnn", macro_f1=0.9500, bal_acc=0.951, worst=0.900, std=0.0085, weakest=0.900)
    assert select_full_winner([svm, nn]) == "tiny_cnn"


def test_selection_tie_break_uses_balanced_accuracy_ladder() -> None:
    svm = _svm(0.9400, bal_acc=0.940, worst=0.90, std=0.01, weakest=0.90)
    nn_a = _candidate(model="matrix_mlp", macro_f1=0.9500, bal_acc=0.945, worst=0.90, std=0.01, weakest=0.90)
    nn_b = _candidate(model="small_resnet", macro_f1=0.9480, bal_acc=0.960, worst=0.90, std=0.01, weakest=0.90)
    # Both neural candidates clear the SVM by > margin; within their near-tie the
    # balanced-accuracy ladder picks the higher one.
    assert select_full_winner([svm, nn_a, nn_b]) == "small_resnet"


def test_selection_excludes_gate_failures() -> None:
    svm = _svm(0.9500, bal_acc=0.950, worst=0.90, std=0.01, weakest=0.90)
    nn = _candidate(
        model="matrix_mlp", passed_gate=False, macro_f1=0.999, bal_acc=0.999,
        worst=0.99, std=0.001, weakest=0.99,
    )
    assert select_full_winner([svm, nn]) == "calibrated_linear_svm"


def test_selection_requires_exactly_one_svm() -> None:
    svm = _svm(0.9500, bal_acc=0.95, worst=0.90, std=0.01, weakest=0.90)
    with pytest.raises(ValueError, match="Exactly one frozen SVM"):
        select_full_winner([svm, _candidate(model="other_svm", is_frozen_svm=True, macro_f1=0.9)])


def test_selection_requires_some_candidate_passing_gate() -> None:
    svm = _svm(0.9500, passed_gate=False)
    with pytest.raises(ValueError, match="gate"):
        select_full_winner([svm])


def test_selection_rejects_svm_failed_gate() -> None:
    svm = _svm(0.9500, passed_gate=False)
    nn = _candidate(model="matrix_mlp", macro_f1=0.9510)
    with pytest.raises(ValueError, match="Frozen SVM"):
        select_full_winner([svm, nn])


def test_candidate_dataclass_is_frozen() -> None:
    c = _svm(0.9)
    with pytest.raises((AttributeError, TypeError)):
        c.record_macro_f1_mean = 0.99  # type: ignore[misc]
