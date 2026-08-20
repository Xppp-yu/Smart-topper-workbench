"""Frozen P5.2-C Full protocol validation + selection-rule tests (no torch)."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from topper_perception.experiments import contracts
from topper_perception.experiments.contracts import validate_experiment_config
from topper_perception.experiments.runner import RUNNER_REGISTRY
from topper_perception.neural.full_protocol import (
    FullCandidateResult,
    record_ece,
    record_multiclass_brier,
    record_multiclass_nll,
    select_full_winner,
    validate_calibration_probabilities,
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
        (
            lambda c: c["parameters"]["inner_epoch_selection"]["optimizer"].__setitem__("name", "SGD"),
            "optimizer.name",
        ),
        (
            lambda c: c["parameters"]["training_params"].__setitem__("batch_size", 64),
            "batch_size",
        ),
        (
            lambda c: c["parameters"]["training_params"].__setitem__("num_workers", 2),
            "num_workers",
        ),
        (
            lambda c: c["parameters"]["training_params"].__setitem__("loss", "mse"),
            "loss",
        ),
        (
            lambda c: c["parameters"]["training_params"].__setitem__("optimizer", "SGD"),
            "optimizer",
        ),
        (
            lambda c: c["parameters"]["training_params"].__setitem__("deterministic_cudnn", False),
            "deterministic_cudnn",
        ),
        (
            lambda c: c["parameters"]["training_params"].__setitem__("cudnn_benchmark", True),
            "cudnn_benchmark",
        ),
        (
            lambda c: c["parameters"]["training_params"].__setitem__("amp", "apex"),
            "amp",
        ),
        (
            lambda c: c["parameters"]["training_params"].__setitem__(
                "frozen_label_order", ["a", "b", "c", "d", "e"]
            ),
            "frozen_label_order",
        ),
        (
            lambda c: c["parameters"]["training_seeds"].__setitem__(
                "stage_a_train_seed", "9_000_000 + outer_seed * 100 + local_fold"
            ),
            "stage_a_train_seed",
        ),
        (
            lambda c: c["parameters"]["training_seeds"].__setitem__(
                "stage_b_refit_seed", "9_000_000 + outer_seed * 100 + local_fold"
            ),
            "stage_b_refit_seed",
        ),
        (
            lambda c: c["parameters"]["training_seeds"].__setitem__(
                "reset_seed_before_each_candidate", False
            ),
            "reset_seed_before_each_candidate",
        ),
        (
            lambda c: c["parameters"]["frozen_svm_reference_artifacts"].pop(),
            "frozen_svm_reference_artifacts",
        ),
        (
            lambda c: c["parameters"]["calibration"].__setitem__("record_nll", "-mean(log(p_true))"),
            "record_nll",
        ),
        (
            lambda c: c["parameters"]["calibration"].__setitem__("ece_bins", 10),
            "ece_bins",
        ),
        (
            lambda c: c["parameters"]["model_selection"].__setitem__(
                "complexity_priority",
                ["matrix_mlp", "tiny_cnn", "small_resnet", "calibrated_linear_svm"],
            ),
            "complexity_priority",
        ),
        (
            lambda c: c.__setitem__("seed", 43),
            "seed",
        ),
        (
            lambda c: c["parameters"].__setitem__("dataset", "other_dataset"),
            "dataset",
        ),
        (
            lambda c: c["data_manifests"][0].__setitem__("path", "other.csv"),
            r"data_manifests\[0\]\.path",
        ),
        (
            lambda c: c["parameters"].__setitem__("quality_manifest", "other.csv"),
            "quality_manifest",
        ),
        (
            lambda c: c["parameters"]["evaluation_protocol"].__setitem__("shuffle", False),
            "shuffle",
        ),
        (
            lambda c: c["parameters"]["evaluation_protocol"].__setitem__("shuffle", "false"),
            "shuffle",
        ),
        (
            lambda c: c["parameters"]["outer_refit"].__setitem__("infer_once_on_outer_test", False),
            "infer_once_on_outer_test",
        ),
        (
            lambda c: c["parameters"]["model_configs"][0]["params"].__setitem__("hidden_dims", [128]),
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


def test_selection_rejects_nan_metric() -> None:
    svm = _svm(0.9500)
    nn = _candidate(model="matrix_mlp", macro_f1=float("nan"))
    with pytest.raises(ValueError, match="non-finite"):
        select_full_winner([svm, nn])


def test_selection_rejects_inf_metric() -> None:
    svm = _svm(0.9500)
    nn = _candidate(model="tiny_cnn", macro_f1=0.95, bal_acc=float("inf"))
    with pytest.raises(ValueError, match="non-finite"):
        select_full_winner([svm, nn])


def test_selection_rejects_out_of_range_metric() -> None:
    svm = _svm(0.9500)
    nn = _candidate(model="small_resnet", macro_f1=0.95, bal_acc=1.5)
    with pytest.raises(ValueError, match="outside reasonable range"):
        select_full_winner([svm, nn])


def test_selection_complexity_priority_prefers_smaller_network() -> None:
    svm = _svm(0.9000, bal_acc=0.90, worst=0.80, std=0.01, weakest=0.80)
    mlp = _candidate(model="matrix_mlp", macro_f1=0.9600, bal_acc=0.960, worst=0.90, std=0.01, weakest=0.90)
    resnet = _candidate(model="small_resnet", macro_f1=0.9600, bal_acc=0.960, worst=0.90, std=0.01, weakest=0.90)
    # Both NNs clear the SVM and tie on primary/balanced-acc/worst-subject, so the
    # fixed complexity_priority breaks the tie: small_resnet < matrix_mlp.
    assert select_full_winner([mlp, resnet, svm]) == "small_resnet"


def test_selection_order_independent_under_tie() -> None:
    svm = _svm(0.9000, bal_acc=0.90, worst=0.80, std=0.01, weakest=0.80)
    mlp = _candidate(model="matrix_mlp", macro_f1=0.9600, bal_acc=0.960, worst=0.90, std=0.01, weakest=0.90)
    tiny = _candidate(model="tiny_cnn", macro_f1=0.9600, bal_acc=0.960, worst=0.90, std=0.01, weakest=0.90)
    resnet = _candidate(model="small_resnet", macro_f1=0.9600, bal_acc=0.960, worst=0.90, std=0.01, weakest=0.90)
    expected = "tiny_cnn"
    assert select_full_winner([svm, mlp, tiny, resnet]) == expected
    assert select_full_winner([resnet, tiny, mlp, svm]) == expected
    assert select_full_winner([tiny, svm, resnet, mlp]) == expected


def test_selection_primary_metric_breaks_tie_before_complexity() -> None:
    # Regression: SVM is out of near-tie; both NNs beat it and share balanced-acc /
    # worst-subject, but differ on the primary metric. complexity_priority must NOT
    # decide — record_macro_f1_mean breaks the tie first.
    svm = _svm(0.9000, bal_acc=0.90, worst=0.85, std=0.01, weakest=0.85)
    mlp = _candidate(model="matrix_mlp", macro_f1=0.9500, bal_acc=0.940, worst=0.90, std=0.01, weakest=0.90)
    tiny = _candidate(model="tiny_cnn", macro_f1=0.9460, bal_acc=0.940, worst=0.90, std=0.01, weakest=0.90)
    assert select_full_winner([svm, mlp, tiny]) == "matrix_mlp"
    assert select_full_winner([tiny, mlp, svm]) == "matrix_mlp"


# ---------------------------------------------------------------------------
# Calibration formulas (record-level, diagnostic-only)
# ---------------------------------------------------------------------------


def test_calibration_validate_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        validate_calibration_probabilities([[0.5, 0.5, float("nan")]], [0])


def test_calibration_validate_rejects_out_of_unit_range() -> None:
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        validate_calibration_probabilities([[1.2, -0.2, 0.0]], [0])


def test_calibration_validate_rejects_row_sum_mismatch() -> None:
    with pytest.raises(ValueError, match="sums to"):
        validate_calibration_probabilities([[0.6, 0.3, 0.3]], [0])


def test_calibration_validate_rejects_label_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        validate_calibration_probabilities([[1.0, 0.0, 0.0]], [3])


def test_record_multiclass_nll_perfect_prediction() -> None:
    probs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    labels = [0, 1, 2]
    assert record_multiclass_nll(probs, labels) == pytest.approx(0.0)


def test_record_multiclass_nll_clips_at_1e_minus_15() -> None:
    probs = [[0.0, 1.0, 0.0]]
    labels = [0]
    assert record_multiclass_nll(probs, labels) == pytest.approx(-math.log(1e-15))


def test_record_multiclass_brier_perfect() -> None:
    probs = [[1.0, 0.0], [0.0, 1.0]]
    labels = [0, 1]
    assert record_multiclass_brier(probs, labels) == pytest.approx(0.0)


def test_record_multiclass_brier_known_value() -> None:
    probs = [[0.7, 0.3]]
    labels = [0]
    # (0.7-1)^2 + (0.3-0)^2 = 0.09 + 0.09 = 0.18
    assert record_multiclass_brier(probs, labels) == pytest.approx(0.18)


def test_record_ece_perfectly_calibrated() -> None:
    # Both rows at confidence 0.5; one correct, one incorrect -> accuracy 0.5.
    probs = [[0.5, 0.5], [0.5, 0.5]]
    labels = [0, 1]
    assert record_ece(probs, labels) == pytest.approx(0.0)


def test_record_ece_overconfident() -> None:
    # Both rows confident (0.9) but only one correct -> ECE > 0.
    probs = [[0.9, 0.1], [0.9, 0.1]]
    labels = [0, 1]
    assert record_ece(probs, labels) > 0.0


def test_record_nll_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        record_multiclass_nll([], [])


def test_record_brier_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        record_multiclass_brier([], [])


def test_record_ece_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        record_ece([], [])


def test_record_ece_rejects_non_frozen_n_bins() -> None:
    probs = [[0.5, 0.5], [0.5, 0.5]]
    labels = [0, 1]
    with pytest.raises(ValueError, match="n_bins"):
        record_ece(probs, labels, n_bins=10)
