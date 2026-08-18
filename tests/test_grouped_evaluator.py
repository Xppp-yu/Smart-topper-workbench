"""Tests for the dataset-agnostic subject-grouped cross-validation evaluator.

The evaluator must never import PoPu, never know file paths, and must make it
structurally impossible for model selection to read any historical held-out
test: selection consumes only out-of-fold predictions produced by this module.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from topper_perception.evaluation import grouped
from topper_perception.models.registry import build_model

# A stratified dummy may predict a class that a small validation fold never
# contains; sklearn warns on that.  It is expected and benign here.
pytestmark = pytest.mark.filterwarnings(
    "ignore:y_pred contains classes not in y_true:UserWarning"
)


def _synthetic_data(n_groups: int = 10, samples_per_group: int = 20, seed: int = 0):
    """Grouped two-class synthetic data whose labels are learnable from X.

    ``x[:, 0]`` carries a strong signal aligned with the group-parity label, so
    logistic regression reliably beats a stratified dummy on out-of-fold data
    without the test being sensitive to random noise.
    """
    rng = np.random.RandomState(seed)
    n = n_groups * samples_per_group
    groups = np.repeat(np.arange(n_groups), samples_per_group)
    y = np.where(groups % 2 == 0, "A", "B").astype(str)
    x = rng.randn(n, 3)
    x[:, 0] += np.where(groups % 2 == 0, 1.5, -1.5)
    return x, y, groups.astype(str), np.arange(n)


def _logreg_model(seed: int = 0):
    return build_model(
        {
            "name": "logistic_regression",
            "version": "logreg@test",
            "estimator": "LogisticRegression",
            "params": {"max_iter": 2000},
            "preprocessing": [
                {"estimator": "SimpleImputer", "params": {"strategy": "median"}},
                {"estimator": "StandardScaler", "params": {}},
            ],
        },
        random_state=seed,
    )


def _dummy_model(seed: int = 0):
    return build_model(
        {
            "name": "dummy",
            "version": "dummy@stratified",
            "estimator": "DummyClassifier",
            "params": {"strategy": "stratified"},
        },
        random_state=seed,
    )


def test_group_folds_do_not_leak_groups() -> None:
    groups = np.array(["1", "1", "2", "2", "3", "3", "4", "4", "5", "5"])
    folds = grouped.generate_group_folds(groups, n_splits=5, shuffle=False)

    grouped.validate_group_folds(folds, groups)  # must not raise

    for train_idx, val_idx in folds.folds:
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        assert train_groups.isdisjoint(val_groups)


def test_every_sample_is_validated_exactly_once() -> None:
    groups = np.array(["1", "1", "2", "2", "3", "3", "4", "4", "5", "5"])
    folds = grouped.generate_group_folds(groups, n_splits=5, shuffle=False)

    counts = np.zeros(len(groups), dtype=int)
    for _, val_idx in folds.folds:
        counts[val_idx] += 1
    assert counts.tolist() == [1] * len(groups)


def test_same_seed_and_config_produce_identical_folds() -> None:
    groups = np.array([str(i % 6) for i in range(60)])
    first = grouped.generate_group_folds(groups, n_splits=4, shuffle=True, seed=99)
    second = grouped.generate_group_folds(groups, n_splits=4, shuffle=True, seed=99)

    for (train_a, val_a), (train_b, val_b) in zip(first.folds, second.folds):
        assert np.array_equal(train_a, train_b)
        assert np.array_equal(val_a, val_b)


def test_different_seeds_change_folds() -> None:
    groups = np.array([str(i % 6) for i in range(60)])
    first = grouped.generate_group_folds(groups, n_splits=4, shuffle=True, seed=1)
    second = grouped.generate_group_folds(groups, n_splits=4, shuffle=True, seed=2)

    assert not all(
        np.array_equal(val_a, val_b)
        for (_, val_a), (_, val_b) in zip(first.folds, second.folds)
    )


def test_repeat_generation_marks_repeats() -> None:
    groups = np.array([str(i % 6) for i in range(60)])
    folds = grouped.generate_repeated_group_folds(
        groups, n_splits=4, shuffle=True, seed=5, n_repeats=2
    )
    assert len(folds) == 2
    assert all(isinstance(item, grouped.GroupFolds) for item in folds)


def test_generate_folds_requires_enough_groups() -> None:
    groups = np.array(["1", "2"])
    with pytest.raises(ValueError):
        grouped.generate_group_folds(groups, n_splits=5, shuffle=False)


def test_evaluator_module_has_no_popu_dependency() -> None:
    tree = ast.parse(Path(grouped.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert not any("popu" in name.lower() for name in imported)
    assert not any("tactilus" in name.lower() for name in imported)


def test_oof_evaluation_predicts_each_sample_once() -> None:
    x, y, groups, sample_ids = _synthetic_data()
    folds = grouped.generate_group_folds(groups, n_splits=4, shuffle=False)

    result = grouped.evaluate_grouped_oof(
        _logreg_model(), x, y, groups, folds, sample_ids=sample_ids
    )

    preds = result.predictions
    assert len(preds) == len(sample_ids)
    assert set(preds["sample_id"]) == set(sample_ids.astype(str))


def test_oof_predictions_carry_traceability_columns() -> None:
    x, y, groups, sample_ids = _synthetic_data()
    folds = grouped.generate_group_folds(groups, n_splits=4, shuffle=False)

    result = grouped.evaluate_grouped_oof(
        _logreg_model(), x, y, groups, folds, sample_ids=sample_ids
    )

    expected = {
        "sample_id",
        "group_id",
        "fold_id",
        "repeat",
        "y_true",
        "y_pred",
        "confidence",
    }
    assert expected <= set(result.predictions.columns)


def test_oof_metrics_per_fold_and_summary() -> None:
    x, y, groups, sample_ids = _synthetic_data()
    folds = grouped.generate_group_folds(groups, n_splits=4, shuffle=False)
    labels = ["A", "B"]

    result = grouped.evaluate_grouped_oof(
        _logreg_model(), x, y, groups, folds, sample_ids=sample_ids, labels=labels
    )

    assert len(result.per_fold_metrics) == 4
    for fold in result.per_fold_metrics:
        assert {"accuracy", "balanced_accuracy", "macro_f1"} <= set(fold)

    summary = grouped.oof_summary(result)
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert "per_group" in summary
    assert "fold_macro_f1_mean" in summary
    assert "worst_group" in summary
    assert result.predictions["y_true"].isin(labels).all()


def test_selection_reads_only_oof_never_held_out() -> None:
    x, y, groups, sample_ids = _synthetic_data()
    folds = grouped.generate_group_folds(groups, n_splits=4, shuffle=False)
    labels = ["A", "B"]

    logreg = grouped.evaluate_grouped_oof(
        _logreg_model(), x, y, groups, folds, sample_ids=sample_ids, labels=labels
    )
    dummy = grouped.evaluate_grouped_oof(
        _dummy_model(), x, y, groups, folds, sample_ids=sample_ids, labels=labels
    )

    best = grouped.select_best_model(
        {"logistic_regression": logreg, "dummy": dummy},
        criterion="macro_f1",
        exclude=("dummy",),
    )
    assert best == "logistic_regression"

    # A dummy-only candidate set with dummy excluded fails loudly instead of
    # silently returning the excluded model.
    with pytest.raises(ValueError):
        grouped.select_best_model({"dummy": dummy}, criterion="macro_f1", exclude=("dummy",))


def test_repeated_oof_assigns_unique_fold_ids() -> None:
    x, y, groups, sample_ids = _synthetic_data()
    folds = grouped.generate_repeated_group_folds(
        groups, n_splits=4, shuffle=True, seed=7, n_repeats=2
    )

    result = grouped.evaluate_grouped_oof(
        _logreg_model(), x, y, groups, folds, sample_ids=sample_ids, labels=["A", "B"]
    )

    assert result.predictions["fold_id"].nunique() == 8
    assert set(result.predictions["repeat"]) == {0, 1}


def test_p5_1_config_models_build_and_run_end_to_end() -> None:
    import json

    config_path = Path("configs/experiments/popu_model_comparison_p5_1_v0.1.json")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))

    x, y, groups, sample_ids = _synthetic_data()
    folds = grouped.generate_group_folds(
        groups,
        n_splits=int(cfg["evaluation_protocol"]["n_splits"]),
        shuffle=False,
    )

    for model_cfg in cfg["models"]:
        model = build_model(model_cfg, random_state=int(cfg["random_seed"]))
        result = grouped.evaluate_grouped_oof(
            model, x, y, groups, folds, sample_ids=sample_ids, labels=cfg["labels"]
        )
        assert len(result.predictions) == len(x)
