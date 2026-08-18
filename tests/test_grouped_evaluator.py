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

from topper_perception.evaluation import aggregation, grouped
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


def _record_synthetic_data(n_records: int = 8, snapshots_per_record: int = 10, seed: int = 0):
    """Grouped synthetic data shaped like the PoPu record contract.

    Each record belongs to exactly one group (subject) and carries exactly
    ``snapshots_per_record`` frames whose sample_id follows the P4a contract
    ``popu-tactilus::<source_relative_path>#frame=<N>``.
    """
    rng = np.random.RandomState(seed)
    n = n_records * snapshots_per_record
    records = np.repeat(np.arange(n_records), snapshots_per_record)
    groups = records  # one subject per record
    y = np.where(records % 2 == 0, "A", "B").astype(str)
    x = rng.randn(n, 3)
    x[:, 0] += np.where(records % 2 == 0, 1.5, -1.5)
    frame_numbers = np.arange(n) % snapshots_per_record
    sample_ids = np.array(
        [
            f"popu-tactilus::sub{r}/rec{r}.json#frame={i}"
            for r, i in zip(records, frame_numbers)
        ]
    )
    return x, y, groups.astype(str), sample_ids


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


def test_oof_evaluation_rejects_groups_misaligned_with_samples() -> None:
    x, y, groups, sample_ids = _synthetic_data()
    folds = grouped.generate_group_folds(groups, n_splits=4, shuffle=False)

    # A groups sequence whose length does not match x must fail loudly with
    # a clear error instead of indexing the group array out of bounds while
    # emitting per-sample rows.
    with pytest.raises(ValueError, match="groups"):
        grouped.evaluate_grouped_oof(
            _logreg_model(),
            x,
            y,
            ["alpha", "beta", "gamma"],
            folds,
            sample_ids=sample_ids,
        )


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

    # The frozen config labels describe PoPu; the synthetic data here is A/B, so
    # pass labels matching the synthetic classes to avoid a label-drift error.
    for model_cfg in cfg["models"]:
        model = build_model(model_cfg, random_state=int(cfg["random_seed"]))
        result = grouped.evaluate_grouped_oof(
            model, x, y, groups, folds, sample_ids=sample_ids, labels=["A", "B"]
        )
        assert len(result.predictions) == len(x)


def test_oof_predictions_include_per_class_proba_columns() -> None:
    x, y, groups, sample_ids = _synthetic_data()
    folds = grouped.generate_group_folds(groups, n_splits=4, shuffle=False)
    result = grouped.evaluate_grouped_oof(
        _logreg_model(), x, y, groups, folds, sample_ids=sample_ids, labels=["A", "B"]
    )
    assert {"proba__A", "proba__B"} <= set(result.predictions.columns)


def test_oof_proba_rows_are_finite_and_sum_to_one() -> None:
    x, y, groups, sample_ids = _synthetic_data()
    folds = grouped.generate_group_folds(groups, n_splits=4, shuffle=False)
    result = grouped.evaluate_grouped_oof(
        _logreg_model(), x, y, groups, folds, sample_ids=sample_ids, labels=["A", "B"]
    )
    proba = result.predictions[["proba__A", "proba__B"]].to_numpy(dtype=float)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)


def test_oof_proba_columns_align_to_frozen_label_order() -> None:
    x, y, groups, sample_ids = _synthetic_data()
    folds = grouped.generate_group_folds(groups, n_splits=4, shuffle=False)
    result = grouped.evaluate_grouped_oof(
        _logreg_model(), x, y, groups, folds, sample_ids=sample_ids, labels=["B", "A"]
    )
    preds = result.predictions
    # argmax over the frozen-order proba columns must equal y_pred regardless of
    # whether the frozen order happens to match sklearn's sorted classes_.
    argmax = np.where(preds["proba__B"] >= preds["proba__A"], "B", "A")
    assert (argmax == preds["y_pred"].to_numpy()).all()


def test_oof_estimator_with_unfrozen_class_raises() -> None:
    x, y, groups, sample_ids = _synthetic_data()
    folds = grouped.generate_group_folds(groups, n_splits=4, shuffle=False)
    # The estimator sees both A and B, but the frozen label set claims only A:
    # that is label drift and must fail loudly, never silently misalign.
    with pytest.raises(ValueError, match="class"):
        grouped.evaluate_grouped_oof(
            _logreg_model(), x, y, groups, folds, sample_ids=sample_ids, labels=["A"]
        )


def test_snapshot_metrics_are_computed_per_repeat() -> None:
    x, y, groups, sample_ids = _synthetic_data()
    folds = grouped.generate_repeated_group_folds(
        groups, n_splits=4, shuffle=True, seed=7, n_repeats=2
    )
    result = grouped.evaluate_grouped_oof(
        _logreg_model(), x, y, groups, folds, sample_ids=sample_ids, labels=["A", "B"]
    )
    per_repeat = grouped.snapshot_metrics_per_repeat(result)
    assert len(per_repeat) == 2
    assert [row["repeat"] for row in per_repeat] == [0, 1]
    for row in per_repeat:
        assert {"accuracy", "balanced_accuracy", "macro_f1"} <= set(row)
    reduced = grouped.reduce_repeat_metrics(per_repeat)
    assert set(reduced) == {"accuracy", "balanced_accuracy", "macro_f1"}
    for metric in reduced.values():
        assert 0.0 <= metric["mean"] <= 1.0
        assert metric["std"] >= 0.0


def test_subject_metrics_averaged_across_repeats_with_worst_flagged() -> None:
    x, y, groups, sample_ids = _synthetic_data(n_groups=6, samples_per_group=20)
    folds = grouped.generate_repeated_group_folds(
        groups, n_splits=3, shuffle=True, seed=3, n_repeats=2
    )
    result = grouped.evaluate_grouped_oof(
        _logreg_model(), x, y, groups, folds, sample_ids=sample_ids, labels=["A", "B"]
    )
    subjects = grouped.repeated_subject_metrics(result.predictions, labels=["A", "B"])
    assert len(subjects) == 6
    for row in subjects:
        assert {"subject_id", "accuracy_mean", "macro_f1_mean"} <= set(row)
    worst = [row for row in subjects if row["is_worst"]]
    assert len(worst) == 1
    assert worst[0]["accuracy_mean"] == min(row["accuracy_mean"] for row in subjects)


def test_record_level_selection_uses_criterion_then_complexity() -> None:
    rows = [
        {"model": "random_forest", "record_macro_f1_mean": 0.91, "record_balanced_acc_mean": 0.90, "worst_subject_macro_f1_mean": 0.72},
        {"model": "logistic_regression", "record_macro_f1_mean": 0.907, "record_balanced_acc_mean": 0.905, "worst_subject_macro_f1_mean": 0.70},
        {"model": "knn", "record_macro_f1_mean": 0.90, "record_balanced_acc_mean": 0.89, "worst_subject_macro_f1_mean": 0.68},
    ]
    order = ["logistic_regression", "knn", "random_forest"]
    # Within margin, a lower-complexity candidate wins even if a more complex
    # one has a slightly higher primary score.
    winner = grouped.select_best_candidate(
        rows, criterion="record_macro_f1_mean", tie_break="record_balanced_acc_mean",
        worst_subject_criterion="worst_subject_macro_f1_mean",
        complexity_order=order, exclude=(), margin=0.005,
    )
    assert winner == "logistic_regression"
    # With a zero margin the top primary score alone decides.
    strict = grouped.select_best_candidate(
        rows, criterion="record_macro_f1_mean", tie_break="record_balanced_acc_mean",
        worst_subject_criterion="worst_subject_macro_f1_mean",
        complexity_order=order, exclude=(), margin=0.0,
    )
    assert strict == "random_forest"


def test_record_level_selection_tie_break_order() -> None:
    rows = [
        {"model": "a", "record_macro_f1_mean": 0.90, "record_balanced_acc_mean": 0.89, "worst_subject_macro_f1_mean": 0.70},
        {"model": "b", "record_macro_f1_mean": 0.90, "record_balanced_acc_mean": 0.90, "worst_subject_macro_f1_mean": 0.70},
        {"model": "c", "record_macro_f1_mean": 0.90, "record_balanced_acc_mean": 0.90, "worst_subject_macro_f1_mean": 0.75},
    ]
    best = grouped.select_best_candidate(
        rows, criterion="record_macro_f1_mean", tie_break="record_balanced_acc_mean",
        worst_subject_criterion="worst_subject_macro_f1_mean",
        complexity_order=["a", "b", "c"], exclude=(), margin=0.0,
    )
    assert best == "c"


def test_record_level_selection_excludes_and_fails_without_candidates() -> None:
    rows = [
        {"model": "dummy", "record_macro_f1_mean": 0.20, "record_balanced_acc_mean": 0.20, "worst_subject_macro_f1_mean": 0.20},
    ]
    with pytest.raises(ValueError):
        grouped.select_best_candidate(
            rows, criterion="record_macro_f1_mean", tie_break="record_balanced_acc_mean",
            worst_subject_criterion="worst_subject_macro_f1_mean",
            complexity_order=["logistic_regression"], exclude=("dummy",),
        )


def test_evaluator_to_aggregation_end_to_end() -> None:
    x, y, groups, sample_ids = _record_synthetic_data(n_records=8, snapshots_per_record=10)
    folds = grouped.generate_repeated_group_folds(
        groups, n_splits=4, shuffle=True, seed=11, n_repeats=2
    )
    result = grouped.evaluate_grouped_oof(
        _logreg_model(), x, y, groups, folds, sample_ids=sample_ids, labels=["A", "B"]
    )
    preds = result.predictions.copy()
    preds["record_id"] = preds["sample_id"].map(
        lambda sid: aggregation.record_id_from_sample_id(str(sid))
    )
    records = aggregation.aggregate_record_predictions(
        preds, record_id_col="record_id", group_id_col="group_id", y_true_col="y_true",
        label_columns=["proba__A", "proba__B"], repeat_id_col="repeat",
    )
    # 8 records x 2 repeats, and the 10 snapshots of each (repeat, record) are
    # never mixed across repeats or across records.
    assert len(records) == 8 * 2
    assert records["n_snapshots"].unique().tolist() == [10]
    first = records.iloc[0]
    src = preds[
        (preds["repeat"] == first["repeat"]) & (preds["record_id"] == first["record_id"])
    ]
    assert first["proba__A"] == pytest.approx(float(src["proba__A"].mean()))
