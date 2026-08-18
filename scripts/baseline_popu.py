"""Run the P5/R5 subject-isolated posture baseline on the P4a feature table.

Freezes the evaluation protocol (subject-isolated split, GroupKFold model
selection, once-only held-out test) and evaluates the fixed candidate set for
the primary cohort (ACCEPT) and a separate ACCEPT+WARN sensitivity analysis.

Only the 71 P4a feature columns enter the model; ``subject_id`` is the grouping
key, and every preprocessing step is inside a per-fold-fitted Pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from topper_perception.baseline.popu import (
    POSTURE_LABELS,
    build_model,
    compute_metrics,
    feature_columns,
    filter_cohort,
    per_subject_metrics,
    predict,
    select_best_model,
    sort_subjects_numeric,
    split_subjects,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FAILING_SUBJECT_ACCURACY_THRESHOLD = 0.9

PREDICTION_COLUMNS = (
    "sample_id",
    "subject_id",
    "cohort",
    "split",
    "model",
    "model_version",
    "y_true",
    "y_pred",
    "confidence",
)

COMPARISON_COLUMNS = (
    "cohort",
    "model",
    "model_version",
    "role",
    "split",
    "n_samples",
    "macro_f1",
    "macro_f1_std",
    "balanced_accuracy",
    "balanced_accuracy_std",
    "accuracy",
    "accuracy_std",
)

PER_CLASS_COLUMNS = ("cohort", "model", "split", "label", "precision", "recall", "f1", "support")

PER_SUBJECT_COLUMNS = (
    "cohort",
    "model",
    "split",
    "subject_id",
    "n_samples",
    "n_correct",
    "n_errors",
    "accuracy",
    "macro_f1",
    "is_failing",
)

CONFUSION_COLUMNS = ("cohort", "model", "split", "true_label", "pred_label", "count")


def _project_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/popu_baseline_p5_v0.1.json"),
        help="P5/R5 baseline protocol config.",
    )
    return parser


def _load_feature_table(config: dict[str, Any]) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    table_path = _project_path(config["input_feature_table"])
    summary_path = _project_path(config["input_feature_summary"])
    df = pd.read_csv(table_path)
    df["subject_id"] = df["subject_id"].astype(str)

    columns = feature_columns(df.columns)
    summary = _read_json(summary_path)
    expected = list(summary["feature_columns"])
    if columns != expected:
        raise ValueError(
            "Feature columns in the table do not match the P4a summary; "
            f"got {len(columns)} columns, expected {len(expected)}."
        )
    if len(columns) != int(config["feature_count"]):
        raise ValueError(
            f"Feature count mismatch: table has {len(columns)}, "
            f"config expects {config['feature_count']}."
        )
    return df, columns, summary


def _model_role(config: dict[str, Any], model_name: str) -> str:
    for model in config["models"]:
        if model["name"] == model_name:
            return str(model.get("role", "candidate"))
    return "candidate"


def _write_csv_rows(rows: list[dict[str, Any]], columns: tuple[str, ...], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _fold_statistics(fold_metrics: list[dict[str, Any]]) -> dict[str, float]:
    keys = ("macro_f1", "balanced_accuracy", "accuracy")
    return {
        key: float(np.std([metrics[key] for metrics in fold_metrics])) if fold_metrics else float("nan")
        for key in keys
    }


def _write_predictions(
    writer: csv.DictWriter,
    *,
    cohort: str,
    model_name: str,
    model_version: str,
    split: str,
    sample_ids: np.ndarray,
    subject_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
) -> None:
    for index in range(len(y_true)):
        writer.writerow(
            {
                "sample_id": str(sample_ids[index]),
                "subject_id": str(subject_ids[index]),
                "cohort": cohort,
                "split": split,
                "model": model_name,
                "model_version": model_version,
                "y_true": str(y_true[index]),
                "y_pred": str(y_pred[index]),
                "confidence": float(confidence[index]),
            }
        )


def _evaluate_cohort(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    cohort: str,
    config: dict[str, Any],
    prediction_writer: csv.DictWriter,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Evaluate every model on one cohort; stream predictions as it goes.

    Returns ``(results_by_model, comparison_rows, per_class_rows,
    per_subject_rows, confusion_rows, cohort_summary)`` for this cohort.
    """
    held_out = [str(subject) for subject in config["evaluation_protocol"]["held_out_subjects"]]
    n_splits = int(config["evaluation_protocol"]["dev_n_splits"])
    random_state = int(config["random_state"])
    labels = list(POSTURE_LABELS)

    subjects = sort_subjects_numeric(df["subject_id"].unique())
    dev_subjects, test_subjects = split_subjects(subjects, held_out=held_out)

    x = df[feature_cols].to_numpy(dtype=float)
    y = df["posture"].to_numpy()
    groups = df["subject_id"].to_numpy()
    sample_ids = df["sample_id"].to_numpy()
    subject_ids = df["subject_id"].to_numpy()

    dev_mask = np.isin(subject_ids, dev_subjects)
    test_mask = np.isin(subject_ids, test_subjects)
    x_dev, y_dev, groups_dev = x[dev_mask], y[dev_mask], groups[dev_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    nan_rows = int(np.isnan(x).any(axis=1).sum())
    nan_cells = int(np.isnan(x).sum())

    results: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    per_subject_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []

    group_kfold = GroupKFold(n_splits=n_splits)

    for model_cfg in config["models"]:
        model_name = str(model_cfg["name"])
        spec = build_model(model_name, random_state=random_state)
        role = _model_role(config, model_name)

        # --- Development-set GroupKFold out-of-fold evaluation -----------------
        oof_pred = np.empty(len(x_dev), dtype=object)
        oof_confidence = np.empty(len(x_dev), dtype=float)
        fold_metrics: list[dict[str, Any]] = []
        for train_idx, val_idx in group_kfold.split(x_dev, y_dev, groups=groups_dev):
            fold_model = build_model(model_name, random_state=random_state).estimator
            fold_model.fit(x_dev[train_idx], y_dev[train_idx])
            oof_pred[val_idx], oof_confidence[val_idx] = predict(fold_model, x_dev[val_idx])
            fold_metrics.append(compute_metrics(y_dev[val_idx], oof_pred[val_idx], labels))

        dev_metrics = compute_metrics(y_dev, oof_pred, labels)
        dev_std = _fold_statistics(fold_metrics)

        # --- Final fit on the whole development set, once-only test evaluation --
        final_model = build_model(model_name, random_state=random_state).estimator
        final_model.fit(x_dev, y_dev)
        test_pred, test_confidence = predict(final_model, x_test)
        test_metrics = compute_metrics(y_test, test_pred, labels)

        results[model_name] = {
            "version": spec.version,
            "role": role,
            "dev": dev_metrics,
            "dev_std": dev_std,
            "test": test_metrics,
        }

        for split, metrics, n_samples in (
            ("dev", dev_metrics, int(len(x_dev))),
            ("test", test_metrics, int(len(x_test))),
        ):
            std = dev_std if split == "dev" else {}
            comparison_rows.append(
                {
                    "cohort": cohort,
                    "model": model_name,
                    "model_version": spec.version,
                    "role": role,
                    "split": split,
                    "n_samples": n_samples,
                    "macro_f1": round(metrics["macro_f1"], 6),
                    "macro_f1_std": round(std.get("macro_f1", float("nan")), 6),
                    "balanced_accuracy": round(metrics["balanced_accuracy"], 6),
                    "balanced_accuracy_std": round(std.get("balanced_accuracy", float("nan")), 6),
                    "accuracy": round(metrics["accuracy"], 6),
                    "accuracy_std": round(std.get("accuracy", float("nan")), 6),
                }
            )

            for label in labels:
                per_class = metrics["per_class"][label]
                per_class_rows.append(
                    {
                        "cohort": cohort,
                        "model": model_name,
                        "split": split,
                        "label": label,
                        "precision": round(per_class["precision"], 6),
                        "recall": round(per_class["recall"], 6),
                        "f1": round(per_class["f1"], 6),
                        "support": per_class["support"],
                    }
                )

            confusion = metrics["confusion"]
            for true_index, true_label in enumerate(labels):
                for pred_index, pred_label in enumerate(labels):
                    confusion_rows.append(
                        {
                            "cohort": cohort,
                            "model": model_name,
                            "split": split,
                            "true_label": true_label,
                            "pred_label": pred_label,
                            "count": int(confusion[true_index, pred_index]),
                        }
                    )

        # Per-subject rows and predictions for both splits.
        for split, y_true, y_pred, subject_set in (
            ("dev", y_dev, oof_pred, subject_ids[dev_mask]),
            ("test", y_test, test_pred, subject_ids[test_mask]),
        ):
            for row in per_subject_metrics(subject_set, y_true, y_pred, labels):
                per_subject_rows.append(
                    {
                        "cohort": cohort,
                        "model": model_name,
                        "split": split,
                        **row,
                        "is_failing": row["accuracy"] < FAILING_SUBJECT_ACCURACY_THRESHOLD,
                    }
                )

        _write_predictions(
            prediction_writer,
            cohort=cohort,
            model_name=model_name,
            model_version=spec.version,
            split="dev",
            sample_ids=sample_ids[dev_mask],
            subject_ids=subject_ids[dev_mask],
            y_true=y_dev,
            y_pred=oof_pred,
            confidence=oof_confidence,
        )
        _write_predictions(
            prediction_writer,
            cohort=cohort,
            model_name=model_name,
            model_version=spec.version,
            split="test",
            sample_ids=sample_ids[test_mask],
            subject_ids=subject_ids[test_mask],
            y_true=y_test,
            y_pred=test_pred,
            confidence=test_confidence,
        )

    cohort_summary = {
        "name": cohort,
        "n_subjects": len(subjects),
        "n_dev_subjects": len(dev_subjects),
        "n_test_subjects": len(test_subjects),
        "held_out_subjects": test_subjects,
        "n_samples": int(len(x)),
        "n_dev_samples": int(len(x_dev)),
        "n_test_samples": int(len(x_test)),
        "nan_rows": nan_rows,
        "nan_cells": nan_cells,
        "posture_counts": {
            posture: int((y == posture).sum()) for posture in labels
        },
    }
    return results, comparison_rows, per_class_rows, per_subject_rows, confusion_rows, cohort_summary


def _render_confusion_figure(
    results: dict[str, dict[str, Any]],
    *,
    cohort: str,
    output_path: Path,
) -> None:
    labels = list(POSTURE_LABELS)
    model_names = [name for name in results if name != "dummy"] + (["dummy"] if "dummy" in results else [])
    model_names = [name for name in model_names if name in results]
    splits = ("dev", "test")

    fig, axes = plt.subplots(
        len(model_names),
        len(splits),
        figsize=(3.2 * len(splits), 3.0 * len(model_names)),
        constrained_layout=True,
    )
    if len(model_names) == 1:
        axes = axes.reshape(1, -1)

    for row, model_name in enumerate(model_names):
        for col, split in enumerate(splits):
            ax = axes[row, col]
            confusion = results[model_name][split]["confusion"]
            image = ax.imshow(confusion, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(labels)), labels, fontsize=8)
            for true_index in range(len(labels)):
                for pred_index in range(len(labels)):
                    ax.text(
                        pred_index,
                        true_index,
                        str(int(confusion[true_index, pred_index])),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if confusion[true_index, pred_index] < confusion.max() * 0.7 else "black",
                    )
            ax.set_title(f"{model_name} / {split}", fontsize=9)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"PoPu P5 posture baseline — {cohort} cohort confusion matrices", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _read_json(_project_path(args.config))
    df, feature_cols, feature_summary = _load_feature_table(config)

    outputs = {key: _project_path(value) for key, value in config["outputs"].items()}
    predictions_path = outputs["predictions"]
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict[str, Any]] = {}
    all_comparison: list[dict[str, Any]] = []
    all_per_class: list[dict[str, Any]] = []
    all_per_subject: list[dict[str, Any]] = []
    all_confusion: list[dict[str, Any]] = []
    cohort_summaries: list[dict[str, Any]] = []

    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        for cohort_cfg in config["cohorts"]:
            cohort_name = str(cohort_cfg["name"])
            df_cohort = filter_cohort(df, cohort_name)
            (
                results,
                comparison_rows,
                per_class_rows,
                per_subject_rows,
                confusion_rows,
                cohort_summary,
            ) = _evaluate_cohort(
                df_cohort,
                feature_cols,
                cohort=cohort_name,
                config=config,
                prediction_writer=writer,
            )
            all_results[cohort_name] = results
            all_comparison.extend(comparison_rows)
            all_per_class.extend(per_class_rows)
            all_per_subject.extend(per_subject_rows)
            all_confusion.extend(confusion_rows)
            cohort_summaries.append(cohort_summary)

    _write_csv_rows(all_comparison, COMPARISON_COLUMNS, outputs["model_comparison"])
    _write_csv_rows(all_per_class, PER_CLASS_COLUMNS, outputs["per_class"])
    _write_csv_rows(all_per_subject, PER_SUBJECT_COLUMNS, outputs["per_subject"])
    _write_csv_rows(all_confusion, CONFUSION_COLUMNS, outputs["confusion"])

    primary_name = next(
        str(c["name"]) for c in config["cohorts"] if c.get("is_primary")
    )
    selected_candidate = select_best_model(all_comparison, cohort=primary_name)

    _render_confusion_figure(all_results[primary_name], cohort=primary_name, output_path=outputs["confusion_figure"])

    summary: dict[str, Any] = {
        "dataset": "PoPu",
        "sensor_layer": "tactilus",
        "stage": "P5/R5",
        "baseline_version": config["baseline_version"],
        "task": config["task"],
        "labels": config["labels"],
        "feature_schema_version": feature_summary.get("feature_schema_version"),
        "feature_count": len(feature_cols),
        "random_state": config["random_state"],
        "evaluation_protocol": config["evaluation_protocol"],
        "model_selection": config["model_selection"],
        "cohorts": cohort_summaries,
        "primary_cohort": primary_name,
        "selected_candidate": {
            "cohort": primary_name,
            "model": selected_candidate,
            "criterion": config["model_selection"]["criterion"],
        },
        "dev_results": {
            cohort: {
                model: {
                    "version": spec["version"],
                    "role": spec["role"],
                    "dev": {
                        key: spec["dev"][key]
                        for key in ("accuracy", "balanced_accuracy", "macro_f1")
                    },
                    "dev_std": spec["dev_std"],
                    "test": {
                        key: spec["test"][key]
                        for key in ("accuracy", "balanced_accuracy", "macro_f1")
                    },
                }
                for model, spec in model_results.items()
            }
            for cohort, model_results in all_results.items()
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Public-data candidate result on PoPu Tactilus only; not a claim of "
            "product capability, self-developed hardware, or closed-loop efficacy."
        ),
    }
    summary_path = outputs["summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
