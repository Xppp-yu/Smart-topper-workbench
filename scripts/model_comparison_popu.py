"""Run the P5.1-B repeated subject-grouped candidate comparison on PoPu.

Ranks the registry-driven candidates (dummy lower bound + 6 candidates) on the
P4a primary cohort with repeated subject-grouped cross-validation: 5 folds x 3
repeats (seeds 11/22/33), all candidates sharing identical folds, every repeat
scored separately, mean/std across repeats reported (never a pooled 150k-row
score).  Produces snapshot/record/subject-level metrics, a feature-ablation
table (top-2 round-1 candidates x 5 feature groups), a frozen "PoPu research
candidate" refit on the full primary cohort, and all P5.1 artifacts under the
``popu_model_comparison_p5_1_v0.1`` prefix.

The raw feature table is read-only; only the configured outputs are written.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix

from topper_perception.baseline.popu import filter_cohort
from topper_perception.evaluation import (
    PROBA_PREFIX,
    aggregate_record_predictions,
    compute_metrics,
    evaluate_grouped_oof,
    generate_group_folds,
    record_id_from_sample_id,
    reduce_repeat_metrics,
    repeated_subject_metrics,
    select_best_candidate,
    snapshot_metrics_per_repeat,
)
from topper_perception.features.groups import feature_group_columns
from topper_perception.models.class_order import FrozenClassOrderClassifier
from topper_perception.models.registry import build_model

# A batch experiment: sklearn convergence/user warnings are filtered so the log
# stays readable.  Real errors still propagate and are never swallowed.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUMMARY_COLUMNS = (
    "model",
    "model_version",
    "role",
    "status",
    "skip_reason",
    "n_snapshot_samples",
    "n_records_per_repeat",
    "n_subjects",
    "snapshot_accuracy_mean",
    "snapshot_accuracy_std",
    "snapshot_balanced_acc_mean",
    "snapshot_balanced_acc_std",
    "snapshot_macro_f1_mean",
    "snapshot_macro_f1_std",
    "record_accuracy_mean",
    "record_accuracy_std",
    "record_balanced_acc_mean",
    "record_balanced_acc_std",
    "record_macro_f1_mean",
    "record_macro_f1_std",
    "worst_subject_id",
    "worst_subject_accuracy_mean",
    "worst_subject_macro_f1_mean",
    "fold_macro_f1_mean",
    "fold_macro_f1_std",
    "fold_accuracy_mean",
    "fold_accuracy_std",
    "fit_time_seconds",
    "inference_time_seconds",
    "model_size_bytes",
)

PER_CLASS_COLUMNS = (
    "model",
    "model_version",
    "label",
    "precision_mean",
    "precision_std",
    "recall_mean",
    "recall_std",
    "f1_mean",
    "f1_std",
    "support_mean",
    "support_std",
)

PER_SUBJECT_COLUMNS = (
    "model",
    "model_version",
    "subject_id",
    "n_repeats",
    "n_samples_mean",
    "accuracy_mean",
    "accuracy_std",
    "macro_f1_mean",
    "macro_f1_std",
    "is_worst",
)

FOLD_REPEAT_COLUMNS = (
    "model",
    "model_version",
    "repeat",
    "seed",
    "local_fold",
    "fold_id",
    "n_samples",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
)

ABLATION_COLUMNS = (
    "model",
    "model_version",
    "feature_group",
    "n_features",
    "record_macro_f1_mean",
    "record_macro_f1_std",
    "record_balanced_acc_mean",
    "record_balanced_acc_std",
    "reused_from_round1",
)

CONFUSION_COLUMNS = ("model", "model_version", "true_label", "pred_label", "count")


def _project_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/popu_model_comparison_p5_1_v0.1.json"),
        help="P5.1 comparison protocol config.",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run only the one-fold smoke-timing pass per candidate, print the "
        "table, and exit without the full CV run.",
    )
    return parser


def _load_primary_cohort(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Load the P4a table, validate the 71 feature columns, filter primary."""
    from topper_perception.baseline.popu import feature_columns

    table_path = _project_path(config["feature_schema"]["feature_table"])
    summary_path = _project_path(config["feature_schema"]["feature_summary"])
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
    if len(columns) != int(config["feature_schema"]["feature_count"]):
        raise ValueError(
            f"Feature count mismatch: table has {len(columns)}, "
            f"config expects {config['feature_schema']['feature_count']}."
        )
    primary = filter_cohort(df, "primary")
    return primary, columns, summary


def _build_record_id_map(primary: pd.DataFrame) -> dict[str, str]:
    """Map every sample_id to its record id under the frozen P4a contract.

    Each JSON record's 10 snapshots share one ``source_relative_path``; the
    sample-id parser must recover exactly that path or the table has drifted.
    """
    record_map: dict[str, str] = {}
    for sample_id, path in zip(primary["sample_id"], primary["source_relative_path"]):
        parsed = record_id_from_sample_id(str(sample_id))
        if parsed != str(path):
            raise ValueError(
                f"sample_id {sample_id!r} does not resolve to source_relative_path "
                f"{path!r} (got {parsed!r}); record aggregation would mix snapshots."
            )
        record_map[str(sample_id)] = parsed
    return record_map


def _smoke_timing(
    model_cfg: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    folds: Sequence[Any],
    *,
    random_state: int,
) -> dict[str, float]:
    """Fit one fold's training slice and time it; return timing + pickle size."""
    registered = build_model(model_cfg, random_state=random_state)
    train_idx, val_idx = folds[0].folds[0]
    estimator = clone(registered.estimator)
    started = time.perf_counter()
    estimator.fit(x[train_idx], y[train_idx])
    fit_time = time.perf_counter() - started
    started = time.perf_counter()
    estimator.predict_proba(x[val_idx])
    inference_time = time.perf_counter() - started
    return {
        "fit_time_seconds": fit_time,
        "inference_time_seconds": inference_time,
        "model_size_bytes": float(len(pickle.dumps(estimator))),
    }


def _aggregate_records(
    oof: pd.DataFrame,
    record_map: dict[str, str],
    labels: list[str],
) -> pd.DataFrame:
    """Add record_id to an OOF frame and aggregate snapshots per record/repeat."""
    frame = oof.copy()
    frame["record_id"] = frame["sample_id"].map(record_map)
    missing = int(frame["record_id"].isna().sum())
    if missing:
        raise ValueError(
            f"{missing} OOF samples have no record_id; record aggregation would "
            "silently drop them."
        )
    return aggregate_record_predictions(
        frame,
        record_id_col="record_id",
        group_id_col="group_id",
        y_true_col="y_true",
        label_columns=[f"{PROBA_PREFIX}{label}" for label in labels],
        proba_prefix=PROBA_PREFIX,
        repeat_id_col="repeat",
    )


def _repeat_metrics(frame: pd.DataFrame, labels: list[str]) -> list[dict[str, Any]]:
    """One metric row per repeat over a frame carrying a ``repeat`` column."""
    rows: list[dict[str, Any]] = []
    for repeat in sorted(int(value) for value in frame["repeat"].unique()):
        sub = frame[frame["repeat"] == repeat]
        metrics = compute_metrics(
            sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(), labels=labels
        )
        rows.append(
            {
                "repeat": repeat,
                "n_samples": int(len(sub)),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
            }
        )
    return rows


def _per_class_rows(
    record: pd.DataFrame,
    labels: list[str],
    *,
    model: str,
    model_version: str,
) -> list[dict[str, Any]]:
    """Per-class precision/recall/f1 mean+std across repeats (record level)."""
    repeats = sorted(int(value) for value in record["repeat"].unique())
    rows: list[dict[str, Any]] = []
    for label in labels:
        precision: list[float] = []
        recall: list[float] = []
        f1: list[float] = []
        support: list[int] = []
        for repeat in repeats:
            sub = record[record["repeat"] == repeat]
            metrics = compute_metrics(
                sub["y_true"].to_numpy(), sub["y_pred"].to_numpy(), labels=labels
            )
            per_class = metrics["per_class"][label]
            precision.append(per_class["precision"])
            recall.append(per_class["recall"])
            f1.append(per_class["f1"])
            support.append(per_class["support"])
        rows.append(
            {
                "model": model,
                "model_version": model_version,
                "label": label,
                "precision_mean": float(np.mean(precision)),
                "precision_std": float(np.std(precision)),
                "recall_mean": float(np.mean(recall)),
                "recall_std": float(np.std(recall)),
                "f1_mean": float(np.mean(f1)),
                "f1_std": float(np.std(f1)),
                "support_mean": float(np.mean(support)),
                "support_std": float(np.std(support)),
            }
        )
    return rows


def _render_confusion_figure(
    confusion_by_model: dict[str, tuple[str, np.ndarray]],
    labels: list[str],
    *,
    output_path: Path,
) -> None:
    models = list(confusion_by_model)
    cols = min(len(models), 4)
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(3.4 * cols, 3.2 * rows),
        constrained_layout=True,
    )
    axes_flat = np.asarray(axes).reshape(-1)
    for index, (model, (version, confusion)) in enumerate(confusion_by_model.items()):
        ax = axes_flat[index]
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
        ax.set_title(f"{model} ({version})", fontsize=9)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    for index in range(len(models), rows * cols):
        axes_flat[index].axis("off")
    fig.suptitle(
        "PoPu P5.1 — record-level confusion matrices (pooled across 3 repeats, display only)",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)


def _render_stability_figure(
    summary_rows: list[dict[str, Any]],
    *,
    output_path: Path,
) -> None:
    rows = [row for row in summary_rows if row["status"] == "OK"]
    models = [str(row["model"]) for row in rows]
    means = [float(row["record_macro_f1_mean"]) for row in rows]
    stds = [float(row["record_macro_f1_std"]) for row in rows]
    order = np.argsort(means)
    models_sorted = [models[index] for index in order]
    means_sorted = [means[index] for index in order]
    stds_sorted = [stds[index] for index in order]

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    x_positions = np.arange(len(models_sorted))
    ax.bar(x_positions, means_sorted, yerr=stds_sorted, capsize=4, color="#4C72B0")
    ax.axhline(1.0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_xticks(x_positions, models_sorted, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("record-level macro-F1 (mean across 3 repeats)")
    ax.set_title("PoPu P5.1 — record macro-F1 mean and repeat std per candidate")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)


def _render_worst_subject_figure(
    summary_rows: list[dict[str, Any]],
    *,
    output_path: Path,
) -> None:
    rows = [row for row in summary_rows if row["status"] == "OK"]
    models = [str(row["model"]) for row in rows]
    accuracies = [float(row["worst_subject_accuracy_mean"]) for row in rows]
    subject_ids = [str(row["worst_subject_id"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    x_positions = np.arange(len(models))
    colors = [
        "#C44E52" if accuracy < 0.9 else "#4C72B0" for accuracy in accuracies
    ]
    ax.bar(x_positions, accuracies, color=colors)
    ax.axhline(0.9, color="gray", linewidth=0.8, linestyle="--", label="0.9 (P5 failing threshold)")
    for x_position, accuracy, subject_id in zip(x_positions, accuracies, subject_ids):
        ax.text(x_position, accuracy + 0.01, f"subj {subject_id}", ha="center", fontsize=8)
    ax.set_xticks(x_positions, models, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("worst-subject record accuracy (mean across repeats)")
    ax.set_title("PoPu P5.1 — worst subject accuracy per candidate")
    ax.legend(fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor="white")
    plt.close(fig)


def _evaluate_on_features(
    registered: Any,
    x_subset: np.ndarray,
    y: np.ndarray,
    groups: Sequence[str],
    sample_ids: Sequence[str],
    folds: Sequence[Any],
    record_map: dict[str, str],
    labels: list[str],
) -> dict[str, Any]:
    """Run grouped OOF + record aggregation on one feature subset."""
    oof = evaluate_grouped_oof(
        registered,
        x_subset,
        y,
        groups,
        folds,
        sample_ids=sample_ids,
        labels=labels,
    )
    record = _aggregate_records(oof.predictions, record_map, labels)
    per_repeat = _repeat_metrics(record, labels)
    reduced = reduce_repeat_metrics(per_repeat)
    return {
        "record_macro_f1_mean": reduced["macro_f1"]["mean"],
        "record_macro_f1_std": reduced["macro_f1"]["std"],
        "record_balanced_acc_mean": reduced["balanced_accuracy"]["mean"],
        "record_balanced_acc_std": reduced["balanced_accuracy"]["std"],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _read_json(_project_path(args.config))
    labels = list(config["labels"])
    protocol = config["evaluation_protocol"]
    n_splits = int(protocol["n_splits"])
    seeds = [int(seed) for seed in protocol["seeds"]]
    random_state = int(config["random_seed"])

    primary, feature_cols, feature_summary = _load_primary_cohort(config)
    x = primary[feature_cols].to_numpy(dtype=float)
    y = primary["posture"].to_numpy()
    groups = [str(subject) for subject in primary["subject_id"].to_numpy()]
    sample_ids = [str(sample) for sample in primary["sample_id"].to_numpy()]
    record_map = _build_record_id_map(primary)

    n_samples = len(x)
    n_subjects = len({str(subject) for subject in groups})
    n_records = len(set(record_map.values()))
    nan_rows = int(np.isnan(x).any(axis=1).sum())
    nan_cells = int(np.isnan(x).sum())

    folds = [
        generate_group_folds(groups, n_splits=n_splits, shuffle=True, seed=seed)
        for seed in seeds
    ]
    seed_by_repeat = {repeat: seed for repeat, seed in enumerate(seeds)}

    candidates = list(config["models"])
    candidate_names = [str(model["name"]) for model in candidates]
    n_candidates = len(candidates)
    estimated_fits = (
        n_candidates * n_splits * len(seeds)  # round-1 full CV
        + n_candidates  # smoke timing
        + 2 * (5 - 1) * n_splits * len(seeds)  # round-2 ablation (all group reused)
        + 1  # freeze
    )

    print("PoPu P5.1-B repeated subject-grouped candidate comparison")
    print("=" * 68)
    print(f"primary cohort:   n_samples={n_samples}  n_subjects={n_subjects}  "
          f"n_records={n_records}")
    print(f"posture counts:   {dict(zip(labels, [int((y == label).sum()) for label in labels]))}")
    print(f"NaN rows/cells:   {nan_rows} / {nan_cells}")
    print(f"candidates:       {n_candidates} ({candidate_names})")
    for repeat, (fold_set, seed) in enumerate(zip(folds, seeds)):
        val_sizes = [
            sum(1 for subject in groups if subject in {groups[i] for i in val_idx})
            for _, val_idx in fold_set.folds
        ]
        print(
            f"  repeat {repeat}: seed={seed}, folds={n_splits}, "
            f"val subjects per fold={val_sizes}"
        )
    print(f"fold fits:        {n_splits} folds x {len(seeds)} repeats = "
          f"{n_splits * len(seeds)} fits per candidate")
    print(f"estimated fits:   ~{estimated_fits} estimator fits "
          "(plus CalibratedLinearSVM's inner calibration cv)")
    print("=" * 68)

    outputs = {key: _project_path(value) for key, value in config["planned_outputs"].items()}

    smoke_times = {model["name"]: _smoke_timing(
        model, x, y, groups, folds, random_state=random_state
    ) for model in candidates}
    if args.smoke_only:
        print("One-fold smoke-timing pass (--smoke-only):")
        for model in candidates:
            timing = smoke_times[model["name"]]
            print(
                f"  {model['name']:<22} fit={timing['fit_time_seconds']:.2f}s "
                f"infer={timing['inference_time_seconds']:.2f}s "
                f"size={timing['model_size_bytes'] / 1024:.1f} KiB"
            )
        return 0

    max_full_run = config.get("runtime_budget", {}).get("max_full_run_seconds")
    n_fold_fits = n_splits * len(seeds)
    full_run_candidate: dict[str, dict[str, Any]] = {}
    for model in candidates:
        name = str(model["name"])
        timing = smoke_times[name]
        estimated = timing["fit_time_seconds"] * n_fold_fits
        if max_full_run is not None and estimated > float(max_full_run):
            full_run_candidate[name] = {
                "status": "SKIPPED",
                "skip_reason": (
                    f"estimated full run {estimated:.0f}s exceeds runtime_budget "
                    f"{float(max_full_run):.0f}s"
                ),
            }
            continue
        full_run_candidate[name] = {
            "status": "OK",
            "skip_reason": "",
        }

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    per_subject_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    confusion_by_model: dict[str, tuple[str, np.ndarray]] = {}
    all_oof_frames: list[pd.DataFrame] = []
    record_metrics: dict[str, dict[str, Any]] = {}

    for model in candidates:
        name = str(model["name"])
        version = str(model["version"])
        role = str(model.get("role", "candidate"))
        timing = smoke_times[name]
        state = full_run_candidate[name]

        summary_row: dict[str, Any] = {
            "model": name,
            "model_version": version,
            "role": role,
            "status": state["status"],
            "skip_reason": state["skip_reason"],
            "n_snapshot_samples": "",
            "n_records_per_repeat": "",
            "n_subjects": "",
            "snapshot_accuracy_mean": "",
            "snapshot_accuracy_std": "",
            "snapshot_balanced_acc_mean": "",
            "snapshot_balanced_acc_std": "",
            "snapshot_macro_f1_mean": "",
            "snapshot_macro_f1_std": "",
            "record_accuracy_mean": "",
            "record_accuracy_std": "",
            "record_balanced_acc_mean": "",
            "record_balanced_acc_std": "",
            "record_macro_f1_mean": "",
            "record_macro_f1_std": "",
            "worst_subject_id": "",
            "worst_subject_accuracy_mean": "",
            "worst_subject_macro_f1_mean": "",
            "fold_macro_f1_mean": "",
            "fold_macro_f1_std": "",
            "fold_accuracy_mean": "",
            "fold_accuracy_std": "",
            "fit_time_seconds": round(timing["fit_time_seconds"], 6),
            "inference_time_seconds": round(timing["inference_time_seconds"], 6),
            "model_size_bytes": int(timing["model_size_bytes"]),
        }

        if state["status"] != "OK":
            summary_rows.append(summary_row)
            continue

        registered = build_model(model, random_state=random_state)
        oof = evaluate_grouped_oof(
            registered,
            x,
            y,
            groups,
            folds,
            sample_ids=sample_ids,
            labels=labels,
        )
        oof_frame = oof.predictions
        oof_frame = oof_frame.copy()
        oof_frame.insert(0, "model", name)
        oof_frame.insert(1, "model_version", version)
        oof_frame.insert(2, "role", role)
        oof_frame["record_id"] = oof_frame["sample_id"].map(record_map)
        all_oof_frames.append(oof_frame)

        record = _aggregate_records(oof.predictions, record_map, labels)
        record_metrics[name] = _repeat_metrics(record, labels)
        reduced_record = reduce_repeat_metrics(record_metrics[name])
        reduced_snapshot = reduce_repeat_metrics(
            snapshot_metrics_per_repeat(oof)
        )

        subjects = repeated_subject_metrics(
            record, labels=labels, repeat_col="repeat", group_col="group_id"
        )
        worst = next(row for row in subjects if row["is_worst"])

        fold_macro = [fold["macro_f1"] for fold in oof.per_fold_metrics]
        fold_acc = [fold["accuracy"] for fold in oof.per_fold_metrics]

        summary_row.update(
            {
                "n_snapshot_samples": int(len(oof_frame)),
                "n_records_per_repeat": int(len(record) // len(seeds)),
                "n_subjects": n_subjects,
                "snapshot_accuracy_mean": round(reduced_snapshot["accuracy"]["mean"], 6),
                "snapshot_accuracy_std": round(reduced_snapshot["accuracy"]["std"], 6),
                "snapshot_balanced_acc_mean": round(reduced_snapshot["balanced_accuracy"]["mean"], 6),
                "snapshot_balanced_acc_std": round(reduced_snapshot["balanced_accuracy"]["std"], 6),
                "snapshot_macro_f1_mean": round(reduced_snapshot["macro_f1"]["mean"], 6),
                "snapshot_macro_f1_std": round(reduced_snapshot["macro_f1"]["std"], 6),
                "record_accuracy_mean": round(reduced_record["accuracy"]["mean"], 6),
                "record_accuracy_std": round(reduced_record["accuracy"]["std"], 6),
                "record_balanced_acc_mean": round(reduced_record["balanced_accuracy"]["mean"], 6),
                "record_balanced_acc_std": round(reduced_record["balanced_accuracy"]["std"], 6),
                "record_macro_f1_mean": round(reduced_record["macro_f1"]["mean"], 6),
                "record_macro_f1_std": round(reduced_record["macro_f1"]["std"], 6),
                "worst_subject_id": worst["subject_id"],
                "worst_subject_accuracy_mean": round(worst["accuracy_mean"], 6),
                "worst_subject_macro_f1_mean": round(worst["macro_f1_mean"], 6),
                "fold_macro_f1_mean": round(float(np.mean(fold_macro)), 6),
                "fold_macro_f1_std": round(float(np.std(fold_macro)), 6),
                "fold_accuracy_mean": round(float(np.mean(fold_acc)), 6),
                "fold_accuracy_std": round(float(np.std(fold_acc)), 6),
            }
        )
        summary_rows.append(summary_row)

        per_class_rows.extend(
            _per_class_rows(record, labels, model=name, model_version=version)
        )
        per_subject_rows.extend(
            [
                {**{"model": name, "model_version": version}, **row}
                for row in subjects
            ]
        )
        for fold in oof.per_fold_metrics:
            fold_rows.append(
                {
                    "model": name,
                    "model_version": version,
                    "repeat": fold["repeat"],
                    "seed": seed_by_repeat[fold["repeat"]],
                    "local_fold": fold["local_fold"],
                    "fold_id": fold["fold_id"],
                    "n_samples": fold["n_samples"],
                    "accuracy": round(fold["accuracy"], 6),
                    "balanced_accuracy": round(fold["balanced_accuracy"], 6),
                    "macro_f1": round(fold["macro_f1"], 6),
                }
            )
        for _, row in record.iterrows():
            record_rows.append(
                {
                    "model": name,
                    "model_version": version,
                    "repeat": row["repeat"],
                    "record_id": row["record_id"],
                    "subject_id": row["group_id"],
                    "y_true": row["y_true"],
                    "y_pred": row["y_pred"],
                    "confidence": round(float(row["confidence"]), 6),
                    "n_snapshots": row["n_snapshots"],
                    **{
                        f"{PROBA_PREFIX}{label}": round(float(row[f"{PROBA_PREFIX}{label}"]), 6)
                        for label in labels
                    },
                }
            )
        pooled_confusion = confusion_matrix(
            record["y_true"].to_numpy(),
            record["y_pred"].to_numpy(),
            labels=labels,
        )
        confusion_by_model[name] = (version, pooled_confusion)
        for true_index, true_label in enumerate(labels):
            for pred_index, pred_label in enumerate(labels):
                confusion_rows.append(
                    {
                        "model": name,
                        "model_version": version,
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "count": int(pooled_confusion[true_index, pred_index]),
                    }
                )
        print(
            f"  {name:<22} record_macro_f1={reduced_record['macro_f1']['mean']:.4f} "
            f"(std={reduced_record['macro_f1']['std']:.4f}) "
            f"balanced_acc={reduced_record['balanced_accuracy']['mean']:.4f} "
            f"worst_subj_acc={worst['accuracy_mean']:.4f}"
        )

    # --- Record-level aggregation CSV (pooled per candidate across repeats) -----
    _write_csv(record_rows, RECORD_COLUMNS, outputs["record_level"])
    _write_csv(summary_rows, SUMMARY_COLUMNS, outputs["model_summary"])
    _write_csv(per_class_rows, PER_CLASS_COLUMNS, outputs["per_class"])
    _write_csv(per_subject_rows, PER_SUBJECT_COLUMNS, outputs["per_subject"])
    _write_csv(fold_rows, FOLD_REPEAT_COLUMNS, outputs["fold_repeat"])
    _write_csv(confusion_rows, CONFUSION_COLUMNS, outputs["confusion"])

    oof_all = pd.concat(all_oof_frames, ignore_index=True)
    oof_path = outputs["oof_predictions"]
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof_all.to_csv(oof_path, index=False)

    # --- Candidate selection (record-level primary caliber) ----------------------
    selectable = [row for row in summary_rows if row["status"] == "OK"]
    selection_cfg = config["model_selection"]
    winner = select_best_candidate(
        selectable,
        criterion=selection_cfg["criterion"],
        tie_break=selection_cfg["tie_break"],
        worst_subject_criterion=selection_cfg["worst_subject_criterion"],
        complexity_order=selection_cfg["complexity_priority"],
        exclude=list(selection_cfg.get("exclude", [])),
        margin=float(selection_cfg.get("margin", 0.0)),
    )
    top_primary = max(float(row[selection_cfg["criterion"]]) for row in selectable)
    winner_row = next(row for row in selectable if row["model"] == winner)
    within_margin = [
        row["model"]
        for row in selectable
        if top_primary - float(row[selection_cfg["criterion"]]) <= float(selection_cfg.get("margin", 0.0))
    ]
    print(f"selected candidate: {winner} (record_macro_f1_mean="
          f"{winner_row['record_macro_f1_mean']:.4f})")

    # --- Feature ablation: top-2 round-1 candidates x 5 groups --------------------
    ranked = sorted(
        selectable,
        key=lambda row: float(row["record_macro_f1_mean"]),
        reverse=True,
    )
    top2 = [row["model"] for row in ranked[:2]]
    # NOTE: must not shadow the outer subject ``groups`` — the ablation
    # evaluator below still needs the per-sample subject groups for folds.
    feature_groups = feature_group_columns(feature_cols)
    ablation_sets: dict[str, list[str]] = {
        "intensity_only": feature_groups["intensity"],
        "mask_geometry_only": feature_groups["geometry"],
        "grid_zones_only": feature_groups["zones"],
        "intensity_geometry": feature_groups["intensity"] + feature_groups["geometry"],
        "all": feature_cols,
    }
    feature_index = {column: index for index, column in enumerate(feature_cols)}

    ablation_rows: list[dict[str, Any]] = []
    for model_cfg in config["models"]:
        name = str(model_cfg["name"])
        if name not in top2:
            continue
        registered = build_model(model_cfg, random_state=random_state)
        for group_name, columns in ablation_sets.items():
            indices = [feature_index[column] for column in columns]
            if group_name == "all":
                base = record_metrics[name]
                reduced = reduce_repeat_metrics(base)
                result = {
                    "record_macro_f1_mean": reduced["macro_f1"]["mean"],
                    "record_macro_f1_std": reduced["macro_f1"]["std"],
                    "record_balanced_acc_mean": reduced["balanced_accuracy"]["mean"],
                    "record_balanced_acc_std": reduced["balanced_accuracy"]["std"],
                }
                reuse = True
            else:
                result = _evaluate_on_features(
                    registered,
                    x[:, indices],
                    y,
                    groups,
                    sample_ids,
                    folds,
                    record_map,
                    labels,
                )
                reuse = False
            ablation_rows.append(
                {
                    "model": name,
                    "model_version": str(model_cfg["version"]),
                    "feature_group": group_name,
                    "n_features": len(columns),
                    "record_macro_f1_mean": round(result["record_macro_f1_mean"], 6),
                    "record_macro_f1_std": round(result["record_macro_f1_std"], 6),
                    "record_balanced_acc_mean": round(result["record_balanced_acc_mean"], 6),
                    "record_balanced_acc_std": round(result["record_balanced_acc_std"], 6),
                    "reused_from_round1": str(reuse).lower(),
                }
            )
            print(
                f"  ablation {name:<18} {group_name:<20} n={len(columns):<3} "
                f"record_macro_f1={result['record_macro_f1_mean']:.4f} "
                f"(std={result['record_macro_f1_std']:.4f})"
            )
    _write_csv(ablation_rows, ABLATION_COLUMNS, outputs["feature_ablation"])

    # --- Figures -----------------------------------------------------------------
    _render_confusion_figure(confusion_by_model, labels, output_path=outputs["confusion_figure"])
    _render_stability_figure(summary_rows, output_path=outputs["stability_figure"])
    _render_worst_subject_figure(summary_rows, output_path=outputs["worst_subject_figure"])

    # --- Freeze: refit the winner on the full primary cohort ----------------------
    # The artifact is wrapped in FrozenClassOrderClassifier so the frozen model's
    # ``classes_`` equals the frozen label order (sklearn sorts classes_ at fit
    # time otherwise, which would make proba columns disagree with metadata).
    winner_cfg = next(model for model in config["models"] if model["name"] == winner)
    inner_estimator = build_model(winner_cfg, random_state=random_state).estimator
    final_estimator = FrozenClassOrderClassifier(inner_estimator, class_order=labels)
    final_estimator.fit(x, y)

    freeze_cfg = config["candidate_freeze"]
    model_path = outputs["candidate_model"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_estimator, model_path)
    model_size_bytes = Path(model_path).stat().st_size

    smoke = _smoke_test_loaded_model(model_path, x, labels, probe_size=200)
    freeze_name = str(freeze_cfg["name"])
    freeze_metadata = {
        "name": freeze_name,
        "label": freeze_cfg["label"],
        "stage": "P5.1/R5.1",
        "baseline_version": config["baseline_version"],
        "task": config["task"],
        "labels": labels,
        "feature_schema": {
            "version": feature_summary.get("feature_schema_version"),
            "count": len(feature_cols),
            "columns": feature_cols,
        },
        "model": {
            "name": winner_cfg["name"],
            "version": winner_cfg["version"],
            "estimator": winner_cfg["estimator"],
            "params": winner_cfg.get("params", {}),
            "preprocessing": winner_cfg.get("preprocessing", []),
            "role": winner_cfg.get("role", "candidate"),
        },
        "training_data": {
            "feature_table": str(config["feature_schema"]["feature_table"]),
            "cohort": "primary",
            "n_snapshots": n_samples,
            "n_subjects": n_subjects,
            "n_records": n_records,
            "posture_counts": {
                label: int((y == label).sum()) for label in labels
            },
            "nan_rows": nan_rows,
            "nan_cells": nan_cells,
        },
        "random_seed": random_state,
        "selection": {
            "criterion": selection_cfg["criterion"],
            "margin": float(selection_cfg.get("margin", 0.0)),
            "winner": winner,
            "winner_record_macro_f1_mean": winner_row["record_macro_f1_mean"],
            "winner_record_balanced_acc_mean": winner_row["record_balanced_acc_mean"],
            "worst_subject_id": winner_row["worst_subject_id"],
            "worst_subject_accuracy_mean": winner_row["worst_subject_accuracy_mean"],
            "within_margin": within_margin,
        },
        "known_limitations": [
            "research-stage result on public PoPu data only; NOT a product model and NOT externally validated",
            "all 60 PoPu subjects were used as dev/OOF in P5.1 or P5 v0.1; there is no re-nameable 'never-seen' PoPu test",
            "external confirmation is deferred to SLP2022/PressurePose/TIP and future self-developed synchronized data; those datasets must be evaluated separately and never row-joined with PoPu",
            "a record whose snapshots disagree on the true label or subject is rejected by aggregation (none present in the primary cohort)",
        ],
        "model_file": str(model_path),
        "model_size_bytes": model_size_bytes,
        "smoke_test": smoke,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = outputs["candidate_metadata"]
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(freeze_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"frozen candidate: {freeze_name} -> {model_path} "
        f"({model_size_bytes / 1024:.1f} KiB) smoke={smoke['status']}"
    )

    # --- Machine-readable summary JSON -------------------------------------------
    summary: dict[str, Any] = {
        "dataset": "PoPu",
        "sensor_layer": "tactilus",
        "stage": "P5.1/R5.1",
        "baseline_version": config["baseline_version"],
        "task": config["task"],
        "labels": labels,
        "feature_schema_version": feature_summary.get("feature_schema_version"),
        "feature_count": len(feature_cols),
        "group_key": "subject_id",
        "random_seed": random_state,
        "evaluation_protocol": protocol,
        "model_selection": selection_cfg,
        "primary_cohort": {
            "n_samples": n_samples,
            "n_subjects": n_subjects,
            "n_records": n_records,
            "posture_counts": {label: int((y == label).sum()) for label in labels},
            "nan_rows": nan_rows,
            "nan_cells": nan_cells,
        },
        "candidates": [
            {key: row[key] for key in SUMMARY_COLUMNS if key in row}
            for row in summary_rows
        ],
        "selection": {
            "winner": winner,
            "winner_model_version": winner_row["model_version"],
            "criterion": selection_cfg["criterion"],
            "top_primary_score": top_primary,
            "margin": float(selection_cfg.get("margin", 0.0)),
            "within_margin": within_margin,
        },
        "feature_ablation": ablation_rows,
        "feature_ablation_top2": top2,
        "worst_subject": {
            "model": winner_row["model"],
            "subject_id": winner_row["worst_subject_id"],
            "accuracy_mean": winner_row["worst_subject_accuracy_mean"],
            "macro_f1_mean": winner_row["worst_subject_macro_f1_mean"],
        },
        "freeze": {
            "name": freeze_name,
            "model_file": str(model_path),
            "model_size_bytes": model_size_bytes,
            "smoke_test": smoke,
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
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"summary written: {summary_path}")
    return 0


def _smoke_test_loaded_model(
    model_path: Path,
    x: np.ndarray,
    labels: list[str],
    *,
    probe_size: int = 200,
) -> dict[str, Any]:
    """Independently load the frozen model file and probe predict/predict_proba."""
    loaded = joblib.load(model_path)
    probe = x[:probe_size]
    predicted = np.asarray(loaded.predict(probe))
    proba = np.asarray(loaded.predict_proba(probe))
    classes_match = list(loaded.classes_) == list(labels)
    ok = (
        predicted.shape == (probe_size,)
        and proba.shape == (probe_size, len(labels))
        and bool(np.isfinite(proba).all())
        and bool(np.allclose(proba.sum(axis=1), 1.0, atol=1e-4))
        and classes_match
    )
    return {
        "status": "OK" if ok else "FAILED",
        "n_probed": probe_size,
        "pred_shape": list(predicted.shape),
        "proba_shape": list(proba.shape),
        "finite": bool(np.isfinite(proba).all()),
        "rows_sum_to_1": bool(np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)),
        "classes_order_matches": classes_match,
    }


# Output column order for the record-level aggregation CSV.
RECORD_COLUMNS = (
    "model",
    "model_version",
    "repeat",
    "record_id",
    "subject_id",
    "y_true",
    "y_pred",
    "confidence",
    "n_snapshots",
) + tuple(f"{PROBA_PREFIX}{label}" for label in ("empty", "supine", "prone", "left", "right"))


if __name__ == "__main__":
    raise SystemExit(main())
