"""Governed P5.2-C Full neural comparison runner.

This module implements the protocol frozen in ``popu_neural_full_v0.1.json``:
three neural candidates, repeated subject-grouped outer CV, an inner
subject-grouped epoch-selection stage, refit on every complete outer-training
fold, record-level aggregation, calibration diagnostics, and comparison with
the immutable P5.1 calibrated-linear-SVM evidence.

The runner is deliberately fold-transactional.  Every completed
``repeat/fold/model`` unit owns a ``complete.json`` whose referenced artifacts
are content-hashed.  A resumed invocation skips only units whose marker and
hashes validate; partial units are recomputed without touching completed ones.
It never freezes the eventual winner: Reviewer acceptance remains a separate
governance step.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from topper_perception.experiments.artifacts import atomic_write_json, sha256_hex
from topper_perception.io.popu_inventory import iter_tactilus_record_paths, resolve_tactilus_root
from topper_perception.neural.checkpoint import (
    build_payload,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint,
)
from topper_perception.neural.data import (
    FROZEN_LABELS,
    LABEL_TO_INDEX,
    MatrixNormalizer,
    build_labeled_samples,
    horizontal_flip,
    to_model_input,
)
from topper_perception.neural.dataset import PressureDataset, build_dataloader
from topper_perception.neural.early_stopping import EarlyStopper
from topper_perception.neural.full_protocol import (
    BATCH_SIZE,
    CALIBRATION_ECE_BINS,
    DATA_BOUNDARY,
    FROZEN_SVM_REFERENCE,
    FROZEN_SVM_REFERENCE_ARTIFACTS,
    FullCandidateResult,
    MAX_CUDA_MB,
    MAX_EPOCHS,
    MAX_TOTAL_TRAIN_SECONDS,
    MIN_EPOCHS,
    MONITOR,
    NEURAL_CANDIDATES,
    N_REPEATS,
    PATIENCE,
    SEED,
    record_ece,
    record_multiclass_brier,
    record_multiclass_nll,
    select_full_winner,
    validate_full_config,
    validate_full_data_boundary,
)
from topper_perception.neural.full_splits import (
    build_full_fold_manifest,
    validate_full_fold_manifest,
)
from topper_perception.neural.metrics import ClassificationMetrics, compute_classification_metrics
from topper_perception.neural.mini import (
    _load_quality_manifest,
    _record_sample_id,
    _resolve_data_root,
    _resolve_manifest_path,
    _verify_quality_manifest,
)
from topper_perception.neural.models import build_model, count_parameters
from topper_perception.neural.training import (
    evaluate,
    make_criterion,
    make_optimizer,
    predict,
    resolve_device,
    set_seed,
    train_epoch,
)


def _project_root() -> Path:
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return here


PROJECT_ROOT = _project_root()
FROZEN_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "popu_neural_full_v0.1.json"
PROBA_COLUMNS = tuple(f"proba__{label}" for label in FROZEN_LABELS)
SVM_SERIALIZATION_MAX_ROW_SUM_DRIFT = 5e-6
SNAPSHOT_COLUMNS = (
    "model",
    "repeat",
    "outer_seed",
    "local_fold",
    "sample_id",
    "record_id",
    "subject_id",
    "y_true",
    "y_pred",
    "confidence",
    *PROBA_COLUMNS,
)
RECORD_COLUMNS = (
    "model",
    "repeat",
    "outer_seed",
    "local_fold",
    "record_id",
    "subject_id",
    "y_true",
    "y_pred",
    "confidence",
    "n_snapshots",
    *PROBA_COLUMNS,
)


@dataclass(frozen=True, slots=True)
class FullCohort:
    """Full ACCEPT-only cohort loaded once and shared by all folds/models."""

    matrices: np.ndarray
    labels: np.ndarray
    sample_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]
    manifest_integrity: Mapping[str, Any]
    n_records_excluded: int

    @property
    def subjects(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.subject_ids)))


def _load_frozen_config() -> dict[str, Any]:
    config = json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    validate_full_config(config)
    return config


def _require_frozen_invocation(parameters: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Reject a runner invocation that differs from the accepted Full config."""
    config = _load_frozen_config()
    if int(seed) != SEED:
        raise ValueError(f"Full runner seed {seed} differs from frozen seed {SEED}.")
    if dict(parameters) != dict(config["parameters"]):
        raise ValueError(
            "Full runner parameters differ from the accepted frozen config; "
            "create a new protocol/config instead of overriding fields at runtime."
        )
    return config


def _resolve_project_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def verify_svm_reference_artifacts() -> list[dict[str, Any]]:
    """Verify all six immutable P5.1 reference artifacts before training."""
    verified: list[dict[str, Any]] = []
    for item in FROZEN_SVM_REFERENCE_ARTIFACTS:
        path = _resolve_project_path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(f"Frozen P5.1 SVM reference artifact missing: {path}")
        size = path.stat().st_size
        if size != int(item["size_bytes"]):
            raise ValueError(
                f"Frozen P5.1 artifact size mismatch for {path}: "
                f"expected {item['size_bytes']}, got {size}."
            )
        actual = sha256_hex(path)
        if actual != str(item["sha256"]).lower():
            raise ValueError(
                f"Frozen P5.1 artifact SHA-256 mismatch for {path}: "
                f"expected {item['sha256']}, got {actual}."
            )
        verified.append({"path": str(path), "size_bytes": size, "sha256": actual})
    return verified


def load_full_cohort(parameters: Mapping[str, Any]) -> FullCohort:
    """Load and validate the complete primary/ACCEPT PoPu cohort exactly once."""
    data_root = _resolve_data_root(parameters)
    manifest_path = _resolve_manifest_path(parameters)
    manifest_integrity = _verify_quality_manifest(manifest_path, parameters)
    statuses = _load_quality_manifest(manifest_path)
    tactilus_root = resolve_tactilus_root(data_root)
    record_paths = list(iter_tactilus_record_paths(data_root))
    if not record_paths:
        raise ValueError(f"No Tactilus records found under {data_root}.")

    accepted_paths: list[Path] = []
    missing: list[str] = []
    excluded = 0
    for path in record_paths:
        sample_id = _record_sample_id(path, tactilus_root)
        status = statuses.get(sample_id)
        if status is None:
            missing.append(sample_id)
        elif status == "ACCEPT":
            accepted_paths.append(path)
        else:
            excluded += 1
    if missing:
        raise ValueError(
            "Records missing from the frozen P2 quality manifest; first entries: "
            f"{missing[:5]}."
        )

    samples = build_labeled_samples(accepted_paths, tactilus_root=data_root)
    matrices, labels = to_model_input(samples)
    sample_ids = tuple(sample.sample_id for sample in samples)
    record_ids = tuple(sample.record_id for sample in samples)
    subject_ids = tuple(str(sample.subject_id) for sample in samples)

    counts: dict[str, int] = {}
    for record_id in record_ids:
        counts[record_id] = counts.get(record_id, 0) + 1
    if not counts or set(counts.values()) != {DATA_BOUNDARY["snapshots_per_record"]}:
        bad = [(record_id, count) for record_id, count in counts.items() if count != 10]
        raise ValueError(f"Every Full record must contain exactly 10 snapshots; first bad: {bad[:5]}.")

    validate_full_data_boundary(
        n_subjects=len(set(subject_ids)),
        n_records=len(counts),
        n_snapshots=len(samples),
        snapshots_per_record=next(iter(counts.values())),
    )
    if len(accepted_paths) != len(counts):
        raise ValueError(
            f"Accepted path count {len(accepted_paths)} differs from labeled record count {len(counts)}."
        )
    return FullCohort(
        matrices=matrices,
        labels=labels,
        sample_ids=sample_ids,
        record_ids=record_ids,
        subject_ids=subject_ids,
        manifest_integrity=manifest_integrity,
        n_records_excluded=excluded,
    )


def _indices_for_subjects(cohort: FullCohort, subjects: Sequence[str]) -> np.ndarray:
    array = np.asarray(cohort.subject_ids, dtype=object)
    indices = np.flatnonzero(np.isin(array, np.asarray(list(subjects), dtype=object)))
    if indices.size == 0:
        raise ValueError(f"Subject subset is empty: {list(subjects)}")
    return indices


def _metadata(cohort: FullCohort, indices: np.ndarray) -> tuple[list[str], list[str], list[str]]:
    return (
        [cohort.sample_ids[int(i)] for i in indices],
        [cohort.record_ids[int(i)] for i in indices],
        [cohort.subject_ids[int(i)] for i in indices],
    )


def _make_dataset(
    matrices: np.ndarray,
    labels: np.ndarray,
    metadata: tuple[Sequence[str], Sequence[str], Sequence[str]],
) -> PressureDataset:
    return PressureDataset(
        matrices,
        labels,
        sample_ids=metadata[0],
        record_ids=metadata[1],
        subject_ids=metadata[2],
    )


def _prepare_train_eval_loaders(
    cohort: FullCohort,
    train_subjects: Sequence[str],
    eval_subjects: Sequence[str],
    *,
    batch_size: int = BATCH_SIZE,
) -> tuple[Any, Any, MatrixNormalizer, int]:
    train_idx = _indices_for_subjects(cohort, train_subjects)
    eval_idx = _indices_for_subjects(cohort, eval_subjects)
    if set(train_subjects) & set(eval_subjects):
        raise ValueError("Train/evaluation subjects overlap.")

    normalizer = MatrixNormalizer().fit(cohort.matrices[train_idx])
    x_train = normalizer.transform(cohort.matrices[train_idx])
    y_train = cohort.labels[train_idx].copy()
    train_meta = _metadata(cohort, train_idx)
    x_eval = normalizer.transform(cohort.matrices[eval_idx])
    y_eval = cohort.labels[eval_idx].copy()
    eval_meta = _metadata(cohort, eval_idx)

    flipped, flipped_labels = horizontal_flip(x_train, y_train)
    assert flipped_labels is not None
    x_train = np.concatenate([x_train, flipped], axis=0)
    y_train = np.concatenate([y_train, flipped_labels], axis=0)
    train_meta = (
        list(train_meta[0]) + [f"{value}#flip" for value in train_meta[0]],
        list(train_meta[1]) + list(train_meta[1]),
        list(train_meta[2]) + list(train_meta[2]),
    )

    train_loader = build_dataloader(
        _make_dataset(x_train, y_train, train_meta), batch_size=batch_size, shuffle=True
    )
    eval_loader = build_dataloader(
        _make_dataset(x_eval, y_eval, eval_meta), batch_size=batch_size, shuffle=False
    )
    return train_loader, eval_loader, normalizer, int(len(flipped))


def _metric_dict(metrics: ClassificationMetrics) -> dict[str, Any]:
    return metrics.as_dict()


def _require_finite(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite {label}: {value!r}.")
    return value


def _stage_a_select_epoch(
    cohort: FullCohort,
    fold: Mapping[str, Any],
    model_config: Mapping[str, Any],
    device: torch.device,
    candidate_dir: Path,
    *,
    amp_enabled: bool,
    lr: float,
    weight_decay: float,
) -> dict[str, Any]:
    seed = int(fold["stage_a_train_seed"])
    set_seed(seed)
    train_loader, val_loader, normalizer, augmented = _prepare_train_eval_loaders(
        cohort, fold["inner_train_subjects"], fold["inner_validation_subjects"]
    )
    model = build_model(model_config).to(device)
    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay)
    criterion = make_criterion()
    stopper = EarlyStopper(
        monitor=MONITOR, mode="min", patience=PATIENCE, min_delta=0.0, min_epochs=MIN_EPOCHS
    )
    best_path = candidate_dir / "stage_a_best.pt"
    history: list[dict[str, Any]] = []
    best_epoch = 0
    best_loss = float("inf")
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        epoch_started = time.perf_counter()
        train_info = train_epoch(
            model, train_loader, optimizer, criterion, device, amp_enabled=amp_enabled
        )
        val = evaluate(model, val_loader, criterion, device)
        train_loss = _require_finite(train_info["loss"], "stage_a train_loss")
        val_loss = _require_finite(val.loss, "stage_a val_loss")
        metrics = compute_classification_metrics(val.labels, val.predictions, FROZEN_LABELS)
        step = stopper.step(epoch, val_loss)
        if step.is_best:
            best_epoch = epoch
            best_loss = val_loss
            save_checkpoint(
                best_path,
                build_payload(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    model_config=model_config,
                    normalization={"mean": normalizer.mean_, "std": normalizer.std_},
                    seed=seed,
                    metrics={"val_loss": val_loss, "val_macro_f1": metrics.macro_f1},
                ),
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": metrics.macro_f1,
                "val_balanced_accuracy": metrics.balanced_accuracy,
                "seconds": round(time.perf_counter() - epoch_started, 6),
                "is_best": bool(step.is_best),
            }
        )
        if step.should_stop:
            break
    if best_epoch < 1 or not best_path.is_file():
        raise RuntimeError("Stage A did not produce a valid best epoch/checkpoint.")
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "actual_epochs": len(history),
        "augmented_train_samples": augmented,
        "seconds": round(time.perf_counter() - started, 6),
        "history": history,
        "checkpoint": str(best_path),
    }


def _write_csv_atomic(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row[column] for column in columns})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _prediction_rows(
    result: Any, model_name: str, fold: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(result.n_samples):
        probabilities = result.probabilities[index]
        prediction = int(result.predictions[index])
        row: dict[str, Any] = {
            "model": model_name,
            "repeat": int(fold["repeat"]),
            "outer_seed": int(fold["outer_seed"]),
            "local_fold": int(fold["local_fold"]),
            "sample_id": result.sample_ids[index],
            "record_id": result.record_ids[index],
            "subject_id": result.subject_ids[index],
            "y_true": FROZEN_LABELS[int(result.labels[index])],
            "y_pred": FROZEN_LABELS[prediction],
            "confidence": float(probabilities[prediction]),
        }
        row.update(
            {column: float(probabilities[i]) for i, column in enumerate(PROBA_COLUMNS)}
        )
        rows.append(row)
    return rows


def aggregate_record_rows(snapshot_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate exactly ten snapshot probabilities into each record row."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in snapshot_rows:
        groups.setdefault(str(row["record_id"]), []).append(row)
    rows: list[dict[str, Any]] = []
    for record_id in sorted(groups):
        group = groups[record_id]
        if len(group) != DATA_BOUNDARY["snapshots_per_record"]:
            raise ValueError(f"Record {record_id!r} has {len(group)} snapshots, expected 10.")
        subjects = {str(row["subject_id"]) for row in group}
        labels = {str(row["y_true"]) for row in group}
        if len(subjects) != 1 or len(labels) != 1:
            raise ValueError(f"Record {record_id!r} has conflicting subject/label provenance.")
        means = np.asarray(
            [[float(row[column]) for column in PROBA_COLUMNS] for row in group], dtype=float
        ).mean(axis=0)
        prediction = int(np.argmax(means))
        first = group[0]
        row = {
            "model": str(first["model"]),
            "repeat": int(first["repeat"]),
            "outer_seed": int(first["outer_seed"]),
            "local_fold": int(first["local_fold"]),
            "record_id": record_id,
            "subject_id": next(iter(subjects)),
            "y_true": next(iter(labels)),
            "y_pred": FROZEN_LABELS[prediction],
            "confidence": float(means[prediction]),
            "n_snapshots": len(group),
        }
        row.update({column: float(means[i]) for i, column in enumerate(PROBA_COLUMNS)})
        rows.append(row)
    return rows


def _labels_from_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    true = np.asarray([LABEL_TO_INDEX[str(row["y_true"])] for row in rows], dtype=np.int64)
    pred = np.asarray([LABEL_TO_INDEX[str(row["y_pred"])] for row in rows], dtype=np.int64)
    return true, pred


def _stage_b_refit_and_predict(
    cohort: FullCohort,
    fold: Mapping[str, Any],
    model_config: Mapping[str, Any],
    device: torch.device,
    candidate_dir: Path,
    *,
    best_epoch: int,
    amp_enabled: bool,
    lr: float,
    weight_decay: float,
) -> dict[str, Any]:
    seed = int(fold["stage_b_refit_seed"])
    set_seed(seed)
    train_loader, test_loader, normalizer, augmented = _prepare_train_eval_loaders(
        cohort, fold["outer_train_subjects"], fold["outer_test_subjects"]
    )
    model = build_model(model_config).to(device)
    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay)
    criterion = make_criterion()
    history: list[dict[str, Any]] = []
    checkpoint = candidate_dir / "stage_b_final.pt"
    started = time.perf_counter()
    for epoch in range(1, best_epoch + 1):
        epoch_started = time.perf_counter()
        info = train_epoch(
            model, train_loader, optimizer, criterion, device, amp_enabled=amp_enabled
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": _require_finite(info["loss"], "stage_b train_loss"),
                "seconds": round(time.perf_counter() - epoch_started, 6),
            }
        )
    train_seconds = round(time.perf_counter() - started, 6)
    save_checkpoint(
        checkpoint,
        build_payload(
            model=model,
            optimizer=optimizer,
            epoch=best_epoch,
            model_config=model_config,
            normalization={"mean": normalizer.mean_, "std": normalizer.std_},
            seed=seed,
            metrics={"train_loss": history[-1]["train_loss"]},
        ),
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    infer_started = time.perf_counter()
    test_result = predict(model, test_loader, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = round(time.perf_counter() - infer_started, 6)
    if test_result.n_samples == 0:
        raise RuntimeError("Outer-test inference returned no samples.")

    payload = load_checkpoint(checkpoint, map_location=device)
    validate_checkpoint(payload)
    reload_model = build_model(model_config).to(device)
    reload_model.load_state_dict(payload["model_state_dict"])
    reload_result = predict(reload_model, test_loader, device)
    reload_consistent = bool(
        np.array_equal(test_result.predictions, reload_result.predictions)
        and np.allclose(test_result.probabilities, reload_result.probabilities, rtol=1e-6, atol=1e-7)
    )
    if not reload_consistent:
        raise RuntimeError("Checkpoint reload predictions differ from in-memory predictions.")

    snapshot_rows = _prediction_rows(test_result, str(model_config["name"]), fold)
    record_rows = aggregate_record_rows(snapshot_rows)
    snapshot_metrics = compute_classification_metrics(
        test_result.labels, test_result.predictions, FROZEN_LABELS
    )
    record_true, record_pred = _labels_from_rows(record_rows)
    record_metrics = compute_classification_metrics(record_true, record_pred, FROZEN_LABELS)
    snapshot_path = candidate_dir / "snapshot_predictions.csv"
    record_path = candidate_dir / "record_predictions.csv"
    _write_csv_atomic(snapshot_path, SNAPSHOT_COLUMNS, snapshot_rows)
    _write_csv_atomic(record_path, RECORD_COLUMNS, record_rows)
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "inference_samples": test_result.n_samples,
        "augmented_train_samples": augmented,
        "history": history,
        "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_reload_consistent": reload_consistent,
        "param_count": count_parameters(model),
        "snapshot_metrics": _metric_dict(snapshot_metrics),
        "record_metrics": _metric_dict(record_metrics),
        "snapshot_predictions": str(snapshot_path),
        "record_predictions": str(record_path),
    }


def _candidate_dir(experiment_dir: Path, fold: Mapping[str, Any], model_name: str) -> Path:
    return (
        experiment_dir
        / "folds"
        / f"repeat_{int(fold['repeat'])}"
        / f"fold_{int(fold['local_fold'])}"
        / model_name
    )


def _relative_artifact(path: Path, experiment_dir: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(experiment_dir).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_hex(path),
    }


def validate_candidate_complete(
    candidate_dir: Path,
    experiment_dir: Path,
    *,
    model_name: str,
    repeat: int,
    local_fold: int,
    split_manifest_sha256: str,
) -> dict[str, Any] | None:
    """Return a verified completion marker, or ``None`` when work is incomplete."""
    marker_path = candidate_dir / "complete.json"
    if not marker_path.is_file():
        return None
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {
        "model": model_name,
        "repeat": repeat,
        "local_fold": local_fold,
        "split_manifest_sha256": split_manifest_sha256,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError(f"Completed candidate marker drift: {key}={marker.get(key)!r}.")
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"Completed candidate marker has no artifact inventory: {marker_path}")
    for item in artifacts:
        path = experiment_dir / str(item["path"])
        if not path.is_file():
            raise ValueError(f"Completed candidate artifact missing: {path}")
        if path.stat().st_size != int(item["size_bytes"]) or sha256_hex(path) != item["sha256"]:
            raise ValueError(f"Completed candidate artifact integrity mismatch: {path}")
    return marker


def _run_candidate_fold(
    cohort: FullCohort,
    fold: Mapping[str, Any],
    model_config: Mapping[str, Any],
    device: torch.device,
    experiment_dir: Path,
    split_manifest_sha256: str,
    *,
    amp_enabled: bool,
    lr: float,
    weight_decay: float,
) -> dict[str, Any]:
    model_name = str(model_config["name"])
    candidate_dir = _candidate_dir(experiment_dir, fold, model_name)
    completed = validate_candidate_complete(
        candidate_dir,
        experiment_dir,
        model_name=model_name,
        repeat=int(fold["repeat"]),
        local_fold=int(fold["local_fold"]),
        split_manifest_sha256=split_manifest_sha256,
    )
    if completed is not None:
        return completed

    candidate_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stage_a = _stage_a_select_epoch(
        cohort,
        fold,
        model_config,
        device,
        candidate_dir,
        amp_enabled=amp_enabled,
        lr=lr,
        weight_decay=weight_decay,
    )
    stage_b = _stage_b_refit_and_predict(
        cohort,
        fold,
        model_config,
        device,
        candidate_dir,
        best_epoch=int(stage_a["best_epoch"]),
        amp_enabled=amp_enabled,
        lr=lr,
        weight_decay=weight_decay,
    )
    peak_cuda_mb = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_cuda_mb = round(float(torch.cuda.max_memory_allocated(device)) / (1024**2), 3)
        if peak_cuda_mb > MAX_CUDA_MB:
            raise RuntimeError(f"Peak CUDA memory {peak_cuda_mb} MB exceeds {MAX_CUDA_MB} MB.")

    summary_path = candidate_dir / "summary.json"
    atomic_write_json(
        summary_path,
        {
            "model": model_name,
            "repeat": int(fold["repeat"]),
            "outer_seed": int(fold["outer_seed"]),
            "local_fold": int(fold["local_fold"]),
            "split_manifest_sha256": split_manifest_sha256,
            "stage_a": stage_a,
            "stage_b": stage_b,
            "peak_cuda_mb": peak_cuda_mb,
        },
    )
    artifact_paths = [
        summary_path,
        candidate_dir / "stage_a_best.pt",
        candidate_dir / "stage_b_final.pt",
        candidate_dir / "snapshot_predictions.csv",
        candidate_dir / "record_predictions.csv",
    ]
    marker = {
        "state": "SUCCEEDED",
        "model": model_name,
        "repeat": int(fold["repeat"]),
        "local_fold": int(fold["local_fold"]),
        "split_manifest_sha256": split_manifest_sha256,
        "artifacts": [_relative_artifact(path, experiment_dir) for path in artifact_paths],
    }
    atomic_write_json(candidate_dir / "complete.json", marker)
    return marker


def _concat_csv(paths: Sequence[Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError("No CSV paths supplied for Full aggregation.")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def _repeat_metrics(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat, group in frame.groupby("repeat", sort=True):
        true = np.asarray([LABEL_TO_INDEX[str(v)] for v in group["y_true"]], dtype=np.int64)
        pred = np.asarray([LABEL_TO_INDEX[str(v)] for v in group["y_pred"]], dtype=np.int64)
        metrics = compute_classification_metrics(true, pred, FROZEN_LABELS)
        rows.append({"repeat": int(repeat), **metrics.as_dict()})
    reduced: dict[str, Any] = {}
    for name in ("accuracy", "balanced_accuracy", "macro_f1"):
        values = np.asarray([float(row[name]) for row in rows], dtype=float)
        reduced[f"{name}_mean"] = float(values.mean())
        reduced[f"{name}_std"] = float(values.std())
    return rows, reduced


def _subject_and_class_summary(record_frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    subject_rows: list[dict[str, Any]] = []
    for subject_id, subject in record_frame.groupby("subject_id", sort=True):
        repeat_values: list[ClassificationMetrics] = []
        for _, group in subject.groupby("repeat", sort=True):
            true = np.asarray([LABEL_TO_INDEX[str(v)] for v in group["y_true"]])
            pred = np.asarray([LABEL_TO_INDEX[str(v)] for v in group["y_pred"]])
            repeat_values.append(compute_classification_metrics(true, pred, FROZEN_LABELS))
        subject_rows.append(
            {
                "subject_id": str(subject_id),
                "accuracy_mean": float(np.mean([m.accuracy for m in repeat_values])),
                "accuracy_std": float(np.std([m.accuracy for m in repeat_values])),
                "macro_f1_mean": float(np.mean([m.macro_f1 for m in repeat_values])),
                "macro_f1_std": float(np.std([m.macro_f1 for m in repeat_values])),
            }
        )
    subject_rows.sort(key=lambda row: (row["accuracy_mean"], row["macro_f1_mean"], row["subject_id"]))

    class_rows: list[dict[str, Any]] = []
    for label_index, label in enumerate(FROZEN_LABELS):
        per_repeat: list[Any] = []
        for _, group in record_frame.groupby("repeat", sort=True):
            true = np.asarray([LABEL_TO_INDEX[str(v)] for v in group["y_true"]])
            pred = np.asarray([LABEL_TO_INDEX[str(v)] for v in group["y_pred"]])
            per_repeat.append(compute_classification_metrics(true, pred, FROZEN_LABELS).per_class[label_index])
        class_rows.append(
            {
                "label": label,
                "precision_mean": float(np.mean([m.precision for m in per_repeat])),
                "recall_mean": float(np.mean([m.recall for m in per_repeat])),
                "f1_mean": float(np.mean([m.f1 for m in per_repeat])),
            }
        )
    return subject_rows, class_rows


def _calibration_summary(
    record_frame: pd.DataFrame, *, normalize_serialized_svm: bool = False
) -> dict[str, Any]:
    frame = record_frame.copy() if normalize_serialized_svm else record_frame
    renormalized_rows = 0
    max_raw_row_sum_drift = 0.0
    if normalize_serialized_svm:
        raw = frame[list(PROBA_COLUMNS)].to_numpy(dtype=float)
        if not np.isfinite(raw).all() or bool(((raw < 0.0) | (raw > 1.0)).any()):
            raise ValueError("Serialized SVM probabilities are non-finite or outside [0, 1].")
        row_sums = raw.sum(axis=1)
        drift = np.abs(row_sums - 1.0)
        max_raw_row_sum_drift = float(drift.max(initial=0.0))
        if max_raw_row_sum_drift > SVM_SERIALIZATION_MAX_ROW_SUM_DRIFT:
            raise ValueError(
                "Frozen SVM probability serialization drift exceeds the reviewed "
                f"{SVM_SERIALIZATION_MAX_ROW_SUM_DRIFT} bound: "
                f"{max_raw_row_sum_drift}."
            )
        if bool((row_sums <= 0.0).any()):
            raise ValueError("Serialized SVM probability row has a non-positive sum.")
        renormalized_rows = int((drift > 1e-6).sum())
        frame.loc[:, list(PROBA_COLUMNS)] = raw / row_sums[:, None]

    rows: list[dict[str, Any]] = []
    for repeat, group in frame.groupby("repeat", sort=True):
        probabilities = group[list(PROBA_COLUMNS)].to_numpy(dtype=float).tolist()
        labels = [LABEL_TO_INDEX[str(value)] for value in group["y_true"]]
        rows.append(
            {
                "repeat": int(repeat),
                "nll": record_multiclass_nll(probabilities, labels),
                "brier": record_multiclass_brier(probabilities, labels),
                "ece": record_ece(probabilities, labels, n_bins=CALIBRATION_ECE_BINS),
            }
        )
    result: dict[str, Any] = {
        "per_repeat": rows,
        "serialized_probability_renormalization": normalize_serialized_svm,
        "renormalized_rows": renormalized_rows,
        "max_raw_row_sum_drift": max_raw_row_sum_drift,
    }
    for name in ("nll", "brier", "ece"):
        values = np.asarray([row[name] for row in rows], dtype=float)
        result[f"{name}_mean"] = float(values.mean())
        result[f"{name}_std"] = float(values.std())
    return result


def _summarize_neural_model(
    model_name: str, experiment_dir: Path, folds: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    dirs = [_candidate_dir(experiment_dir, fold, model_name) for fold in folds]
    snapshot = _concat_csv([path / "snapshot_predictions.csv" for path in dirs])
    record = _concat_csv([path / "record_predictions.csv" for path in dirs])
    expected_snapshot_rows = DATA_BOUNDARY["n_snapshots"] * N_REPEATS
    expected_record_rows = DATA_BOUNDARY["n_records"] * N_REPEATS
    if len(snapshot) != expected_snapshot_rows or len(record) != expected_record_rows:
        raise ValueError(
            f"{model_name} OOF row-count mismatch: snapshots={len(snapshot)} "
            f"(expected {expected_snapshot_rows}), records={len(record)} "
            f"(expected {expected_record_rows})."
        )
    for repeat in range(N_REPEATS):
        snapshot_repeat_frame = snapshot.loc[snapshot["repeat"] == repeat]
        record_repeat_frame = record.loc[record["repeat"] == repeat]
        if (
            len(snapshot_repeat_frame) != DATA_BOUNDARY["n_snapshots"]
            or snapshot_repeat_frame["sample_id"].nunique() != DATA_BOUNDARY["n_snapshots"]
            or len(record_repeat_frame) != DATA_BOUNDARY["n_records"]
            or record_repeat_frame["record_id"].nunique() != DATA_BOUNDARY["n_records"]
        ):
            raise ValueError(
                f"{model_name} repeat {repeat} is not exact once-only OOF coverage."
            )
    snapshot_repeat, snapshot_reduced = _repeat_metrics(snapshot)
    record_repeat, record_reduced = _repeat_metrics(record)
    subjects, classes = _subject_and_class_summary(record)
    calibration = _calibration_summary(record)
    fold_summaries = [json.loads((path / "summary.json").read_text(encoding="utf-8")) for path in dirs]
    fold_metrics = [
        {
            "repeat": item["repeat"],
            "outer_seed": item["outer_seed"],
            "local_fold": item["local_fold"],
            "best_epoch": item["stage_a"]["best_epoch"],
            "snapshot": item["stage_b"]["snapshot_metrics"],
            "record": item["stage_b"]["record_metrics"],
            "stage_a_seconds": item["stage_a"]["seconds"],
            "stage_b_train_seconds": item["stage_b"]["train_seconds"],
            "inference_seconds": item["stage_b"]["inference_seconds"],
            "peak_cuda_mb": item["peak_cuda_mb"],
        }
        for item in fold_summaries
    ]
    return {
        "model": model_name,
        "status": "OK",
        "n_snapshot_samples": int(len(snapshot)),
        "n_records": int(len(record)),
        "snapshot": {**snapshot_reduced, "per_repeat": snapshot_repeat},
        "record": {**record_reduced, "per_repeat": record_repeat},
        "worst_subject": subjects[0],
        "per_subject": subjects,
        "per_class": classes,
        "weakest_class_record_f1": min(row["f1_mean"] for row in classes),
        "calibration": calibration,
        "per_fold": fold_metrics,
        "param_count": int(fold_summaries[0]["stage_b"]["param_count"]),
        "checkpoint_size_bytes_mean": float(
            np.mean([item["stage_b"]["checkpoint_size_bytes"] for item in fold_summaries])
        ),
        "total_train_seconds": float(
            sum(item["stage_a"]["seconds"] + item["stage_b"]["train_seconds"] for item in fold_summaries)
        ),
        "total_inference_seconds": float(
            sum(item["stage_b"]["inference_seconds"] for item in fold_summaries)
        ),
        "peak_cuda_mb": max(
            (item["peak_cuda_mb"] for item in fold_summaries if item["peak_cuda_mb"] is not None),
            default=None,
        ),
    }


def _artifact_path(name_fragment: str) -> Path:
    for item in FROZEN_SVM_REFERENCE_ARTIFACTS:
        if name_fragment in str(item["path"]):
            return _resolve_project_path(item["path"])
    raise KeyError(name_fragment)


def _summarize_frozen_svm() -> dict[str, Any]:
    summary = pd.read_csv(_artifact_path("_summary_"))
    row = summary.loc[summary["model"] == FROZEN_SVM_REFERENCE]
    if len(row) != 1:
        raise ValueError("Frozen P5.1 summary must contain exactly one calibrated_linear_svm row.")
    item = row.iloc[0]
    per_class = pd.read_csv(_artifact_path("_per_class_"))
    per_class = per_class.loc[per_class["model"] == FROZEN_SVM_REFERENCE]
    per_subject = pd.read_csv(_artifact_path("_per_subject_"))
    per_subject = per_subject.loc[per_subject["model"] == FROZEN_SVM_REFERENCE]
    worst = per_subject.loc[per_subject["is_worst"].astype(str).str.lower() == "true"]
    if len(worst) != 1 or len(per_class) != len(FROZEN_LABELS):
        raise ValueError("Frozen SVM per-subject/per-class evidence has an unexpected shape.")
    record = pd.read_csv(_artifact_path("_record_level_"))
    record = record.loc[record["model"] == FROZEN_SVM_REFERENCE]
    if len(record) != DATA_BOUNDARY["n_records"] * 3:
        raise ValueError("Frozen SVM record evidence has an unexpected row count.")
    return {
        "model": FROZEN_SVM_REFERENCE,
        "status": "OK",
        "record": {
            "macro_f1_mean": float(item["record_macro_f1_mean"]),
            "macro_f1_std": float(item["record_macro_f1_std"]),
            "balanced_accuracy_mean": float(item["record_balanced_acc_mean"]),
            "balanced_accuracy_std": float(item["record_balanced_acc_std"]),
            "accuracy_mean": float(item["record_accuracy_mean"]),
            "accuracy_std": float(item["record_accuracy_std"]),
        },
        "worst_subject": {
            "subject_id": str(worst.iloc[0]["subject_id"]),
            "accuracy_mean": float(worst.iloc[0]["accuracy_mean"]),
            "macro_f1_mean": float(worst.iloc[0]["macro_f1_mean"]),
        },
        "per_class": per_class.to_dict(orient="records"),
        "weakest_class_record_f1": float(per_class["f1_mean"].min()),
        "calibration": _calibration_summary(record, normalize_serialized_svm=True),
        "frozen_reference": True,
    }


def _candidate_result(summary: Mapping[str, Any]) -> FullCandidateResult:
    record = summary["record"]
    return FullCandidateResult(
        model=str(summary["model"]),
        passed_gate=summary.get("status") == "OK",
        is_frozen_svm=summary["model"] == FROZEN_SVM_REFERENCE,
        record_macro_f1_mean=float(record["macro_f1_mean"]),
        record_balanced_acc_mean=float(record["balanced_accuracy_mean"]),
        worst_subject_macro_f1_mean=float(summary["worst_subject"]["macro_f1_mean"]),
        record_macro_f1_std=float(record["macro_f1_std"]),
        weakest_class_record_f1=float(summary["weakest_class_record_f1"]),
    )


def _write_combined_predictions(
    experiment_dir: Path, folds: Sequence[Mapping[str, Any]], model_names: Sequence[str]
) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for kind, filename in (("snapshot", "oof_snapshot_predictions.csv"), ("record", "record_predictions.csv")):
        paths = [
            _candidate_dir(experiment_dir, fold, model) / f"{kind}_predictions.csv"
            for model in model_names
            for fold in folds
        ]
        output = experiment_dir / filename
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as target:
                wrote_header = False
                for path in paths:
                    with path.open("r", encoding="utf-8", newline="") as source:
                        header = source.readline()
                        if not wrote_header:
                            target.write(header)
                            wrote_header = True
                        for line in source:
                            target.write(line)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        outputs[kind] = str(output)
    return outputs


def _run_fold_set(
    cohort: FullCohort,
    folds: Sequence[Mapping[str, Any]],
    parameters: Mapping[str, Any],
    device: torch.device,
    experiment_dir: Path,
    split_sha: str,
) -> None:
    inner = parameters["inner_epoch_selection"]
    amp_enabled = bool(parameters["resources_and_stop"]["amp_enabled"])
    for fold in folds:
        for model_config in parameters["model_configs"]:
            _run_candidate_fold(
                cohort,
                fold,
                model_config,
                device,
                experiment_dir,
                split_sha,
                amp_enabled=amp_enabled,
                lr=float(inner["optimizer"]["lr"]),
                weight_decay=float(inner["optimizer"]["weight_decay"]),
            )
            completed_train_seconds = 0.0
            for summary_path in experiment_dir.glob("folds/repeat_*/fold_*/*/summary.json"):
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                completed_train_seconds += float(summary["stage_a"]["seconds"])
                completed_train_seconds += float(summary["stage_b"]["train_seconds"])
            if completed_train_seconds > MAX_TOTAL_TRAIN_SECONDS:
                raise TimeoutError(
                    "Completed Full training time "
                    f"{completed_train_seconds:.3f}s exceeded frozen budget "
                    f"{MAX_TOTAL_TRAIN_SECONDS}s."
                )


def run_popu_neural_full(
    parameters: Mapping[str, Any], seed: int, experiment_dir: Path
) -> dict[str, Any]:
    """Execute the accepted P5.2-C Full comparison and return JSON-safe metrics."""
    _require_frozen_invocation(parameters, seed)
    device = resolve_device(parameters["resources_and_stop"]["device"])
    svm_artifacts = verify_svm_reference_artifacts()
    cohort = load_full_cohort(parameters)
    split_manifest = build_full_fold_manifest(cohort.subjects)
    validate_full_fold_manifest(split_manifest, cohort.subjects)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    split_path = experiment_dir / "split_manifest.json"
    if split_path.exists():
        existing = json.loads(split_path.read_text(encoding="utf-8"))
        if existing != split_manifest:
            raise ValueError("Existing split manifest differs from the frozen rebuild.")
    else:
        atomic_write_json(split_path, split_manifest)

    atomic_write_json(experiment_dir / "svm_reference_verification.json", {"artifacts": svm_artifacts})
    folds = list(split_manifest["folds"])
    _run_fold_set(cohort, folds, parameters, device, experiment_dir, split_manifest["sha256"])

    model_summaries = [
        _summarize_neural_model(name, experiment_dir, folds) for name in NEURAL_CANDIDATES
    ]
    svm_summary = _summarize_frozen_svm()
    all_summaries = [svm_summary, *model_summaries]
    winner = select_full_winner([_candidate_result(summary) for summary in all_summaries])
    prediction_outputs = _write_combined_predictions(experiment_dir, folds, NEURAL_CANDIDATES)
    comparison = {
        "protocol": "popu_neural_full_v0.1",
        "split_manifest_sha256": split_manifest["sha256"],
        "subjects": len(cohort.subjects),
        "records": len(set(cohort.record_ids)),
        "snapshots": len(cohort.sample_ids),
        "models": {summary["model"]: summary for summary in all_summaries},
        "recommended_winner_pending_reviewer": winner,
        "candidate_frozen": False,
        "prediction_outputs": prediction_outputs,
    }
    atomic_write_json(experiment_dir / "full_comparison.json", comparison)
    return comparison


def run_one_fold_preflight(
    parameters: Mapping[str, Any], seed: int, output_dir: Path, *, repeat: int = 0, local_fold: int = 0
) -> dict[str, Any]:
    """Run the frozen three-model Stage-A/B path for one outer fold only."""
    _require_frozen_invocation(parameters, seed)
    if output_dir.exists():
        raise FileExistsError(f"Preflight output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    device = resolve_device(parameters["resources_and_stop"]["device"])
    svm_artifacts = verify_svm_reference_artifacts()
    cohort = load_full_cohort(parameters)
    manifest = build_full_fold_manifest(cohort.subjects)
    validate_full_fold_manifest(manifest, cohort.subjects)
    matches = [
        fold
        for fold in manifest["folds"]
        if int(fold["repeat"]) == repeat and int(fold["local_fold"]) == local_fold
    ]
    if len(matches) != 1:
        raise ValueError("Preflight fold selection did not resolve exactly one fold.")
    atomic_write_json(output_dir / "split_manifest.json", manifest)
    started = time.perf_counter()
    _run_fold_set(cohort, matches, parameters, device, output_dir, manifest["sha256"])
    observed = round(time.perf_counter() - started, 6)
    estimate = round(observed * len(manifest["folds"]), 6)
    result = {
        "state": "SUCCEEDED",
        "scope": "one_fold_timing_preflight",
        "repeat": repeat,
        "local_fold": local_fold,
        "observed_seconds": observed,
        "estimated_full_seconds": estimate,
        "within_frozen_budget": estimate <= MAX_TOTAL_TRAIN_SECONDS,
        "split_manifest_sha256": manifest["sha256"],
        "svm_artifacts_verified": svm_artifacts,
        "full_not_run": True,
    }
    atomic_write_json(output_dir / "preflight.json", result)
    return result
