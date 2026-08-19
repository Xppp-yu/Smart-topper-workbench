"""P5.2-B Mini screening runner for the PoPu neural path.

Torch-only, imported lazily by :mod:`topper_perception.experiments.runner`. This
runner drives one governed Mini screening over a *fixed* development-subject
subset of the primary labeled PoPu cohort. It reuses the P5.2-A training
primitives (subject-isolated split, train-only normalization, label-aware
flip augmentation, checkpoint/resume/reload, fixed-seed reproducibility) and
adds the Mini-specific pieces: a multi-epoch loop with early stopping, a fixed
best-checkpoint rule, per-epoch/per-class metrics, timing/device/AMP/memory
capture, and a pre-registered viability gate.

The cohort is enforced against the frozen P2 quality manifest
(``parameters.quality_manifest``): only records whose ``quality_status`` is in
``parameters.cohort`` (``primary`` = ACCEPT only) are built into samples, so
WARN/EXCLUDED/REJECT records cannot leak into a primary-cohort Mini. The
train/val split is *explicit*: ``parameters.train_subject_ids`` /
``parameters.val_subject_ids`` are frozen in config (at least 2 validation
subjects) and are never derived from a ratio.

It never writes to the shared raw data directory, never reads a test split
(the Mini config freezes an explicit train/val subject partition with no test
subjects), and never trains a Full model.
"""

from __future__ import annotations

import csv
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from topper_perception.experiments.artifacts import atomic_write_json
from topper_perception.healthcheck import load_path_config
from topper_perception.io.popu_inventory import iter_tactilus_record_paths, resolve_tactilus_root
from topper_perception.neural.checkpoint import (
    build_payload,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
    validate_checkpoint,
)
from topper_perception.neural.data import (
    FROZEN_LABELS,
    MatrixNormalizer,
    SubjectSplit,
    build_labeled_samples,
    horizontal_flip,
    to_model_input,
    validate_subject_split,
)
from topper_perception.neural.dataset import PressureDataset, build_dataloader
from topper_perception.neural.early_stopping import EarlyStopper, best_checkpoint_rule
from topper_perception.neural.metrics import compute_classification_metrics
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
from topper_perception.neural.viability import assess_viability, overall_verdict


def _project_root() -> Path:
    here = Path(__file__).resolve().parent
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return here


def _resolve_data_root(parameters: Mapping[str, Any]) -> Path:
    """Resolve the PoPu data root from ``parameters.data_root`` or the local path config."""
    raw = parameters.get("data_root")
    if raw:
        return Path(str(raw)).expanduser()
    paths = load_path_config(_project_root() / "configs" / "paths.local.json")
    return paths["popu_data"]


_COHORT_ACCEPTED: dict[str, frozenset[str]] = {
    "primary": frozenset({"ACCEPT"}),
    "combined": frozenset({"ACCEPT", "WARN"}),
}


def _record_sample_id(record_path: Path, tactilus_root: Path) -> str:
    """Return the record-level sample id used by the P1/P2/P4a manifests."""
    return f"popu-tactilus::{record_path.relative_to(tactilus_root).as_posix()}"


def _load_quality_manifest(manifest_path: Path) -> dict[str, str]:
    """Read the frozen P2 quality manifest (CSV) into ``{sample_id: quality_status}``.

    The manifest is the *only* source of truth for cohort membership: a record
    whose ``sample_id`` is absent is treated as a stale-manifest error, never
    silently promoted into the cohort.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"P2 quality manifest not found: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "sample_id" not in fields or "quality_status" not in fields:
            raise ValueError(
                "P2 quality manifest must contain 'sample_id' and 'quality_status' columns."
            )
        statuses: dict[str, str] = {}
        for row in reader:
            sample_id = str(row.get("sample_id", "")).strip()
            status = str(row.get("quality_status", "")).strip()
            if sample_id:
                statuses[sample_id] = status
    return statuses


def _resolve_manifest_path(parameters: Mapping[str, Any]) -> Path:
    raw = parameters.get("quality_manifest")
    if not raw:
        raise ValueError("parameters.quality_manifest is required to apply the frozen P2 cohort.")
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else _project_root() / path


def collect_labeled_samples(
    parameters: Mapping[str, Any], data_root: Path
) -> tuple[list[Any], dict[str, Any]]:
    """Collect a balanced, deterministic, cohort-filtered sample subset for Mini.

    Mirrors the frozen P5.2-A smoke sampling (round-robin across subject/label
    groups) but first applies the frozen P2 quality manifest: only records whose
    ``quality_status`` belongs to the configured cohort (``primary`` = ACCEPT
    only, or ``combined`` = ACCEPT+WARN) are considered. WARN/EXCLUDED/REJECT
    records are dropped *before* any sample is built, so they can never leak
    into the primary cohort. The subject subset is fixed *before* any model
    result and is the only place subject selection happens.
    """
    subject_ids = list(dict.fromkeys(str(s) for s in parameters.get("subject_ids", ["1", "2"])))
    max_samples = int(parameters.get("max_samples", 1000))
    if max_samples <= 0:
        raise ValueError("parameters.max_samples must be a positive integer.")

    cohort = str(parameters.get("cohort", "primary"))
    if cohort not in _COHORT_ACCEPTED:
        raise ValueError(
            f"parameters.cohort must be one of {sorted(_COHORT_ACCEPTED)}; got {cohort!r}."
        )
    accepted = _COHORT_ACCEPTED[cohort]

    manifest_path = _resolve_manifest_path(parameters)
    statuses = _load_quality_manifest(manifest_path)

    tactilus_root = resolve_tactilus_root(data_root)
    record_paths = [
        path
        for path in iter_tactilus_record_paths(data_root)
        if path.parent.name in set(subject_ids)
    ]
    if not record_paths:
        raise ValueError(
            f"No Tactilus records found for subjects {subject_ids} under {data_root}."
        )

    cohort_kept: list[Path] = []
    n_excluded_by_cohort = 0
    missing_from_manifest: list[str] = []
    for path in record_paths:
        sample_id = _record_sample_id(path, tactilus_root)
        status = statuses.get(sample_id)
        if status is None:
            missing_from_manifest.append(sample_id)
        elif status in accepted:
            cohort_kept.append(path)
        else:
            n_excluded_by_cohort += 1
    if missing_from_manifest:
        raise ValueError(
            "Records missing from the P2 quality manifest (stale manifest?); "
            f"first: {missing_from_manifest[:5]}."
        )

    available = build_labeled_samples(cohort_kept, tactilus_root=data_root)
    if not available:
        raise ValueError(
            f"No labeled samples remain for cohort={cohort!r}; every selected-subject "
            "record was excluded by the P2 quality manifest."
        )

    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for sample in available:
        groups[(sample.subject_id, sample.posture)].append(sample)
    required_groups = [
        (subject_id, posture) for subject_id in subject_ids for posture in FROZEN_LABELS
    ]
    missing = [group for group in required_groups if not groups[group]]
    if missing:
        raise ValueError(
            "Mini requires every selected subject to contain all frozen labels in the "
            f"{cohort!r} cohort; missing groups: {missing}."
        )
    if max_samples < len(required_groups):
        raise ValueError(
            f"parameters.max_samples must be at least {len(required_groups)} to "
            "include every subject/label group."
        )

    selected: list[Any] = []
    offset = 0
    while len(selected) < max_samples:
        added = False
        for group in required_groups:
            bucket = groups[group]
            if offset < len(bucket):
                selected.append(bucket[offset])
                added = True
                if len(selected) == max_samples:
                    break
        if not added:
            break
        offset += 1

    cohort_stats = {
        "cohort": cohort,
        "quality_manifest": str(manifest_path),
        "n_records_considered": len(record_paths),
        "n_records_excluded_by_cohort": n_excluded_by_cohort,
        "n_labeled_available": len(available),
    }
    return selected, cohort_stats


def _class_counts(labels: np.ndarray) -> dict[str, int]:
    counts = Counter(int(label) for label in labels)
    return {posture: int(counts.get(index, 0)) for index, posture in enumerate(FROZEN_LABELS)}


@dataclass(frozen=True, slots=True)
class MiniData:
    """Everything the training loop needs, prepared once for all models."""

    train_loader: Any
    val_loader: Any
    normalizer: MatrixNormalizer
    split_signature: str
    train_subjects: tuple[str, ...]
    val_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]
    n_train: int
    n_val: int
    augmented: int
    train_class_counts_before_augmentation: dict[str, int]
    train_class_counts_after_augmentation: dict[str, int]
    val_class_counts: dict[str, int]
    n_labeled_available: int
    n_selected_samples: int
    no_leakage: bool
    cohort: str
    quality_manifest: str
    n_records_excluded_by_cohort: int


def _prepare_data(parameters: Mapping[str, Any], seed: int, data_root: Path) -> MiniData:
    params = dict(parameters)
    batch_size = int(params.get("batch_size", 32))
    flip_augmentation = bool(params.get("flip_augmentation", True))

    set_seed(seed)
    samples, cohort_stats = collect_labeled_samples(params, data_root)
    n_selected_samples = len(samples)

    train_subject_ids = [str(s) for s in params.get("train_subject_ids", [])]
    val_subject_ids = [str(s) for s in params.get("val_subject_ids", [])]
    subject_set = {str(s) for s in params.get("subject_ids", [])}
    train_set = set(train_subject_ids)
    val_set = set(val_subject_ids)
    if not train_subject_ids or not val_subject_ids:
        raise ValueError(
            "Mini config must explicitly freeze train_subject_ids and val_subject_ids."
        )
    if len(val_subject_ids) < 2:
        raise ValueError("Mini config must freeze at least 2 validation subjects.")
    if train_set & val_set:
        raise ValueError("train_subject_ids and val_subject_ids must be disjoint.")
    if subject_set:
        if not (train_set <= subject_set and val_set <= subject_set):
            raise ValueError("train_subject_ids and val_subject_ids must be drawn from subject_ids.")
        if (train_set | val_set) != subject_set:
            raise ValueError("train_subject_ids ∪ val_subject_ids must equal the frozen subject_ids.")

    subject_seq = [s.subject_id for s in samples]
    subject_array = np.asarray(subject_seq, dtype=object)
    train_indices = np.flatnonzero(
        np.isin(subject_array, np.asarray(train_subject_ids, dtype=object))
    )
    val_indices = np.flatnonzero(
        np.isin(subject_array, np.asarray(val_subject_ids, dtype=object))
    )
    if train_indices.size == 0 or val_indices.size == 0:
        raise ValueError("Both train and validation splits must be non-empty.")
    split = SubjectSplit(
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=np.asarray([], dtype=np.int64),
        train_subjects=tuple(train_subject_ids),
        val_subjects=tuple(val_subject_ids),
        test_subjects=(),
    )
    validate_subject_split(split, subject_seq)
    no_leakage = True

    x, y = to_model_input(samples)
    metadata = [(s.sample_id, s.record_id, s.subject_id) for s in samples]

    normalizer = MatrixNormalizer().fit(x[split.train_indices])
    x_norm = normalizer.transform(x)

    x_train, y_train = x_norm[split.train_indices], y[split.train_indices]
    meta_train = [metadata[i] for i in split.train_indices]
    x_val, y_val = x_norm[split.val_indices], y[split.val_indices]
    meta_val = [metadata[i] for i in split.val_indices]

    train_before = _class_counts(y[split.train_indices])

    augmented = 0
    if flip_augmentation and len(x_train):
        flipped, flipped_labels = horizontal_flip(x_train, y_train)
        x_train = np.concatenate([x_train, flipped], axis=0)
        y_train = np.concatenate([y_train, flipped_labels], axis=0)
        meta_train = meta_train + [
            (f"{sample_id}#flip", record_id, subject_id)
            for sample_id, record_id, subject_id in meta_train
        ]
        augmented = int(flipped.shape[0])

    train_dataset = PressureDataset(
        x_train,
        y_train,
        sample_ids=[m[0] for m in meta_train],
        record_ids=[m[1] for m in meta_train],
        subject_ids=[m[2] for m in meta_train],
    )
    val_dataset = (
        PressureDataset(
            x_val,
            y_val,
            sample_ids=[m[0] for m in meta_val],
            record_ids=[m[1] for m in meta_val],
            subject_ids=[m[2] for m in meta_val],
        )
        if len(x_val)
        else None
    )
    if val_dataset is None:
        raise ValueError("Mini requires a non-empty validation split for early stopping.")

    train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    split_signature = (
        f"train={','.join(train_subject_ids)}"
        f"|val={','.join(val_subject_ids)}"
        f"|test="
        f"|seed={seed}"
    )

    return MiniData(
        train_loader=train_loader,
        val_loader=val_loader,
        normalizer=normalizer,
        split_signature=split_signature,
        train_subjects=tuple(train_subject_ids),
        val_subjects=tuple(val_subject_ids),
        test_subjects=(),
        n_train=int(len(x_train)),
        n_val=int(len(x_val)),
        augmented=augmented,
        train_class_counts_before_augmentation=train_before,
        train_class_counts_after_augmentation=_class_counts(y_train),
        val_class_counts=_class_counts(y_val),
        n_labeled_available=cohort_stats["n_labeled_available"],
        n_selected_samples=n_selected_samples,
        no_leakage=no_leakage,
        cohort=cohort_stats["cohort"],
        quality_manifest=cohort_stats["quality_manifest"],
        n_records_excluded_by_cohort=cohort_stats["n_records_excluded_by_cohort"],
    )


def _write_predictions(path: Path, result: Any) -> None:
    rows = [
        {
            "sample_id": result.sample_ids[i],
            "record_id": result.record_ids[i],
            "subject_id": result.subject_ids[i],
            "label": int(result.labels[i]),
            "prediction": int(result.predictions[i]),
            "probabilities": [float(p) for p in result.probabilities[i]],
        }
        for i in range(result.n_samples)
    ]
    atomic_write_json(
        path,
        {"frozen_labels": list(FROZEN_LABELS), "n_samples": result.n_samples, "predictions": rows},
    )


def _prediction_consistent(model_a: Any, model_b: Any, val_loader: Any, device: torch.device) -> bool:
    first = predict(model_a, val_loader, device)
    second = predict(model_b, val_loader, device)
    same_pred = bool(np.array_equal(first.predictions, second.predictions))
    same_prob = bool(np.allclose(first.probabilities, second.probabilities, atol=1e-6))
    return same_pred and same_prob


def _verify_checkpoint_resume_reload(
    *,
    name: str,
    model: Any,
    optimizer: Any,
    model_config: Mapping[str, Any],
    seed: int,
    actual_epochs: int,
    device: torch.device,
    train_loader: Any,
    val_loader: Any,
    criterion: Any,
    optimizer_cfg: Mapping[str, Any],
    amp_enabled: bool,
    normalizer: MatrixNormalizer,
    experiment_dir: Path,
    best_predictions: np.ndarray,
) -> dict[str, Any]:
    """Verify resume (from latest) and independent reload (latest + best)."""
    latest_path = experiment_dir / "checkpoints" / f"{name}_latest.pt"
    best_path = experiment_dir / "checkpoints" / f"{name}_best.pt"

    lr = float(optimizer_cfg.get("lr", 1e-3))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))

    checkpoint = load_checkpoint(latest_path, map_location=device)
    validate_checkpoint(checkpoint)
    resume_model = build_model(model_config).to(device)
    resume_model.load_state_dict(checkpoint["model_state_dict"])
    resume_optimizer = make_optimizer(resume_model, lr=lr, weight_decay=weight_decay)
    resume_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    resumed_epoch = int(checkpoint["epoch"])
    parameters_before_resume = {
        key: value.detach().cpu().clone()
        for key, value in resume_model.state_dict().items()
    }
    restore_rng_state(checkpoint["rng_state"])
    resume_info = train_epoch(
        resume_model, train_loader, resume_optimizer, criterion, device, amp_enabled=amp_enabled
    )
    parameters_changed = any(
        not torch.equal(parameters_before_resume[key], value.detach().cpu())
        for key, value in resume_model.state_dict().items()
    )
    resume_ok = (
        resumed_epoch == actual_epochs
        and resume_info["samples"] > 0
        and np.isfinite(float(resume_info["loss"]))
        and parameters_changed
    )

    reload_model = build_model(model_config).to(device)
    reload_model.load_state_dict(
        load_checkpoint(latest_path, map_location=device)["model_state_dict"]
    )
    reload_latest_consistent = _prediction_consistent(model, reload_model, val_loader, device)

    best_checkpoint = load_checkpoint(best_path, map_location=device)
    validate_checkpoint(best_checkpoint)
    best_model = build_model(model_config).to(device)
    best_model.load_state_dict(best_checkpoint["model_state_dict"])
    best_reload = predict(best_model, val_loader, device)
    reload_best_consistent = bool(
        np.array_equal(best_reload.predictions, best_predictions)
        if best_predictions.size
        else True
    )

    return {
        "checkpoint_latest": str(latest_path),
        "checkpoint_best": str(best_path),
        "checkpoint_ok": latest_path.is_file() and best_path.is_file(),
        "resume_ok": resume_ok,
        "resume_from_epoch": resumed_epoch,
        "resume_parameters_changed": parameters_changed,
        "reload_latest_prediction_consistent": reload_latest_consistent,
        "reload_best_prediction_consistent": reload_best_consistent,
        "reload_ok": reload_latest_consistent and reload_best_consistent,
    }


def _train_model(
    model_config: Mapping[str, Any],
    *,
    seed: int,
    device: torch.device,
    train_loader: Any,
    val_loader: Any,
    criterion: Any,
    optimizer_cfg: Mapping[str, Any],
    amp_enabled: bool,
    normalizer: MatrixNormalizer,
    experiment_dir: Path,
    early_stopping: Mapping[str, Any],
    max_epochs: int,
    split_signature: str,
    no_leakage: bool,
    same_split: bool,
    augmented_train_samples: int,
) -> dict[str, Any]:
    """Train one candidate model for up to ``max_epochs`` with early stopping."""
    name = str(model_config["name"])
    set_seed(seed)
    model = build_model(model_config).to(device)
    optimizer = make_optimizer(
        model,
        lr=float(optimizer_cfg.get("lr", 1e-3)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )
    param_count = count_parameters(model)

    stopper = EarlyStopper(
        monitor=str(early_stopping.get("monitor", "val_loss")),
        mode=str(early_stopping.get("mode", "min")),
        patience=int(early_stopping.get("patience", 2)),
        min_delta=float(early_stopping.get("min_delta", 0.0)),
        min_epochs=int(early_stopping.get("min_epochs", 1)),
    )
    rule = best_checkpoint_rule(early_stopping)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latest_path = experiment_dir / "checkpoints" / f"{name}_latest.pt"
    best_path = experiment_dir / "checkpoints" / f"{name}_best.pt"

    epoch_history: list[dict[str, Any]] = []
    best_epoch = 0
    best_val_result: Any = None
    best_metrics: Any = None
    last_val_result: Any = None
    last_metrics: Any = None
    final_train_loss = float("nan")
    amp_active_any = False

    total_started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        epoch_started = time.perf_counter()
        info = train_epoch(
            model, train_loader, optimizer, criterion, device, amp_enabled=amp_enabled
        )
        epoch_seconds = time.perf_counter() - epoch_started
        train_loss = float(info["loss"])
        final_train_loss = train_loss
        amp_active_any = amp_active_any or bool(info["amp_active"])

        val_result = evaluate(model, val_loader, criterion, device)
        if val_result.n_samples == 0:
            raise RuntimeError(f"Mini invariant failed: {name} produced no validation samples.")
        val_loss = float(val_result.loss)
        metrics = compute_classification_metrics(
            val_result.labels, val_result.predictions, FROZEN_LABELS
        )
        last_val_result = val_result
        last_metrics = metrics

        step = stopper.step(epoch, val_loss)

        payload = build_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            model_config=model_config,
            normalization={"mean": normalizer.mean_, "std": normalizer.std_},
            seed=seed,
            metrics={"val_loss": val_loss, "val_balanced_accuracy": metrics.balanced_accuracy},
        )
        save_checkpoint(latest_path, payload)
        if step.is_best:
            save_checkpoint(best_path, payload)
            best_epoch = epoch
            best_val_result = val_result
            best_metrics = metrics

        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": metrics.accuracy,
                "val_macro_f1": metrics.macro_f1,
                "val_balanced_accuracy": metrics.balanced_accuracy,
                "seconds": round(epoch_seconds, 6),
                "amp_active": bool(info["amp_active"]),
                "is_best": step.is_best,
            }
        )

        if step.should_stop:
            break

    total_train_seconds = round(time.perf_counter() - total_started, 6)
    actual_epochs = len(epoch_history)

    peak_cuda_mb = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_cuda_mb = round(float(torch.cuda.max_memory_allocated(device)) / (1024**2), 3)

    # ``best_*`` always exists: the first epoch is the initial best.
    assert best_val_result is not None and best_metrics is not None and best_epoch >= 1
    verify = _verify_checkpoint_resume_reload(
        name=name,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        seed=seed,
        actual_epochs=actual_epochs,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer_cfg=optimizer_cfg,
        amp_enabled=amp_enabled,
        normalizer=normalizer,
        experiment_dir=experiment_dir,
        best_predictions=best_val_result.predictions,
    )

    if last_val_result is not None and last_val_result.n_samples:
        _write_predictions(experiment_dir / "predictions" / f"{name}.json", last_val_result)
    _write_predictions(experiment_dir / "predictions" / f"{name}_best.json", best_val_result)

    return {
        "name": name,
        "param_count": param_count,
        "actual_epochs": actual_epochs,
        "max_epochs": max_epochs,
        "epoch_history": epoch_history,
        "final_train_loss": final_train_loss,
        "val_loss": float(last_val_result.loss) if last_val_result is not None else None,
        "val_accuracy": last_metrics.accuracy,
        "val_macro_f1": last_metrics.macro_f1,
        "val_macro_precision": last_metrics.macro_precision,
        "val_macro_recall": last_metrics.macro_recall,
        "val_balanced_accuracy": last_metrics.balanced_accuracy,
        "per_class": [c.as_dict() for c in last_metrics.per_class],
        "confusion_matrix": [list(row) for row in last_metrics.confusion_matrix],
        "val_samples": last_metrics.n_samples,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_result.loss) if best_val_result is not None else None,
        "best_val_accuracy": best_metrics.accuracy,
        "best_val_macro_f1": best_metrics.macro_f1,
        "best_val_balanced_accuracy": best_metrics.balanced_accuracy,
        "total_train_seconds": total_train_seconds,
        "peak_cuda_mb": peak_cuda_mb,
        "device": str(device),
        "amp_active": amp_active_any,
        "best_checkpoint_rule": rule,
        "train_samples": int(len(train_loader.dataset)),
        "augmented_train_samples": augmented_train_samples,
        "split_signature": split_signature,
        "no_leakage": no_leakage,
        "same_split": same_split,
        **verify,
    }


def _train_once(
    model_config: Mapping[str, Any],
    seed: int,
    device: torch.device,
    train_loader: Any,
    criterion: Any,
    optimizer_cfg: Mapping[str, Any],
    amp_enabled: bool,
) -> float:
    set_seed(seed)
    model = build_model(model_config).to(device)
    optimizer = make_optimizer(
        model,
        lr=float(optimizer_cfg.get("lr", 1e-3)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )
    info = train_epoch(model, train_loader, optimizer, criterion, device, amp_enabled=amp_enabled)
    return float(info["loss"])


def run_popu_neural_mini(
    parameters: Mapping[str, Any],
    seed: int,
    experiment_dir: Path,
) -> dict[str, Any]:
    """Execute the governed P5.2-B Mini screening and return JSON-safe metrics."""
    params = dict(parameters)
    data_root = _resolve_data_root(params)
    device = resolve_device(params.get("device", "auto"))
    amp_enabled = bool(params.get("amp_enabled", False))
    max_epochs = int(params.get("epochs", 5))
    if max_epochs < 1:
        raise ValueError("parameters.epochs must be >= 1.")

    optimizer_cfg = dict(params.get("optimizer", {}))
    model_configs = list(params.get("model_configs", []))
    if not model_configs:
        raise ValueError("parameters.model_configs must be a non-empty list.")

    early_stopping = dict(params.get("early_stopping", {}))
    resource_limits = dict(params.get("resource_limits", {}))
    viability_cfg = dict(params.get("viability", {}))
    chance_margin = float(viability_cfg.get("chance_margin", 0.05))

    data = _prepare_data(params, seed, data_root)
    criterion = make_criterion()

    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "predictions").mkdir(parents=True, exist_ok=True)

    model_summaries: list[dict[str, Any]] = []
    for model_config in model_configs:
        summary = _train_model(
            model_config,
            seed=seed,
            device=device,
            train_loader=data.train_loader,
            val_loader=data.val_loader,
            criterion=criterion,
            optimizer_cfg=optimizer_cfg,
            amp_enabled=amp_enabled,
            normalizer=data.normalizer,
            experiment_dir=experiment_dir,
            early_stopping=early_stopping,
            max_epochs=max_epochs,
            split_signature=data.split_signature,
            no_leakage=data.no_leakage,
            same_split=True,
            augmented_train_samples=data.augmented,
        )
        model_summaries.append(summary)

    # Gate each model, then aggregate.
    for summary in model_summaries:
        summary["viability"] = assess_viability(
            summary,
            num_classes=len(FROZEN_LABELS),
            resource_limits=resource_limits,
            chance_margin=chance_margin,
        ).as_dict()
    verdicts = [summary["viability"]["verdict"] for summary in model_summaries]

    # Same-seed reproducibility: retrain the first model for one epoch and
    # compare against its recorded first-epoch training loss.
    first_summary = model_summaries[0]
    rerun_loss = _train_once(
        model_configs[0],
        seed=seed,
        device=device,
        train_loader=data.train_loader,
        criterion=criterion,
        optimizer_cfg=optimizer_cfg,
        amp_enabled=amp_enabled,
    )
    reproducible = bool(
        np.isclose(float(first_summary["epoch_history"][0]["train_loss"]), rerun_loss, rtol=1e-5, atol=1e-6)
    )

    atomic_write_json(
        experiment_dir / "train_log.json",
        {
            "frozen_labels": list(FROZEN_LABELS),
            "seed": seed,
            "device": str(device),
            "cohort": data.cohort,
            "quality_manifest": data.quality_manifest,
            "n_records_excluded_by_cohort": data.n_records_excluded_by_cohort,
            "train_subjects": list(data.train_subjects),
            "val_subjects": list(data.val_subjects),
            "test_subjects": list(data.test_subjects),
            "split_signature": data.split_signature,
            "early_stopping": early_stopping,
            "best_checkpoint_rule": best_checkpoint_rule(early_stopping),
            "reproducible_seed": reproducible,
            "models": model_summaries,
        },
    )

    models = {summary["name"]: summary for summary in model_summaries}
    return {
        "dataset": "popu_tactilus",
        "scope": "mini",
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "data_root": str(data_root),
        "subject_selection_rule": str(params.get("subject_selection_rule", "")),
        "cohort": data.cohort,
        "quality_manifest": data.quality_manifest,
        "n_records_excluded_by_cohort": data.n_records_excluded_by_cohort,
        "subjects": [str(s) for s in params.get("subject_ids", [])],
        "n_labeled_available": data.n_labeled_available,
        "n_selected_samples": data.n_selected_samples,
        "train_subjects": list(data.train_subjects),
        "val_subjects": list(data.val_subjects),
        "test_subjects": list(data.test_subjects),
        "split_signature": data.split_signature,
        "train_samples": data.n_train,
        "val_samples": data.n_val,
        "augmented_train_samples": data.augmented,
        "train_class_counts_before_augmentation": data.train_class_counts_before_augmentation,
        "train_class_counts_after_augmentation": data.train_class_counts_after_augmentation,
        "val_class_counts": data.val_class_counts,
        "max_epochs": max_epochs,
        "early_stopping": early_stopping,
        "best_checkpoint_rule": best_checkpoint_rule(early_stopping),
        "amp_enabled": amp_enabled,
        "frozen_labels": list(FROZEN_LABELS),
        "models": models,
        "viability": {
            "per_model": {summary["name"]: summary["viability"] for summary in model_summaries},
            "overall_verdict": overall_verdict(verdicts),
        },
        "reproducible_seed": reproducible,
    }
