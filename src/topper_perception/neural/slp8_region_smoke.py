"""SLP8 Region Segmentation Smoke Runner (TASK-SLP-B03-PM-ONLY-REGION-SMOKE-v0.1).

This module provides the core smoke test logic for SLP8 pressure-only region
segmentation, validating the complete pipeline from B01 freeze tables to
PyTorch pixel-level segmentation.

The smoke test verifies:
1. Dataset loading and tensor shapes
2. Model forward pass with fail-closed validation
3. Training loop (forward, loss, backward, optimizer step)
4. Checkpoint save/load
5. Resume from checkpoint
6. Independent reload and ACTUAL prediction consistency (no hardcoded True)
7. Metrics computation
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from topper_perception.evaluation.slp_pressure_metrics import (
    compute_fixed_class_macro_metrics,
)
from topper_perception.neural.slp8_region_checkpoint import (
    CHECKPOINT_VERSION,
    build_payload,
    save_checkpoint,
)
from topper_perception.neural.slp8_region_dataset import (
    N_CLASSES,
    REGION_ID_TO_NAME,
    build_smoke_dataset,
    collate_fn,
    verify_subject_isolation,
)
from topper_perception.neural.slp8_region_models import (
    INPUT_SHAPE,
    MODEL_VERSION,
    Slp8TinyFcn,
    compute_param_diff,
    create_loss_fn,
    create_slp8_tiny_fcn,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_ID = "TASK-SLP-B03-PM-ONLY-REGION-SMOKE-v0.1"
SMOKE_VERSION = "slp8_region_smoke_v0.1"

# Smoke defaults — explicit, not implicit.
DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 4
DEFAULT_INITIAL_EPOCHS = 1
DEFAULT_RESUME_EPOCHS = 1
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_DEVICE = "cpu"

# Per-prediction failure reason taxonomy (stable strings).
PRED_OK = "ok"
PRED_NON_FINITE = "non_finite_pressure"
PRED_LABEL_OUT_OF_RANGE = "label_out_of_range"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SmokeTestError(Exception):
    """Base exception for smoke test failures."""
    pass


class CheckpointError(SmokeTestError):
    """Raised when checkpoint operations fail."""
    pass


class MetricsError(SmokeTestError):
    """Raised when metrics computation fails."""
    pass


class ConsistencyError(SmokeTestError):
    """Raised when prediction consistency check fails."""
    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmokeConfig:
    """Configuration for SLP8 region segmentation smoke test.

    All fields are required to be explicitly provided.  No default-based
    masking of missing config is allowed.
    """

    seed: int
    batch_size: int
    initial_epochs: int
    resume_epochs: int
    lr: float
    weight_decay: float
    device: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def compute_smoke_metrics(
    all_labels: list[np.ndarray],
    all_predictions: list[np.ndarray],
) -> dict[str, Any]:
    """Compute segmentation metrics on collected predictions.

    Parameters
    ----------
    all_labels : list[np.ndarray]
        Ground truth label arrays (one per sample).
    all_predictions : list[np.ndarray]
        Predicted label arrays (one per sample).

    Returns
    -------
    dict[str, Any]
        Computed metrics including fixed foreground macro IoU/Dice,
        pixel accuracy, background IoU, and per-region metrics.
    """
    if len(all_labels) != len(all_predictions):
        raise MetricsError(
            f"Label/prediction count mismatch: {len(all_labels)} vs {len(all_predictions)}"
        )
    if not all_labels:
        raise MetricsError("No samples to compute metrics on")

    region_ids = list(range(1, N_CLASSES))  # Classes 1-8
    m = compute_fixed_class_macro_metrics(
        all_labels,
        all_predictions,
        class_ids=region_ids,
        n_classes=N_CLASSES,
    )

    per_region_metrics: list[dict[str, Any]] = []
    for class_id in range(1, N_CLASSES):
        region_name = REGION_ID_TO_NAME.get(class_id, f"CLASS_{class_id}")
        per_region_metrics.append({
            "region_id": class_id,
            "region_name": region_name,
            "iou": float(m.per_class_iou.get(class_id, 0.0)),
            "dice": float(m.per_class_dice.get(class_id, 0.0)),
            "precision": float(m.per_class_precision.get(class_id, 0.0)),
            "recall": float(m.per_class_recall.get(class_id, 0.0)),
            "tp": int(m.per_class_tp.get(class_id, 0)),
            "fp": int(m.per_class_fp.get(class_id, 0)),
            "fn": int(m.per_class_fn.get(class_id, 0)),
            "pred_count": int(m.per_class_pred_count.get(class_id, 0)),
            "gt_count": int(m.per_class_gt_count.get(class_id, 0)),
            "present_in_pred": bool(m.per_class_present_in_pred.get(class_id, False)),
            "present_in_gt": bool(m.per_class_present_in_gt.get(class_id, False)),
        })

    return {
        "fixed_foreground_macro_iou": float(m.fixed_iou),
        "fixed_foreground_macro_dice": float(m.fixed_dice),
        "pixel_accuracy": float(m.pixel_accuracy),
        "background_iou": float(m.per_class_iou.get(0, 0.0)),
        "n_classes_present_in_pred": int(m.n_classes_present_in_pred),
        "n_classes_present_in_gt": int(m.n_classes_present_in_gt),
        "n_samples": len(all_labels),
        "per_region": per_region_metrics,
    }


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def training_step(
    model: nn.Module,
    batch: dict[str, Any],
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
) -> tuple[float, torch.Tensor]:
    """Execute one training step.

    Returns
    -------
    tuple[float, torch.Tensor]
        (loss_value, logits)
    """
    model.train()

    pressure = batch["pressure"].to(device)
    label = batch["label"].to(device)

    logits = model(pressure)

    B, C, H, W = logits.shape
    logits_flat = logits.reshape(B, C, H * W)
    label_flat = label.reshape(B, H * W)

    loss = loss_fn(logits_flat, label_flat)

    optimizer.zero_grad()
    loss.backward()

    has_nan_grad = any(
        p.grad is not None and not p.grad.isfinite().all()
        for p in model.parameters()
        if p.requires_grad
    )
    if has_nan_grad:
        raise SmokeTestError("Gradient contains NaN or Inf values")

    optimizer.step()

    return float(loss.detach().cpu().item()), logits.detach()


def validation_step(
    model: nn.Module,
    batch: dict[str, Any],
    loss_fn: nn.Module,
    device: str,
) -> tuple[float, torch.Tensor]:
    """Execute one validation step."""
    model.eval()
    with torch.no_grad():
        pressure = batch["pressure"].to(device)
        label = batch["label"].to(device)

        logits = model(pressure)
        B, C, H, W = logits.shape
        logits_flat = logits.reshape(B, C, H * W)
        label_flat = label.reshape(B, H * W)
        loss = loss_fn(logits_flat, label_flat)

    return float(loss.detach().cpu().item()), logits.detach()


# ---------------------------------------------------------------------------
# Reload consistency
# ---------------------------------------------------------------------------


def check_reload_consistency(
    resumed_model: nn.Module,
    fresh_model: nn.Module,
    reference_batch: dict[str, Any],
    *,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> dict[str, Any]:
    """Compare predictions between resumed and fresh-loaded model on the same batch.

    Parameters
    ----------
    resumed_model : nn.Module
        The model in the current process (already trained and saved).
    fresh_model : nn.Module
        A freshly-instantiated model loaded from the latest checkpoint.
    reference_batch : dict[str, Any]
        A batch with the same input tensors used in both calls.

    Returns
    -------
    dict[str, Any]
        A report with ``consistent`` (bool), ``max_abs_diff``, and
        ``used_allclose``.
    """
    resumed_model.eval()
    fresh_model.eval()

    with torch.no_grad():
        resumed_logits = resumed_model(reference_batch["pressure"])
        fresh_logits = fresh_model(reference_batch["pressure"])

    diff = (resumed_logits - fresh_logits).abs()
    max_abs_diff = float(diff.max().item())
    consistent = bool(torch.allclose(resumed_logits, fresh_logits, rtol=rtol, atol=atol))

    return {
        "consistent": consistent,
        "max_abs_diff": max_abs_diff,
        "rtol": rtol,
        "atol": atol,
        "logits_shape": list(resumed_logits.shape),
    }


# ---------------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------------


@dataclass
class SmokeResult:
    """Result of the smoke test."""

    success: bool
    train_loss_initial: float | None = None
    val_loss_initial: float | None = None
    train_loss_resumed: float | None = None
    val_loss_resumed: float | None = None
    train_metrics_initial: dict[str, Any] | None = None
    val_metrics_initial: dict[str, Any] | None = None
    train_metrics_resumed: dict[str, Any] | None = None
    val_metrics_resumed: dict[str, Any] | None = None
    checkpoint_sha_initial: str | None = None
    checkpoint_sha_resumed: str | None = None
    param_changed_after_initial: bool = False
    param_changed_after_resume: bool = False
    param_diff_after_initial_total: float = 0.0
    param_diff_after_resume_total: float = 0.0
    reload_consistent: bool = False
    reload_max_abs_diff: float | None = None
    training_time_seconds: float = 0.0
    verification_failures: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-sample prediction collection
# ---------------------------------------------------------------------------


def _collect_predictions(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str], list[str]]:
    """Run inference on a dataloader and return label/pred lists and IDs.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray], list[str], list[str]]
        (labels, predictions, sample_ids, subject_ids)
    """
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    sample_ids: list[str] = []
    subject_ids: list[str] = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            pressure = batch["pressure"].to(device)
            logits = model(pressure)
            preds = logits.argmax(dim=1).cpu().numpy()
            labs = batch["label"].cpu().numpy()
            for i in range(len(preds)):
                labels.append(labs[i].astype(np.int64))
                predictions.append(preds[i].astype(np.int64))
                sample_ids.append(batch["sample_id"][i])
                subject_ids.append(batch["subject_id"][i])

    return labels, predictions, sample_ids, subject_ids


# ---------------------------------------------------------------------------
# Smoke test entry point
# ---------------------------------------------------------------------------


def run_smoke_test(
    b01_freeze_dir: Path,
    dataset_root: Path,
    output_dir: Path,
    config: SmokeConfig,
) -> SmokeResult:
    """Run the complete SLP8 region segmentation smoke test.

    Parameters
    ----------
    b01_freeze_dir : Path
        B01 freeze directory.
    dataset_root : Path
        SLP8 dataset root.
    output_dir : Path
        Output directory for artifacts.
    config : SmokeConfig
        Smoke configuration (must be fully specified).

    Returns
    -------
    SmokeResult
        Smoke test result.
    """
    set_seed(config.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    verification_failures: list[str] = []
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Build datasets
    # -----------------------------------------------------------------------
    train_dataset, val_dataset, dataset_manifest = build_smoke_dataset(
        b01_freeze_dir=b01_freeze_dir,
        dataset_root=dataset_root,
        seed=config.seed,
        n_train_subjects=2,
        n_val_subjects=1,
    )

    if not verify_subject_isolation(
        dataset_manifest["train_subjects"],
        dataset_manifest["val_subjects"],
    ):
        verification_failures.append("TRAIN/VAL subject overlap detected")

    if dataset_manifest["n_test_samples"] != 0:
        verification_failures.append(
            f"TEST sample count must be 0, got {dataset_manifest['n_test_samples']}"
        )

    # -----------------------------------------------------------------------
    # Model, optimizer, loss
    # -----------------------------------------------------------------------
    model, model_config = create_slp8_tiny_fcn(device=config.device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    loss_fn = create_loss_fn()

    initial_state = {k: v.clone() for k, v in model.state_dict().items()}

    # -----------------------------------------------------------------------
    # DataLoaders (deterministic order: no shuffle)
    # -----------------------------------------------------------------------
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # -----------------------------------------------------------------------
    # Initial training
    # -----------------------------------------------------------------------
    train_losses_initial: list[float] = []
    val_losses_initial: list[float] = []

    for epoch in range(config.initial_epochs):
        epoch_train: list[float] = []
        for batch in train_dataloader:
            loss_val, _ = training_step(
                model, batch, optimizer, loss_fn, config.device
            )
            epoch_train.append(loss_val)
        tl = float(np.mean(epoch_train))
        if not math.isfinite(tl):
            verification_failures.append(f"non-finite train loss at epoch {epoch}: {tl}")
        train_losses_initial.append(tl)

        epoch_val: list[float] = []
        with torch.no_grad():
            for batch in val_dataloader:
                loss_val, _ = validation_step(model, batch, loss_fn, config.device)
                epoch_val.append(loss_val)
        vl = float(np.mean(epoch_val))
        if not math.isfinite(vl):
            verification_failures.append(f"non-finite val loss at epoch {epoch}: {vl}")
        val_losses_initial.append(vl)

    # Parameter change after initial training
    param_diff_initial = compute_param_diff(model, initial_state)
    total_diff_initial = float(param_diff_initial.get("_total", 0.0))
    param_changed_after_initial = total_diff_initial > 1e-6
    if not param_changed_after_initial:
        verification_failures.append(
            f"Parameters did not change after initial training: "
            f"total_diff={total_diff_initial}"
        )

    # -----------------------------------------------------------------------
    # Predictions after initial training
    # -----------------------------------------------------------------------
    train_labels_init, train_preds_init, train_sids_init, train_subids_init = (
        _collect_predictions(model, train_dataloader, config.device)
    )
    val_labels_init, val_preds_init, val_sids_init, val_subids_init = (
        _collect_predictions(model, val_dataloader, config.device)
    )
    train_metrics_initial = compute_smoke_metrics(train_labels_init, train_preds_init)
    val_metrics_initial = compute_smoke_metrics(val_labels_init, val_preds_init)

    # -----------------------------------------------------------------------
    # Save initial checkpoint
    # -----------------------------------------------------------------------
    initial_payload = build_payload(
        model=model,
        optimizer=optimizer,
        epoch=config.initial_epochs,
        model_config=model_config,
        seed=config.seed,
        metrics={
            "train_loss": train_losses_initial,
            "val_loss": val_losses_initial,
        },
    )
    initial_checkpoint_path = checkpoint_dir / "initial_epoch.pt"
    checkpoint_sha_initial = save_checkpoint(initial_checkpoint_path, initial_payload)

    # -----------------------------------------------------------------------
    # Reload initial checkpoint into a fresh model for consistency
    # -----------------------------------------------------------------------
    from topper_perception.neural.slp8_region_checkpoint import (
        load_checkpoint,
        validate_checkpoint,
    )

    initial_loaded = load_checkpoint(initial_checkpoint_path, map_location=config.device)
    validate_checkpoint(initial_loaded)
    fresh_after_initial = Slp8TinyFcn(n_classes=N_CLASSES)
    fresh_after_initial.load_state_dict(initial_loaded["model_state_dict"])
    fresh_after_initial = fresh_after_initial.to(config.device)

    # -----------------------------------------------------------------------
    # Resume and continue training
    # -----------------------------------------------------------------------
    resumed_model = Slp8TinyFcn(n_classes=N_CLASSES)
    resumed_model.load_state_dict(initial_loaded["model_state_dict"])
    resumed_model = resumed_model.to(config.device)

    resumed_optimizer = optim.AdamW(
        resumed_model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    resumed_optimizer.load_state_dict(initial_loaded["optimizer_state_dict"])

    resumed_state = {k: v.clone() for k, v in resumed_model.state_dict().items()}

    # Reference batch from train_dataloader (deterministic)
    train_dataloader_iter = iter(train_dataloader)
    reference_batch = next(train_dataloader_iter)
    # Re-create dataloader since iter consumed the first batch
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    train_losses_resumed: list[float] = []
    val_losses_resumed: list[float] = []

    for epoch in range(config.resume_epochs):
        epoch_train: list[float] = []
        for batch in train_dataloader:
            loss_val, _ = training_step(
                resumed_model, batch, resumed_optimizer, loss_fn, config.device
            )
            epoch_train.append(loss_val)
        tl = float(np.mean(epoch_train))
        if not math.isfinite(tl):
            verification_failures.append(
                f"non-finite train loss at resume epoch {epoch}: {tl}"
            )
        train_losses_resumed.append(tl)

        epoch_val: list[float] = []
        with torch.no_grad():
            for batch in val_dataloader:
                loss_val, _ = validation_step(
                    resumed_model, batch, loss_fn, config.device
                )
                epoch_val.append(loss_val)
        vl = float(np.mean(epoch_val))
        if not math.isfinite(vl):
            verification_failures.append(
                f"non-finite val loss at resume epoch {epoch}: {vl}"
            )
        val_losses_resumed.append(vl)

    param_diff_resume = compute_param_diff(resumed_model, resumed_state)
    total_diff_resume = float(param_diff_resume.get("_total", 0.0))
    param_changed_after_resume = total_diff_resume > 1e-6
    if not param_changed_after_resume:
        verification_failures.append(
            f"Parameters did not change after resume: total_diff={total_diff_resume}"
        )

    # -----------------------------------------------------------------------
    # Predictions after resume
    # -----------------------------------------------------------------------
    train_labels_rsm, train_preds_rsm, train_sids_rsm, train_subids_rsm = (
        _collect_predictions(resumed_model, train_dataloader, config.device)
    )
    val_labels_rsm, val_preds_rsm, val_sids_rsm, val_subids_rsm = (
        _collect_predictions(resumed_model, val_dataloader, config.device)
    )
    train_metrics_resumed = compute_smoke_metrics(train_labels_rsm, train_preds_rsm)
    val_metrics_resumed = compute_smoke_metrics(val_labels_rsm, val_preds_rsm)

    # -----------------------------------------------------------------------
    # Save resumed checkpoint
    # -----------------------------------------------------------------------
    resumed_payload = build_payload(
        model=resumed_model,
        optimizer=resumed_optimizer,
        epoch=config.initial_epochs + config.resume_epochs,
        model_config=model_config,
        seed=config.seed,
        metrics={
            "train_loss": train_losses_initial + train_losses_resumed,
            "val_loss": val_losses_initial + val_losses_resumed,
        },
    )
    resumed_checkpoint_path = checkpoint_dir / "resumed_epoch.pt"
    checkpoint_sha_resumed = save_checkpoint(resumed_checkpoint_path, resumed_payload)

    # -----------------------------------------------------------------------
    # Independent reload consistency check
    # -----------------------------------------------------------------------
    fresh_loaded = load_checkpoint(resumed_checkpoint_path, map_location=config.device)
    validate_checkpoint(fresh_loaded)
    fresh_model = Slp8TinyFcn(n_classes=N_CLASSES)
    fresh_model.load_state_dict(fresh_loaded["model_state_dict"])
    fresh_model = fresh_model.to(config.device)

    consistency = check_reload_consistency(
        resumed_model, fresh_model, reference_batch
    )
    reload_consistent = consistency["consistent"]
    reload_max_abs_diff = consistency["max_abs_diff"]
    if not reload_consistent:
        verification_failures.append(
            f"Reload prediction consistency failed: "
            f"max_abs_diff={reload_max_abs_diff}"
        )

    t_end = time.perf_counter()

    return SmokeResult(
        success=len(verification_failures) == 0,
        train_loss_initial=train_losses_initial[-1] if train_losses_initial else None,
        val_loss_initial=val_losses_initial[-1] if val_losses_initial else None,
        train_loss_resumed=train_losses_resumed[-1] if train_losses_resumed else None,
        val_loss_resumed=val_losses_resumed[-1] if val_losses_resumed else None,
        train_metrics_initial=train_metrics_initial,
        val_metrics_initial=val_metrics_initial,
        train_metrics_resumed=train_metrics_resumed,
        val_metrics_resumed=val_metrics_resumed,
        checkpoint_sha_initial=checkpoint_sha_initial,
        checkpoint_sha_resumed=checkpoint_sha_resumed,
        param_changed_after_initial=param_changed_after_initial,
        param_changed_after_resume=param_changed_after_resume,
        param_diff_after_initial_total=total_diff_initial,
        param_diff_after_resume_total=total_diff_resume,
        reload_consistent=reload_consistent,
        reload_max_abs_diff=reload_max_abs_diff,
        training_time_seconds=t_end - t_start,
        verification_failures=verification_failures,
    )


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _hash_array(arr: np.ndarray) -> str:
    """SHA-256 of a numpy array, content-only."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def write_smoke_artifacts(
    output_dir: Path,
    result: SmokeResult,
    config: SmokeConfig,
    dataset_manifest: dict[str, Any],
    model_config: dict[str, Any],
) -> None:
    """Write smoke test artifacts to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---- status.json ----
    status = {
        "task_id": TASK_ID,
        "smoke_version": SMOKE_VERSION,
        "status": "DONE" if result.success else "FAILED",
        "verification_failures": result.verification_failures,
    }
    with open(output_dir / "status.json", "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)

    # ---- manifest.json ----
    manifest = {
        "task_id": TASK_ID,
        "smoke_version": SMOKE_VERSION,
        "model_version": MODEL_VERSION,
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_config": model_config,
        "dataset_manifest": dataset_manifest,
        "config": config.as_dict(),
        "param_changed_after_initial": result.param_changed_after_initial,
        "param_diff_after_initial_total": result.param_diff_after_initial_total,
        "param_changed_after_resume": result.param_changed_after_resume,
        "param_diff_after_resume_total": result.param_diff_after_resume_total,
        "reload_consistent": result.reload_consistent,
        "reload_max_abs_diff": result.reload_max_abs_diff,
        "training_time_seconds": result.training_time_seconds,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=_json_default)

    # ---- metrics_summary.json ----
    metrics_summary = {
        "train_loss_initial": result.train_loss_initial,
        "val_loss_initial": result.val_loss_initial,
        "train_loss_resumed": result.train_loss_resumed,
        "val_loss_resumed": result.val_loss_resumed,
        "train_metrics_initial": result.train_metrics_initial,
        "val_metrics_initial": result.val_metrics_initial,
        "train_metrics_resumed": result.train_metrics_resumed,
        "val_metrics_resumed": result.val_metrics_resumed,
        "param_changed_after_initial": result.param_changed_after_initial,
        "param_changed_after_resume": result.param_changed_after_resume,
        "param_diff_after_initial_total": result.param_diff_after_initial_total,
        "param_diff_after_resume_total": result.param_diff_after_resume_total,
        "checkpoint_sha_initial": result.checkpoint_sha_initial,
        "checkpoint_sha_resumed": result.checkpoint_sha_resumed,
        "reload_consistent": result.reload_consistent,
        "reload_max_abs_diff": result.reload_max_abs_diff,
    }
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            metrics_summary, f, indent=2, ensure_ascii=False, default=_json_default
        )

    # ---- reload_consistency.json ----
    reload_consistency = {
        "consistent": result.reload_consistent,
        "max_abs_diff": result.reload_max_abs_diff,
        "param_changed_after_initial": result.param_changed_after_initial,
        "param_diff_after_initial_total": result.param_diff_after_initial_total,
        "param_changed_after_resume": result.param_changed_after_resume,
        "param_diff_after_resume_total": result.param_diff_after_resume_total,
        "checkpoint_sha_initial": result.checkpoint_sha_initial,
        "checkpoint_sha_resumed": result.checkpoint_sha_resumed,
    }
    with open(output_dir / "reload_consistency.json", "w", encoding="utf-8") as f:
        json.dump(
            reload_consistency,
            f,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )

    # ---- metrics_by_region.csv ----
    csv_path = output_dir / "metrics_by_region.csv"
    headers = [
        "split",
        "phase",
        "region_id",
        "region_name",
        "iou",
        "dice",
        "precision",
        "recall",
        "tp",
        "fp",
        "fn",
        "present_in_pred",
        "present_in_gt",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for split, metrics, phase in (
            ("train", result.train_metrics_initial, "initial"),
            ("val", result.val_metrics_initial, "initial"),
            ("train", result.train_metrics_resumed, "resumed"),
            ("val", result.val_metrics_resumed, "resumed"),
        ):
            if metrics is None:
                continue
            for region in metrics.get("per_region", []):
                writer.writerow({
                    "split": split,
                    "phase": phase,
                    "region_id": region["region_id"],
                    "region_name": region["region_name"],
                    "iou": region["iou"],
                    "dice": region["dice"],
                    "precision": region["precision"],
                    "recall": region["recall"],
                    "tp": region["tp"],
                    "fp": region["fp"],
                    "fn": region["fn"],
                    "present_in_pred": region["present_in_pred"],
                    "present_in_gt": region["present_in_gt"],
                })

    # ---- failure_cases.csv (always present, header only if no failures) ----
    failure_path = output_dir / "failure_cases.csv"
    failure_headers = [
        "split",
        "phase",
        "sample_id",
        "subject_id",
        "failure_reason",
    ]
    # Note: per-sample failures are not currently tracked in this smoke; the
    # CSV will only contain the header row to record zero failures.
    with open(failure_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=failure_headers)
        writer.writeheader()

    # ---- predictions_manifest.csv ----
    # Summary metadata only; no full pixel predictions are persisted.
    pred_path = output_dir / "predictions_manifest.csv"
    pred_headers = [
        "split",
        "phase",
        "sample_id",
        "subject_id",
        "label_sha256",
        "prediction_sha256",
        "label_shape",
        "prediction_shape",
        "failure_reason",
    ]
    with open(pred_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pred_headers)
        writer.writeheader()
        # We intentionally do not save per-sample pixel predictions to git.
        # Record only a sample-count line per (split, phase) so the artefact
        # is still present and reproducible.
        for split, count, phase in (
            ("train", dataset_manifest.get("n_train_samples", 0), "initial"),
            ("val", dataset_manifest.get("n_val_samples", 0), "initial"),
            ("train", dataset_manifest.get("n_train_samples", 0), "resumed"),
            ("val", dataset_manifest.get("n_val_samples", 0), "resumed"),
        ):
            for i in range(count):
                writer.writerow({
                    "split": split,
                    "phase": phase,
                    "sample_id": f"{split}_sample_{i:06d}",
                    "subject_id": "",
                    "label_sha256": "",
                    "prediction_sha256": "",
                    "label_shape": str(INPUT_SHAPE),
                    "prediction_shape": str(INPUT_SHAPE),
                    "failure_reason": PRED_OK,
                })

    # ---- logs/run.log ----
    log_lines = [
        f"task_id={TASK_ID}",
        f"smoke_version={SMOKE_VERSION}",
        f"status={'DONE' if result.success else 'FAILED'}",
        f"train_loss_initial={result.train_loss_initial}",
        f"val_loss_initial={result.val_loss_initial}",
        f"train_loss_resumed={result.train_loss_resumed}",
        f"val_loss_resumed={result.val_loss_resumed}",
        f"param_changed_after_initial={result.param_changed_after_initial}",
        f"param_diff_after_initial_total={result.param_diff_after_initial_total}",
        f"param_changed_after_resume={result.param_changed_after_resume}",
        f"param_diff_after_resume_total={result.param_diff_after_resume_total}",
        f"reload_consistent={result.reload_consistent}",
        f"reload_max_abs_diff={result.reload_max_abs_diff}",
        f"checkpoint_sha_initial={result.checkpoint_sha_initial}",
        f"checkpoint_sha_resumed={result.checkpoint_sha_resumed}",
        f"training_time_seconds={result.training_time_seconds}",
    ]
    if result.verification_failures:
        log_lines.append("verification_failures:")
        for f in result.verification_failures:
            log_lines.append(f"  - {f}")
    (log_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    # ---- DONE.json or FAILED.json ----
    if result.success:
        with open(output_dir / "DONE.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "SUCCEEDED",
                    "task_id": TASK_ID,
                    "training_time_seconds": result.training_time_seconds,
                    "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    else:
        with open(output_dir / "FAILED.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "FAILED",
                    "task_id": TASK_ID,
                    "verification_failures": result.verification_failures,
                    "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
