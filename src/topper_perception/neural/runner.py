"""PoPu neural training smoke runner (P5.2-A2).

Torch-only, imported lazily by :mod:`topper_perception.experiments.runner` so the
governed runner stays dataset- and framework-agnostic. This runner drives one
governed CPU/GPU smoke over the primary labeled PoPu cohort:

- subject-isolated train/val split (no subject spans two splits);
- normalization statistics fit on train subjects only;
- horizontal-flip augmentation applied to the train split only;
- one training epoch per candidate model (MatrixMLP / TinyCNN / SmallResNet);
- checkpoint save, resume, and independent-reload prediction consistency;
- probability/prediction/provenance output written into the experiment dir.

It never writes to the shared raw data directory and never trains a full model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from topper_perception.experiments.artifacts import atomic_write_json
from topper_perception.healthcheck import load_path_config
from topper_perception.io.popu_inventory import (
    iter_tactilus_record_paths,
    resolve_tactilus_root,
)
from topper_perception.neural.checkpoint import (
    build_payload,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint,
)
from topper_perception.neural.data import (
    FROZEN_LABELS,
    MatrixNormalizer,
    build_labeled_samples,
    horizontal_flip,
    subject_split,
    to_model_input,
    validate_subject_split,
)
from topper_perception.neural.dataset import PressureDataset, build_dataloader
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


def _resolve_data_root(parameters: Mapping[str, Any]) -> Path:
    """Resolve the PoPu data root from ``parameters.data_root`` or the local path config."""
    raw = parameters.get("data_root")
    if raw:
        return Path(str(raw)).expanduser()
    paths = load_path_config(_project_root() / "configs" / "paths.local.json")
    return paths["popu_data"]


def _require_finite(value: Any, label: str) -> None:
    if value is None:
        return
    if not np.isfinite(float(value)):
        raise RuntimeError(f"Smoke invariant failed: {label} is not finite (got {value!r}).")


def _require_valid_probabilities(result: Any) -> None:
    probs = np.asarray(result.probabilities, dtype=np.float64)
    if not np.isfinite(result.logits).all():
        raise RuntimeError("Smoke invariant failed: logits contain NaN or infinity.")
    if not np.isfinite(probs).all():
        raise RuntimeError("Smoke invariant failed: probabilities contain NaN or infinity.")
    if bool(((probs < 0.0) | (probs > 1.0)).any()):
        raise RuntimeError("Smoke invariant failed: probabilities fall outside [0, 1].")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
        raise RuntimeError("Smoke invariant failed: probability rows do not sum to 1.")


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


def _prediction_consistent(model_a, model_b, val_loader, device: torch.device) -> bool:
    first = predict(model_a, val_loader, device)
    second = predict(model_b, val_loader, device)
    same_pred = bool(np.array_equal(first.predictions, second.predictions))
    same_prob = bool(np.allclose(first.probabilities, second.probabilities, atol=1e-6))
    return same_pred and same_prob


def _collect_labeled_samples(parameters: Mapping[str, Any], data_root: Path) -> list[Any]:
    subject_ids = [str(s) for s in parameters.get("subject_ids", ["1", "2"])]
    max_samples = int(parameters.get("max_samples", 1000))
    if max_samples <= 0:
        raise ValueError("parameters.max_samples must be a positive integer.")
    record_paths = [
        path
        for path in iter_tactilus_record_paths(data_root)
        if path.parent.name in set(subject_ids)
    ]
    if not record_paths:
        raise ValueError(
            f"No Tactilus records found for subjects {subject_ids} under {data_root}."
        )
    samples = build_labeled_samples(record_paths, tactilus_root=data_root)
    if not samples:
        raise ValueError("No labeled samples found for the selected primary-cohort subjects.")
    return samples[:max_samples]


def _train_once(
    model_config: Mapping[str, Any],
    seed: int,
    device: torch.device,
    train_loader: Any,
    criterion: Any,
    optimizer_cfg: Mapping[str, Any],
    amp_enabled: bool,
) -> tuple[Any, Any, dict[str, Any]]:
    set_seed(seed)
    model = build_model(model_config).to(device)
    optimizer = make_optimizer(
        model,
        lr=float(optimizer_cfg.get("lr", 1e-3)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )
    info = train_epoch(model, train_loader, optimizer, criterion, device, amp_enabled=amp_enabled)
    return model, optimizer, info


def _checkpoint_resume_reload(
    *,
    name: str,
    model: Any,
    optimizer: Any,
    model_config: Mapping[str, Any],
    seed: int,
    epochs: int,
    device: torch.device,
    train_loader: Any,
    val_loader: Any,
    criterion: Any,
    optimizer_cfg: Mapping[str, Any],
    amp_enabled: bool,
    normalizer: MatrixNormalizer,
    experiment_dir: Path,
    final_train_loss: float,
    val_result: Any,
) -> dict[str, Any]:
    """Save latest/best checkpoints, verify resume, and verify reload consistency."""
    latest_path = experiment_dir / "checkpoints" / f"{name}_latest.pt"
    best_path = experiment_dir / "checkpoints" / f"{name}_best.pt"
    payload = build_payload(
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        model_config=model_config,
        normalization={"mean": normalizer.mean_, "std": normalizer.std_},
        seed=seed,
        metrics={"final_train_loss": final_train_loss},
    )
    save_checkpoint(latest_path, payload)
    save_checkpoint(best_path, payload)  # single-epoch smoke: latest == best

    lr = float(optimizer_cfg.get("lr", 1e-3))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))

    checkpoint = load_checkpoint(latest_path, map_location=device)
    validate_checkpoint(checkpoint)
    resume_model = build_model(model_config).to(device)
    resume_model.load_state_dict(checkpoint["model_state_dict"])
    resume_optimizer = make_optimizer(resume_model, lr=lr, weight_decay=weight_decay)
    resume_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    resumed_epoch = int(checkpoint["epoch"])
    resume_info = train_epoch(
        resume_model, train_loader, resume_optimizer, criterion, device, amp_enabled=amp_enabled
    )
    _require_finite(resume_info["loss"], f"{name} resumed loss")

    reload_model = build_model(model_config).to(device)
    reload_model.load_state_dict(
        load_checkpoint(latest_path, map_location=device)["model_state_dict"]
    )
    reload_consistent = (
        _prediction_consistent(model, reload_model, val_loader, device)
        if val_loader is not None
        else True
    )

    if val_result is not None and val_result.n_samples:
        _write_predictions(experiment_dir / "predictions" / f"{name}.json", val_result)

    return {
        "checkpoint_latest": str(latest_path),
        "checkpoint_best": str(best_path),
        "resume_ok": resumed_epoch == epochs and resume_info["samples"] > 0,
        "resumed_epoch": resumed_epoch,
        "reload_prediction_consistent": reload_consistent,
    }


def _run_one_model(
    model_config: Mapping[str, Any],
    *,
    seed: int,
    epochs: int,
    device: torch.device,
    train_loader: Any,
    val_loader: Any,
    criterion: Any,
    optimizer_cfg: Mapping[str, Any],
    amp_enabled: bool,
    normalizer: MatrixNormalizer,
    experiment_dir: Path,
    augmented_train_samples: int,
) -> dict[str, Any]:
    name = str(model_config["name"])
    set_seed(seed)
    model = build_model(model_config).to(device)
    optimizer = make_optimizer(
        model,
        lr=float(optimizer_cfg.get("lr", 1e-3)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )
    param_count = count_parameters(model)

    final_train_loss = float("nan")
    amp_active = False
    for _ in range(epochs):
        info = train_epoch(
            model, train_loader, optimizer, criterion, device, amp_enabled=amp_enabled
        )
        final_train_loss = float(info["loss"])
        amp_active = bool(info["amp_active"])
    _require_finite(final_train_loss, f"{name} final train loss")

    val_result = evaluate(model, val_loader, criterion, device) if val_loader is not None else None
    if val_result is not None and val_result.n_samples:
        _require_valid_probabilities(val_result)
        _require_finite(val_result.loss, f"{name} validation loss")

    verify = _checkpoint_resume_reload(
        name=name,
        model=model,
        optimizer=optimizer,
        model_config=model_config,
        seed=seed,
        epochs=epochs,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer_cfg=optimizer_cfg,
        amp_enabled=amp_enabled,
        normalizer=normalizer,
        experiment_dir=experiment_dir,
        final_train_loss=final_train_loss,
        val_result=val_result,
    )

    return {
        "name": name,
        "param_count": param_count,
        "final_train_loss": final_train_loss,
        "val_loss": val_result.loss if val_result is not None else None,
        "val_accuracy": val_result.accuracy if val_result is not None else None,
        "val_samples": val_result.n_samples if val_result is not None else 0,
        "train_samples": int(len(train_loader.dataset)),
        "augmented_train_samples": augmented_train_samples,
        "epochs": epochs,
        "amp_active": amp_active,
        **verify,
    }


def run_popu_neural_smoke(
    parameters: Mapping[str, Any],
    seed: int,
    experiment_dir: Path,
) -> dict[str, Any]:
    """Execute the governed P5.2-A2 CPU/GPU smoke and return JSON-safe metrics."""
    params = dict(parameters)
    data_root = _resolve_data_root(params)
    device = resolve_device(params.get("device", "auto"))
    amp_enabled = bool(params.get("amp_enabled", False))
    epochs = int(params.get("epochs", 1))
    if epochs < 1:
        raise ValueError("parameters.epochs must be >= 1.")
    batch_size = int(params.get("batch_size", 32))
    val_ratio = float(params.get("val_ratio", 0.2))
    test_ratio = float(params.get("test_ratio", 0.0))
    flip_augmentation = bool(params.get("flip_augmentation", True))
    optimizer_cfg = dict(params.get("optimizer", {}))
    model_configs = list(params.get("model_configs", []))
    if not model_configs:
        raise ValueError("parameters.model_configs must be a non-empty list.")

    set_seed(seed)
    samples = _collect_labeled_samples(params, data_root)
    n_labeled_total = len(samples)

    subject_seq = [s.subject_id for s in samples]
    split = subject_split(
        subject_seq, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed, shuffle=True
    )
    validate_subject_split(split, subject_seq)

    x, y = to_model_input(samples)
    metadata = [(s.sample_id, s.record_id, s.subject_id) for s in samples]

    normalizer = MatrixNormalizer().fit(x[split.train_indices])
    x_norm = normalizer.transform(x)

    x_train, y_train = x_norm[split.train_indices], y[split.train_indices]
    meta_train = [metadata[i] for i in split.train_indices]
    x_val, y_val = x_norm[split.val_indices], y[split.val_indices]
    meta_val = [metadata[i] for i in split.val_indices]

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

    train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    criterion = make_criterion()

    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (experiment_dir / "predictions").mkdir(parents=True, exist_ok=True)

    model_results: dict[str, Any] = {}
    train_log: list[dict[str, Any]] = []
    for model_config in model_configs:
        result = _run_one_model(
            model_config,
            seed=seed,
            epochs=epochs,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer_cfg=optimizer_cfg,
            amp_enabled=amp_enabled,
            normalizer=normalizer,
            experiment_dir=experiment_dir,
            augmented_train_samples=augmented,
        )
        model_results[result["name"]] = result
        train_log.append(result)

    # Same-seed reproducibility: retrain the first model and compare the loss.
    first_name = model_results[model_configs[0]["name"]]["name"]
    _, _, rerun_info = _train_once(
        model_configs[0],
        seed=seed,
        device=device,
        train_loader=train_loader,
        criterion=criterion,
        optimizer_cfg=optimizer_cfg,
        amp_enabled=amp_enabled,
    )
    first_loss = float(model_results[first_name]["final_train_loss"])
    reproducible = bool(np.isclose(first_loss, float(rerun_info["loss"]), rtol=1e-5, atol=1e-6))

    atomic_write_json(
        experiment_dir / "train_log.json",
        {
            "frozen_labels": list(FROZEN_LABELS),
            "seed": seed,
            "device": str(device),
            "models": train_log,
            "reproducible_seed": reproducible,
        },
    )

    return {
        "dataset": "popu_tactilus",
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "data_root": str(data_root),
        "subjects": [str(s) for s in params.get("subject_ids", ["1", "2"])],
        "n_labeled_total": n_labeled_total,
        "train_subjects": list(split.train_subjects),
        "val_subjects": list(split.val_subjects),
        "test_subjects": list(split.test_subjects),
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "augmented_train_samples": augmented,
        "epochs": epochs,
        "amp_enabled": amp_enabled,
        "frozen_labels": list(FROZEN_LABELS),
        "models": model_results,
        "reproducible_seed": reproducible,
    }
