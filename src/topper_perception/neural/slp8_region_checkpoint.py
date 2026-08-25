"""Checkpoint management for SLP8 region segmentation smoke (TASK-SLP-B03-PM-ONLY-REGION-SMOKE-v0.1).

This module provides checkpoint save/load functionality for the SLP8 Slp8TinyFcn model,
adapted from the PoPu neural checkpoint module.

Key features:
* Versioned checkpoint format
* RNG state capture/restore (Python/NumPy/torch)
* weights_only-safe loading
* Model config embedding
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from topper_perception.neural.slp8_region_models import (
    INPUT_SHAPE,
    MODEL_VERSION,
    N_CLASSES,
    Slp8TinyFcn,
)


# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

CHECKPOINT_VERSION = "slp8_region_smoke_v0.1"

# Required keys in checkpoint
_REQUIRED_KEYS = (
    "version",
    "model_state_dict",
    "optimizer_state_dict",
    "epoch",
    "model_config",
    "seed",
    "rng_state",
    "metrics",
    "n_classes",
    "input_shape",
)


# ---------------------------------------------------------------------------
# RNG state capture/restore
# ---------------------------------------------------------------------------


def capture_rng_state() -> dict[str, Any]:
    """Return the Python/NumPy/torch RNG states in a JSON-safe form."""
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    return {
        "torch": torch.get_rng_state().cpu().tolist(),
        "torch_cuda": (
            [state.cpu().tolist() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
        "numpy": {
            "legacy": str(numpy_state[0]),
            "state": numpy_state[1].tolist(),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "python": {
            "version": int(python_state[0]),
            "state": list(python_state[1]),
            "gauss": None if python_state[2] is None else float(python_state[2]),
        },
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore Python/NumPy/torch RNG states captured by capture_rng_state."""
    torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
    cuda_states = state.get("torch_cuda", [])
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [torch.tensor(item, dtype=torch.uint8) for item in cuda_states]
        )

    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["legacy"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["pos"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )

    python_state = state["python"]
    random.setstate(
        (int(python_state["version"]), tuple(python_state["state"]), python_state["gauss"])
    )


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def _to_cpu(value: Any) -> Any:
    """Move any torch tensors in a nested structure to detached CPU tensors."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


# ---------------------------------------------------------------------------
# Checkpoint build/save/load
# ---------------------------------------------------------------------------


def build_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_config: Mapping[str, Any],
    seed: int,
    metrics: Mapping[str, Any] | None = None,
    n_classes: int = N_CLASSES,
    input_shape: tuple[int, int] = INPUT_SHAPE,
) -> dict[str, Any]:
    """Assemble a versioned checkpoint payload.

    Parameters
    ----------
    model : nn.Module
        Model to save.
    optimizer : torch.optim.Optimizer
        Optimizer to save.
    epoch : int
        Current epoch number.
    model_config : Mapping[str, Any]
        Model configuration dict.
    seed : int
        Random seed.
    metrics : Mapping[str, Any] | None
        Metrics dict to save.
    n_classes : int
        Number of classes.
    input_shape : tuple[int, int]
        Input spatial shape.

    Returns
    -------
    dict[str, Any]
        Checkpoint payload.
    """
    return {
        "version": CHECKPOINT_VERSION,
        "model_state_dict": _to_cpu(model.state_dict()),
        "optimizer_state_dict": _to_cpu(optimizer.state_dict()),
        "epoch": int(epoch),
        "model_config": dict(model_config),
        "seed": int(seed),
        "rng_state": capture_rng_state(),
        "metrics": dict(metrics or {}),
        "n_classes": int(n_classes),
        "input_shape": list(input_shape),
    }


def save_checkpoint(
    path: Path | str,
    payload: Mapping[str, Any],
) -> str:
    """Atomically write checkpoint and return SHA-256 of the file.

    Parameters
    ----------
    path : Path | str
        Output path.
    payload : Mapping[str, Any]
        Checkpoint payload.

    Returns
    -------
    str
        SHA-256 hex digest of the saved checkpoint file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

    # Compute SHA-256 of saved file
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_checkpoint(
    path: Path | str,
    *,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint with weights_only=True (no unpickling of objects).

    Parameters
    ----------
    path : Path | str
        Checkpoint path.
    map_location : Any
        Device to map tensors to.

    Returns
    -------
    dict[str, Any]
        Checkpoint payload.
    """
    return torch.load(
        Path(path),
        map_location=map_location,
        weights_only=True,
    )


def validate_checkpoint(payload: Mapping[str, Any]) -> None:
    """Validate checkpoint payload structure.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Checkpoint payload.

    Raises
    ------
    ValueError
        If required keys are missing or version mismatches.
    """
    for key in _REQUIRED_KEYS:
        if key not in payload:
            raise ValueError(f"Checkpoint is missing required key {key!r}.")

    if payload["version"] != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {payload['version']!r}; "
            f"expected {CHECKPOINT_VERSION!r}."
        )

    if payload["n_classes"] != N_CLASSES:
        raise ValueError(
            f"Checkpoint n_classes={payload['n_classes']} "
            f"does not match model n_classes={N_CLASSES}."
        )

    if tuple(payload["input_shape"]) != INPUT_SHAPE:
        raise ValueError(
            f"Checkpoint input_shape={payload['input_shape']} "
            f"does not match model input_shape={INPUT_SHAPE}."
        )


# ---------------------------------------------------------------------------
# Model restore from checkpoint
# ---------------------------------------------------------------------------


def restore_model_from_checkpoint(
    checkpoint_path: Path | str,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> tuple[Slp8TinyFcn, torch.optim.Optimizer | None, int, dict[str, Any]]:
    """Restore model and optimizer from checkpoint.

    Parameters
    ----------
    checkpoint_path : Path | str
        Path to checkpoint file.
    optimizer : torch.optim.Optimizer | None
        Optimizer to restore state into. Must be same type as saved optimizer.
    device : str
        Device to load model to.

    Returns
    -------
    tuple[Slp8TinyFcn, torch.optim.Optimizer | None, int, dict[str, Any]]
        (model, optimizer, epoch, metrics)

    Raises
    ------
    ValueError
        If checkpoint validation fails.
    """
    payload = load_checkpoint(checkpoint_path, map_location=device)
    validate_checkpoint(payload)

    # Create model
    model = Slp8TinyFcn(n_classes=payload["n_classes"])
    model.load_state_dict(payload["model_state_dict"])
    model = model.to(device)

    # Restore optimizer if provided
    restored_optimizer = None
    if optimizer is not None and "optimizer_state_dict" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            restored_optimizer = optimizer
        except Exception as e:
            # Log but don't fail - optimizer may have different state
            print(f"Warning: Could not restore optimizer state: {e}")
            restored_optimizer = optimizer

    # Restore RNG state
    if "rng_state" in payload:
        restore_rng_state(payload["rng_state"])

    return model, restored_optimizer, int(payload["epoch"]), dict(payload.get("metrics", {}))


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def compute_checkpoint_diff(
    checkpoint1_path: Path | str,
    checkpoint2_path: Path | str,
) -> dict[str, Any]:
    """Compute parameter differences between two checkpoints.

    Parameters
    ----------
    checkpoint1_path : Path | str
        First checkpoint.
    checkpoint2_path : Path | str
        Second checkpoint.

    Returns
    -------
    dict[str, Any]
        Per-layer and total L2 differences.
    """
    ckpt1 = load_checkpoint(checkpoint1_path, map_location="cpu")
    ckpt2 = load_checkpoint(checkpoint2_path, map_location="cpu")

    diff = {}
    total = 0.0
    for key in ckpt1.get("model_state_dict", {}):
        if key not in ckpt2.get("model_state_dict", {}):
            continue
        t1 = torch.tensor(ckpt1["model_state_dict"][key])
        t2 = torch.tensor(ckpt2["model_state_dict"][key])
        layer_diff = float((t1 - t2).float().pow(2).sum().sqrt().item())
        diff[key] = layer_diff
        total += layer_diff ** 2

    diff["_total"] = float(math.sqrt(total))
    return diff
