"""Checkpoint save/load for the PoPu neural path (P5.2-A2).

A checkpoint bundles the model/optimizer/epoch state with the model config, the
frozen label order, the normalization statistics, the seed, and the RNG state so
training can be resumed and independently reloaded. RNG states are stored in a
JSON/``weights_only``-safe form (lists of primitives), never as raw pickle
objects.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from topper_perception.neural.data import FROZEN_LABELS

CHECKPOINT_VERSION = "v0.1"

_REQUIRED_KEYS = (
    "version",
    "model_state_dict",
    "optimizer_state_dict",
    "epoch",
    "model_config",
    "frozen_labels",
    "normalization",
    "seed",
    "rng_state",
    "metrics",
)


def capture_rng_state() -> dict[str, Any]:
    """Return the Python/NumPy/torch RNG states in a JSON-safe form."""
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    return {
        "torch": torch.get_rng_state().cpu().tolist(),
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
    """Restore Python/NumPy/torch RNG states captured by :func:`capture_rng_state`."""
    torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))

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


def build_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    model_config: Mapping[str, Any],
    normalization: Mapping[str, float],
    seed: int,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a versioned checkpoint payload."""
    return {
        "version": CHECKPOINT_VERSION,
        "model_state_dict": _to_cpu(model.state_dict()),
        "optimizer_state_dict": _to_cpu(optimizer.state_dict()),
        "epoch": int(epoch),
        "model_config": dict(model_config),
        "frozen_labels": list(FROZEN_LABELS),
        "normalization": {
            "mean": float(normalization["mean"]),
            "std": float(normalization["std"]),
        },
        "seed": int(seed),
        "rng_state": capture_rng_state(),
        "metrics": dict(metrics or {}),
    }


def save_checkpoint(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` with ``torch.save``; parents are created as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(payload), path)


def load_checkpoint(
    path: Path | str,
    *,
    map_location: Any = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint with ``weights_only=True`` (no unpickling of objects)."""
    return torch.load(Path(path), map_location=map_location, weights_only=True)


def validate_checkpoint(payload: Mapping[str, Any]) -> None:
    """Fail loudly on a missing key, version mismatch, or label-order drift."""
    for key in _REQUIRED_KEYS:
        if key not in payload:
            raise ValueError(f"Checkpoint is missing required key {key!r}.")
    if payload["version"] != CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {payload['version']!r}; "
            f"expected {CHECKPOINT_VERSION!r}."
        )
    if tuple(payload["frozen_labels"]) != tuple(FROZEN_LABELS):
        raise ValueError(
            "Checkpoint frozen_labels do not match the frozen label order; "
            "refusing to load a checkpoint trained under a different mapping."
        )
