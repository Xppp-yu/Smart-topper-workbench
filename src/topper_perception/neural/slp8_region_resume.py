"""Checkpoint identity + resume verification for the B04 Mini (R02).

R02 hardens the resume contract:

* Every checkpoint embeds a :class:`CheckpointIdentity` block
  containing the config SHA, A06 / freeze SHAs, the class-weight
  vector, the candidate / model / version / seed, the early-stopper
  state, the RNG states and the loss / metric history up to that
  point.
* :func:`verify_resume_identity` rejects resume with a mismatched
  identity (different config, different candidate, different A06, …)
  with :class:`ResumeIdentityError`.
* :func:`refuse_resume_for_done_run` rejects resume for a run that
  already produced ``DONE.json`` (a completed run is a closed
  experiment and must not be silently extended).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ResumeIdentityError(Exception):
    """Raised when a resume attempt's identity does not match the saved one."""


class ResumeRefusedError(Exception):
    """Raised when a resume is refused because the run is already DONE."""


# ---------------------------------------------------------------------------
# Identity block
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointIdentity:
    """Stable identity fields embedded in every B04 checkpoint.

    The B04 R02 contract: a resume attempt that disagrees on **any**
    field in this block is rejected fail-closed.
    """

    task_id: str
    candidate: str
    model_version: str
    seed: int
    n_classes: int
    image_shape: tuple[int, int]
    config_sha256: str
    a06_split_sha256: str
    freeze_manifest_sha256: str
    train_class_stats_sha256: str
    class_weight_sha256: str
    input_manifest_hashes_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": str(self.task_id),
            "candidate": str(self.candidate),
            "model_version": str(self.model_version),
            "seed": int(self.seed),
            "n_classes": int(self.n_classes),
            "image_shape": list(self.image_shape),
            "config_sha256": str(self.config_sha256),
            "a06_split_sha256": str(self.a06_split_sha256),
            "freeze_manifest_sha256": str(self.freeze_manifest_sha256),
            "train_class_stats_sha256": str(self.train_class_stats_sha256),
            "class_weight_sha256": str(self.class_weight_sha256),
            "input_manifest_hashes_sha256": str(self.input_manifest_hashes_sha256),
        }


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def identity_from_dict(payload: Mapping[str, Any]) -> CheckpointIdentity:
    if "identity" not in payload:
        raise ResumeIdentityError("checkpoint payload missing 'identity' block")
    raw = payload["identity"]
    return CheckpointIdentity(
        task_id=str(raw["task_id"]),
        candidate=str(raw["candidate"]),
        model_version=str(raw["model_version"]),
        seed=int(raw["seed"]),
        n_classes=int(raw["n_classes"]),
        image_shape=tuple(int(v) for v in raw["image_shape"]),
        config_sha256=str(raw["config_sha256"]),
        a06_split_sha256=str(raw["a06_split_sha256"]),
        freeze_manifest_sha256=str(raw["freeze_manifest_sha256"]),
        train_class_stats_sha256=str(raw["train_class_stats_sha256"]),
        class_weight_sha256=str(raw["class_weight_sha256"]),
        input_manifest_hashes_sha256=str(raw["input_manifest_hashes_sha256"]),
    )


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------


def canonical_json_dumps(payload: Any) -> str:
    """Stable JSON dump used for class-weight and input-hash SHA-256."""

    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def class_weight_sha256(weights_payload: Mapping[str, Any]) -> str:
    """Stable SHA-256 of the class-weight vector."""

    text = canonical_json_dumps(weights_payload)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def input_manifest_hashes_sha256(
    input_manifest_hashes_payload: Mapping[str, Any],
) -> str:
    """Stable SHA-256 of the input-manifest-hashes dict."""

    text = canonical_json_dumps(input_manifest_hashes_payload)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Identity comparison
# ---------------------------------------------------------------------------


def verify_resume_identity(
    saved: CheckpointIdentity,
    requested: CheckpointIdentity,
) -> None:
    """Compare two :class:`CheckpointIdentity` blocks field-by-field.

    Raises :class:`ResumeIdentityError` on the first mismatch.  The
    error message lists every mismatch so a Reviewer can audit the
    cause.
    """

    expected = saved.as_dict()
    actual = requested.as_dict()
    diffs: list[str] = []
    for key in sorted(expected):
        if expected[key] != actual.get(key):
            diffs.append(
                f"{key}: saved={expected[key]!r} requested={actual.get(key)!r}"
            )
    if diffs:
        raise ResumeIdentityError(
            "checkpoint identity mismatch; refusing to resume: " + "; ".join(diffs)
        )


# ---------------------------------------------------------------------------
# Refuse to resume a DONE run
# ---------------------------------------------------------------------------


def refuse_resume_for_done_run(output_dir: Path) -> None:
    """Refuse resume for an output directory that already has ``DONE.json``.

    The :class:`ResumeRefusedError` is raised when the file exists.
    The same call is a no-op for an empty or partial output dir.
    """

    output_dir = Path(output_dir)
    if (output_dir / "DONE.json").is_file():
        raise ResumeRefusedError(
            f"output directory {output_dir} already contains DONE.json; "
            "the run is closed and must not be resumed"
        )


# ---------------------------------------------------------------------------
# RNG capture / restore
# ---------------------------------------------------------------------------


def capture_rng_state() -> dict[str, Any]:
    """Capture Python / NumPy / torch RNG state into a JSON-safe form."""

    rng_python = torch.get_rng_state().cpu().tolist()
    cuda_states: list[list[int]] = []
    if torch.cuda.is_available():
        cuda_states = [
            s.cpu().tolist() for s in torch.cuda.get_rng_state_all()
        ]
    np_state = np.random.get_state()
    return {
        "torch": rng_python,
        "torch_cuda": cuda_states,
        "numpy": {
            "legacy": str(np_state[0]),
            "state": np_state[1].tolist(),
            "pos": int(np_state[2]),
            "has_gauss": int(np_state[3]),
            "cached_gaussian": float(np_state[4]),
        },
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore Python / NumPy / torch RNG state captured by
    :func:`capture_rng_state`."""

    torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
    cuda_states = state.get("torch_cuda", [])
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [torch.tensor(s, dtype=torch.uint8) for s in cuda_states]
        )
    np_state = state["numpy"]
    np.random.set_state(
        (
            np_state["legacy"],
            np.asarray(np_state["state"], dtype=np.uint32),
            int(np_state["pos"]),
            int(np_state["has_gauss"]),
            float(np_state["cached_gaussian"]),
        )
    )


# ---------------------------------------------------------------------------
# Early-stopper state (serialized to checkpoint)
# ---------------------------------------------------------------------------


@dataclass
class EarlyStopperState:
    """The serializable state of the B04 :class:`_EarlyStopper`."""

    best_metric: float | None
    best_epoch: int | None
    patience: int
    min_delta: float
    min_epochs: int
    mode: str
    monitor: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_metric": (
                float(self.best_metric) if self.best_metric is not None else None
            ),
            "best_epoch": (
                int(self.best_epoch) if self.best_epoch is not None else None
            ),
            "patience": int(self.patience),
            "min_delta": float(self.min_delta),
            "min_epochs": int(self.min_epochs),
            "mode": str(self.mode),
            "monitor": str(self.monitor),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EarlyStopperState":
        return cls(
            best_metric=(
                float(payload["best_metric"])
                if payload.get("best_metric") is not None
                else None
            ),
            best_epoch=(
                int(payload["best_epoch"])
                if payload.get("best_epoch") is not None
                else None
            ),
            patience=int(payload["patience"]),
            min_delta=float(payload["min_delta"]),
            min_epochs=int(payload["min_epochs"]),
            mode=str(payload["mode"]),
            monitor=str(payload["monitor"]),
        )
