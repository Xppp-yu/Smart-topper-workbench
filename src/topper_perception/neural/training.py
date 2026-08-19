"""Training-loop primitives for the PoPu neural path (P5.2-A2).

Torch-only and imported lazily. Provides a fixed seed helper, CPU/CUDA device
abstraction, one train epoch, and loss/accuracy/probability/prediction
evaluation with provenance.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from topper_perception.neural.data import NUM_CLASSES


def set_seed(seed: int) -> None:
    """Seed Python/NumPy/torch (and CUDA, when present) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(spec: str = "auto") -> torch.device:
    """Return the requested device; ``"auto"`` prefers CUDA and falls back to CPU."""
    spec = str(spec).strip().lower()
    if spec in ("auto", ""):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Device 'cuda' was requested but torch.cuda.is_available() is False."
            )
        return torch.device("cuda")
    raise ValueError(f"Unknown device spec {spec!r}; expected 'auto', 'cpu', or 'cuda'.")


def make_optimizer(model: nn.Module, *, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Return an AdamW optimizer over ``model``'s parameters."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def make_criterion() -> nn.Module:
    """Return the multi-class CrossEntropyLoss for ``[N, 5]`` logits."""
    return nn.CrossEntropyLoss()


@dataclass(frozen=True)
class PredictionResult:
    """Forward/eval output with provenance, kept as CPU NumPy arrays."""

    loss: float
    accuracy: float
    n_samples: int
    logits: np.ndarray  # [N, 5]
    probabilities: np.ndarray  # [N, 5] softmax
    predictions: np.ndarray  # [N] argmax
    labels: np.ndarray  # [N]
    sample_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    subject_ids: tuple[str, ...]


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    amp_enabled: bool = False,
) -> dict[str, Any]:
    """Run one training epoch and return mean loss, sample count, and AMP flag.

    AMP (mixed precision) is only active on CUDA; on CPU it is a no-op so the
    CPU smoke runs the plain fp32 path.
    """
    model.train()
    use_amp = bool(amp_enabled) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None
    total_loss = 0.0
    n = 0
    for batch in dataloader:
        x = batch["matrix"].to(device)
        y = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast(device_type="cuda", enabled=True):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().cpu().item()) * int(x.size(0))
        n += int(x.size(0))
    return {
        "loss": total_loss / n if n else float("nan"),
        "samples": n,
        "amp_active": use_amp,
    }


def _collect(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str], list[str]]:
    """Return concatenated CPU logits/labels and provenance lists."""
    model.eval()
    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []
    sample_ids: list[str] = []
    record_ids: list[str] = []
    subject_ids: list[str] = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch["matrix"].to(device)
            logits_list.append(model(x).detach().cpu())
            labels_list.append(batch["label"].cpu())
            sample_ids.extend(batch["sample_id"])
            record_ids.extend(batch["record_id"])
            subject_ids.extend(batch["subject_id"])
    logits = torch.cat(logits_list, dim=0) if logits_list else torch.empty((0, NUM_CLASSES))
    labels = torch.cat(labels_list, dim=0) if labels_list else torch.empty((0,), dtype=torch.int64)
    return logits, labels, sample_ids, record_ids, subject_ids


def _result(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: list[str],
    record_ids: list[str],
    subject_ids: list[str],
    *,
    loss: float,
) -> PredictionResult:
    probabilities = torch.softmax(logits, dim=1).numpy().astype(np.float32)
    predictions = logits.argmax(dim=1).numpy().astype(np.int64)
    labels_np = labels.numpy().astype(np.int64)
    n = int(labels.numel())
    accuracy = float((predictions == labels_np).mean()) if n else float("nan")
    return PredictionResult(
        loss=loss,
        accuracy=accuracy,
        n_samples=n,
        logits=logits.numpy().astype(np.float32),
        probabilities=probabilities,
        predictions=predictions,
        labels=labels_np,
        sample_ids=tuple(sample_ids),
        record_ids=tuple(record_ids),
        subject_ids=tuple(subject_ids),
    )


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> PredictionResult:
    """Evaluate loss/accuracy and collect probabilities/predictions/provenance."""
    logits, labels, sample_ids, record_ids, subject_ids = _collect(model, dataloader, device)
    loss = float(criterion(logits, labels).detach().cpu().item()) if labels.numel() else float("nan")
    return _result(
        logits, labels, sample_ids, record_ids, subject_ids, loss=loss,
    )


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> PredictionResult:
    """Forward-only inference; ``loss`` is ``NaN`` (no criterion is applied).

    ``accuracy`` is still computed against the true labels (it needs no
    criterion), but no backward pass is performed.
    """
    logits, labels, sample_ids, record_ids, subject_ids = _collect(model, dataloader, device)
    return _result(
        logits, labels, sample_ids, record_ids, subject_ids, loss=float("nan"),
    )
