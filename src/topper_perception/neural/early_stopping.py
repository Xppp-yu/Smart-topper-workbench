"""Early-stopping bookkeeping for the PoPu neural Mini runner (P5.2-B).

Pure Python, no ``torch``/NumPy dependency. The stopper tracks a single monitor
metric over validation epochs and reports whether each epoch is a new "best"
(so the caller can checkpoint it) and whether training should stop.

Rules are fixed up front from the resolved config:

- ``monitor`` is fixed to ``val_loss`` (the Mini contract rejects any other
  monitor up front); ``mode`` selects the direction (``min``/``max``);
- an improvement is ``metric < best - min_delta`` (``min``) or
  ``metric > best + min_delta`` (``max``);
- the first epoch is always the initial best;
- training stops only after ``min_epochs`` epochs and after ``patience``
  consecutive non-improving epochs;
- tie-break is "earliest epoch" (a non-strict improvement does not replace best).

The instance holds mutable counters, but it is a small, single-purpose state
machine — callers pass it by reference and treat its returned step records as
the source of truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class EarlyStoppingStep:
    """Outcome of one validation epoch under the early-stopping rule."""

    is_best: bool
    should_stop: bool
    patience_counter: int


class EarlyStopper:
    """Track one monitor metric and decide best-epoch / early-stop per epoch."""

    def __init__(
        self,
        *,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 2,
        min_delta: float = 0.0,
        min_epochs: int = 1,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max'; got {mode!r}.")
        if str(monitor) != "val_loss":
            raise ValueError(
                f"Mini contract supports only monitor='val_loss'; got {monitor!r}. "
                "Selecting a different monitor is not implemented, so reject it up front."
            )
        if not isinstance(patience, int) or isinstance(patience, bool) or patience < 0:
            raise ValueError("patience must be a non-negative integer.")
        if not isinstance(min_epochs, int) or isinstance(min_epochs, bool) or min_epochs < 1:
            raise ValueError("min_epochs must be a positive integer.")
        min_delta = float(min_delta)
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be a finite non-negative number.")

        self.monitor = str(monitor)
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.best_metric: float | None = None
        self.best_epoch: int | None = None
        self._patience: int = 0

    def step(self, epoch: int, metric: float) -> EarlyStoppingStep:
        """Record ``epoch``'s ``metric`` and return whether it is best / stop."""
        if epoch < 1:
            raise ValueError("epoch must be >= 1.")
        metric = float(metric)
        if not math.isfinite(metric):
            raise ValueError("monitor metric must be finite (NaN/Inf rejected).")

        if self.best_metric is None:
            is_best = True
        elif self.mode == "min":
            is_best = metric < self.best_metric - self.min_delta
        else:
            is_best = metric > self.best_metric + self.min_delta

        if is_best:
            self.best_metric = metric
            self.best_epoch = epoch
            self._patience = 0
        else:
            self._patience += 1

        should_stop = (
            epoch >= self.min_epochs
            and not is_best
            and self._patience >= self.patience
        )
        return EarlyStoppingStep(
            is_best=is_best,
            should_stop=should_stop,
            patience_counter=self._patience,
        )


def best_checkpoint_rule(early_stopping: Mapping[str, Any]) -> str:
    """Return the human-readable, fixed best-checkpoint selection rule."""
    monitor = str(early_stopping.get("monitor", "val_loss"))
    mode = str(early_stopping.get("mode", "min"))
    min_delta = float(early_stopping.get("min_delta", 0.0))
    if mode == "max":
        return (
            f"best = argmax({monitor}) over completed epochs; "
            f"improvement = {monitor} > best_{monitor} + {min_delta}; "
            "tie-break: earliest epoch"
        )
    return (
        f"best = argmin({monitor}) over completed epochs; "
        f"improvement = {monitor} < best_{monitor} - {min_delta}; "
        "tie-break: earliest epoch"
    )
