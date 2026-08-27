"""Resource budget monitor for the B04 PM-only Region Mini.

B04 v0.1 R02 hardens the resource-budget contract:

* Each candidate's wall-clock is measured with
  :func:`time.monotonic` (immune to wall-clock adjustments) and
  compared against ``max_wall_minutes_per_candidate`` after every
  validation epoch.
* The total wall-clock for all candidates is the sum of per-candidate
  deltas; it is compared against ``max_total_wall_minutes`` at the same
  hooks.
* CUDA peak memory is measured with
  :func:`torch.cuda.reset_peak_memory_stats` at the start of each
  candidate and :func:`torch.cuda.max_memory_allocated` after every
  validation epoch; the running peak in MiB is compared against
  ``max_peak_cuda_mb``.
* When any check fails the candidate transitions to the ``STOPPED``
  state (not ``FAILED``) and the run never writes ``DONE.json`` — the
  CLI writes a mutually exclusive ``STOPPED.json`` instead.

This module is **pure** (no I/O, no torch model state).  It is used by
the orchestrator and the CLI; the per-candidate runner queries it via
the helper functions :func:`check_budget` and
:func:`record_epoch_measurements`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping


# ---------------------------------------------------------------------------
# Budget dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResourceBudget:
    """The frozen B04 v0.1 R02 resource budget.

    All time fields are in **seconds**; the caller converts from
    minutes at the JSON config boundary.  CUDA peak memory is in MiB.
    """

    max_wall_seconds_per_candidate: float
    max_wall_seconds_total: float
    max_peak_cuda_mb: float

    def __post_init__(self) -> None:
        if not (isinstance(self.max_wall_seconds_per_candidate, (int, float)) and self.max_wall_seconds_per_candidate > 0):
            raise ValueError(
                f"max_wall_seconds_per_candidate must be a positive number; got {self.max_wall_seconds_per_candidate!r}"
            )
        if not (isinstance(self.max_wall_seconds_total, (int, float)) and self.max_wall_seconds_total > 0):
            raise ValueError(
                f"max_wall_seconds_total must be a positive number; got {self.max_wall_seconds_total!r}"
            )
        if not (isinstance(self.max_peak_cuda_mb, (int, float)) and self.max_peak_cuda_mb > 0):
            raise ValueError(
                f"max_peak_cuda_mb must be a positive number; got {self.max_peak_cuda_mb!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_wall_seconds_per_candidate": float(self.max_wall_seconds_per_candidate),
            "max_wall_seconds_total": float(self.max_wall_seconds_total),
            "max_peak_cuda_mb": float(self.max_peak_cuda_mb),
        }


# ---------------------------------------------------------------------------
# Outcome dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetCheck:
    """Result of a single budget check."""

    exceeded: bool
    reason: str  # "ok" | "per_candidate_wall_exceeded" | "total_wall_exceeded" | "cuda_peak_exceeded"
    elapsed_seconds: float
    total_elapsed_seconds: float
    peak_cuda_mb: float
    threshold_per_candidate_seconds: float
    threshold_total_seconds: float
    threshold_peak_cuda_mb: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "exceeded": bool(self.exceeded),
            "reason": str(self.reason),
            "elapsed_seconds": float(self.elapsed_seconds),
            "total_elapsed_seconds": float(self.total_elapsed_seconds),
            "peak_cuda_mb": float(self.peak_cuda_mb),
            "threshold_per_candidate_seconds": float(self.threshold_per_candidate_seconds),
            "threshold_total_seconds": float(self.threshold_total_seconds),
            "threshold_peak_cuda_mb": float(self.threshold_peak_cuda_mb),
        }


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class ResourceBudgetState:
    """Mutable per-run state that holds the running totals.

    The orchestrator creates one :class:`ResourceBudgetState` per B04
    run and passes it to each candidate.  The candidate queries
    :meth:`check` at every validation epoch; if a check fails the
    candidate transitions to ``STOPPED``.
    """

    def __init__(self, budget: ResourceBudget) -> None:
        if budget.max_wall_seconds_per_candidate <= 0:
            raise ValueError("max_wall_seconds_per_candidate must be positive")
        if budget.max_wall_seconds_total <= 0:
            raise ValueError("max_wall_seconds_total must be positive")
        if budget.max_peak_cuda_mb <= 0:
            raise ValueError("max_peak_cuda_mb must be positive")
        self._budget = budget
        self._t_run_start = time.monotonic()
        self._t_candidate_start = time.monotonic()
        self._candidate_seconds_consumed: float = 0.0
        self._peak_cuda_mb: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin_candidate(self) -> None:
        """Mark the start of a new candidate's wall-clock window.

        The elapsed time of the previous candidate is rolled into the
        total, so a single :class:`ResourceBudgetState` is reused across
        candidates in a B04 run.
        """

        now = time.monotonic()
        elapsed_this_candidate = now - self._t_candidate_start
        self._candidate_seconds_consumed += float(elapsed_this_candidate)
        self._t_candidate_start = now
        self._peak_cuda_mb = 0.0

    def snapshot(self) -> BudgetAccumulatorState:
        """Return the persisted accumulated state for the checkpoint."""

        return BudgetAccumulatorState(
            candidate_seconds_consumed=float(self._candidate_seconds_consumed),
            last_candidate_peak_cuda_mb=float(self._peak_cuda_mb),
        )

    def restore(self, state: BudgetAccumulatorState) -> None:
        """Restore the accumulated state from a checkpoint.

        The wall-clock is reset to "now" because the previous run's
        absolute time stamps are meaningless; only the cumulative
        spent budget and the last-candidate peak memory are carried
        over so the resumed run inherits the same effective budget.
        """

        if state.candidate_seconds_consumed < 0:
            raise ValueError(
                f"negative accumulated time {state.candidate_seconds_consumed}; refusing to restore"
            )
        if state.last_candidate_peak_cuda_mb < 0:
            raise ValueError(
                f"negative peak CUDA MiB {state.last_candidate_peak_cuda_mb}; refusing to restore"
            )
        self._candidate_seconds_consumed = float(state.candidate_seconds_consumed)
        self._peak_cuda_mb = float(state.last_candidate_peak_cuda_mb)
        self._t_candidate_start = time.monotonic()

    @property
    def total_elapsed_seconds(self) -> float:
        """Sum of all per-candidate elapseds plus the current candidate's."""

        return self._candidate_seconds_consumed + self.elapsed_seconds

    @property
    def elapsed_seconds(self) -> float:
        """Seconds spent on the current candidate so far."""

        return float(time.monotonic() - self._t_candidate_start)

    @property
    def peak_cuda_mb(self) -> float:
        return float(self._peak_cuda_mb)

    @property
    def budget(self) -> ResourceBudget:
        return self._budget

    # ------------------------------------------------------------------
    # Measurement helpers
    # ------------------------------------------------------------------

    def update_cuda_peak(self) -> float:
        """Update the running peak CUDA memory and return the new value.

        No-op (and returns 0) when CUDA is unavailable; the threshold is
        still enforced as a no-op so the contract is well-defined.
        """

        try:
            import torch
        except ImportError:  # pragma: no cover - torch is required upstream
            return 0.0
        if not torch.cuda.is_available():
            return float(self._peak_cuda_mb)
        peak_bytes = int(torch.cuda.max_memory_allocated())
        peak_mb = float(peak_bytes) / float(1024 * 1024)
        if peak_mb > self._peak_cuda_mb:
            self._peak_cuda_mb = peak_mb
        return float(self._peak_cuda_mb)

    def reset_cuda_peak(self) -> None:
        """Reset the CUDA peak counter.  No-op without CUDA."""

        try:
            import torch
        except ImportError:  # pragma: no cover
            return
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._peak_cuda_mb = 0.0

    # ------------------------------------------------------------------
    # The check
    # ------------------------------------------------------------------

    def check(self) -> BudgetCheck:
        """Evaluate the budget now and return the outcome.

        The check is fail-closed: any exceeded threshold flips
        ``exceeded=True`` and the candidate MUST transition to
        ``STOPPED``.
        """

        elapsed = self.elapsed_seconds
        total = self.total_elapsed_seconds
        peak = self.peak_cuda_mb
        per_thr = self._budget.max_wall_seconds_per_candidate
        tot_thr = self._budget.max_wall_seconds_total
        peak_thr = self._budget.max_peak_cuda_mb

        # Wall-clock is checked first because a wall-clock violation is
        # the most severe signal (the work cannot continue meaningfully).
        if elapsed > per_thr:
            return BudgetCheck(
                exceeded=True,
                reason="per_candidate_wall_exceeded",
                elapsed_seconds=elapsed,
                total_elapsed_seconds=total,
                peak_cuda_mb=peak,
                threshold_per_candidate_seconds=per_thr,
                threshold_total_seconds=tot_thr,
                threshold_peak_cuda_mb=peak_thr,
            )
        if total > tot_thr:
            return BudgetCheck(
                exceeded=True,
                reason="total_wall_exceeded",
                elapsed_seconds=elapsed,
                total_elapsed_seconds=total,
                peak_cuda_mb=peak,
                threshold_per_candidate_seconds=per_thr,
                threshold_total_seconds=tot_thr,
                threshold_peak_cuda_mb=peak_thr,
            )
        if peak > peak_thr:
            return BudgetCheck(
                exceeded=True,
                reason="cuda_peak_exceeded",
                elapsed_seconds=elapsed,
                total_elapsed_seconds=total,
                peak_cuda_mb=peak,
                threshold_per_candidate_seconds=per_thr,
                threshold_total_seconds=tot_thr,
                threshold_peak_cuda_mb=peak_thr,
            )
        return BudgetCheck(
            exceeded=False,
            reason="ok",
            elapsed_seconds=elapsed,
            total_elapsed_seconds=total,
            peak_cuda_mb=peak,
            threshold_per_candidate_seconds=per_thr,
            threshold_total_seconds=tot_thr,
            threshold_peak_cuda_mb=peak_thr,
        )


# ---------------------------------------------------------------------------
# Construction helper
# ---------------------------------------------------------------------------


def resource_budget_from_config(payload: Mapping[str, Any]) -> ResourceBudget:
    """Build a :class:`ResourceBudget` from the B04 config payload.

    The caller must supply ``max_wall_minutes_per_candidate``,
    ``max_total_wall_minutes`` and ``max_peak_cuda_mb``; the B04
    validation layer already guarantees those fields exist.
    """

    return ResourceBudget(
        max_wall_seconds_per_candidate=float(payload["max_wall_minutes_per_candidate"]) * 60.0,
        max_wall_seconds_total=float(payload["max_total_wall_minutes"]) * 60.0,
        max_peak_cuda_mb=float(payload["max_peak_cuda_mb"]),
    )


# ---------------------------------------------------------------------------
# Snapshot of accumulated budget state (R03)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetAccumulatorState:
    """The serializable accumulated state of :class:`ResourceBudgetState`.

    Persisted into every checkpoint so a resume does not double-count
    time across the previous (interrupted) run and the new run.
    """

    candidate_seconds_consumed: float
    last_candidate_peak_cuda_mb: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_seconds_consumed": float(self.candidate_seconds_consumed),
            "last_candidate_peak_cuda_mb": float(self.last_candidate_peak_cuda_mb),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BudgetAccumulatorState":
        return cls(
            candidate_seconds_consumed=float(
                payload.get("candidate_seconds_consumed", 0.0)
            ),
            last_candidate_peak_cuda_mb=float(
                payload.get("last_candidate_peak_cuda_mb", 0.0)
            ),
        )
