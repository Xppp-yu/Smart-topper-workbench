"""Viability gate for the PoPu neural Mini screening (P5.2-B).

Pure Python/``math`` only, so the gate is unit-testable with hand-built summary
dicts. It decides, per candidate model, whether that candidate may continue to
Full screening. Verdicts are limited to exactly three — ``proceed`` /
``exclude`` / ``needs_fix`` — and never express a ranking between candidates.

Mapping from the pre-registered minimum conditions to verdicts:

- **exclude** (candidate not viable as a model): non-finite loss/metrics, or no
  learning signal above the chance baseline. These are candidate-quality
  failures, not protocol bugs.
- **needs_fix** (protocol/infrastructure failure, re-run after fixing): subject
  leakage, models not sharing the same split, missing checkpoints, failed
  resume, failed independent-reload consistency, or a resource-limit breach.
- **proceed**: all conditions hold.

The learning-signal threshold is ``1/num_classes + chance_margin``, computed
from the chance level only — it is identical for every candidate and is *not*
derived from any P5.2-A single-epoch smoke accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping


class ViabilityVerdict(StrEnum):
    PROCEED = "proceed"
    EXCLUDE = "exclude"
    NEEDS_FIX = "needs_fix"


@dataclass(frozen=True, slots=True)
class ViabilityResult:
    """Gate output: a verdict, the failed checks, and human-readable reasons."""

    verdict: str
    reasons: tuple[str, ...]
    checks: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "checks": dict(self.checks),
        }


#: Checks whose failure indicates a protocol/infrastructure problem.
_NEEDS_FIX_CHECKS = (
    "no_leakage",
    "same_split",
    "checkpoint_ok",
    "resume_ok",
    "reload_ok",
    "resource_ok",
)

#: Checks whose failure indicates the candidate model is not viable.
_EXCLUDE_CHECKS = ("finite", "learning_signal_ok")

_REASON = {
    "finite": "non-finite train/val loss or metric",
    "no_leakage": "subject leakage between train/val/test splits",
    "same_split": "models did not share the same subject split",
    "checkpoint_ok": "latest/best checkpoint files missing",
    "resume_ok": "checkpoint resume failed or parameters did not change",
    "reload_ok": "independent reload prediction consistency failed",
    "resource_ok": "training time or GPU memory exceeded resource limits",
}


def _is_finite(value: Any) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _finite_or_inf(value: Any) -> float:
    return float(value) if _is_finite(value) else math.inf


def _learning_reason(num_classes: int, chance_margin: float) -> str:
    threshold = (1.0 / num_classes) + chance_margin
    return f"balanced accuracy did not exceed chance baseline {threshold:.4f}"


def assess_viability(
    summary: Mapping[str, Any],
    *,
    num_classes: int,
    resource_limits: Mapping[str, Any],
    chance_margin: float = 0.05,
) -> ViabilityResult:
    """Assess one model summary and return a :class:`ViabilityResult`.

    ``summary`` is the per-model JSON-safe dict produced by the Mini runner.
    ``resource_limits`` supplies ``max_train_seconds_per_model`` and
    ``max_cuda_mb`` (absent/None means "no limit"). Illegal inputs raise
    ``ValueError`` so a mis-configured gate fails loudly instead of silently
    passing.
    """
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 2:
        raise ValueError("num_classes must be an integer >= 2.")
    if not isinstance(resource_limits, Mapping):
        raise ValueError("resource_limits must be a mapping.")
    chance_margin = float(chance_margin)
    if not math.isfinite(chance_margin) or chance_margin < 0:
        raise ValueError("chance_margin must be a finite number >= 0.")

    finite_fields = (
        "final_train_loss",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "val_balanced_accuracy",
        "best_val_balanced_accuracy",
    )
    finite = all(_is_finite(summary.get(field)) for field in finite_fields)

    no_leakage = summary.get("no_leakage") is True
    same_split = summary.get("same_split") is True
    checkpoint_ok = summary.get("checkpoint_ok") is True
    resume_ok = summary.get("resume_ok") is True
    reload_ok = summary.get("reload_ok") is True

    total_seconds = _finite_or_inf(summary.get("total_train_seconds"))
    max_seconds = _finite_or_inf(resource_limits.get("max_train_seconds_per_model"))
    time_ok = total_seconds <= max_seconds

    peak_cuda = summary.get("peak_cuda_mb")
    max_cuda = resource_limits.get("max_cuda_mb")
    if peak_cuda is None or max_cuda is None:
        cuda_ok = True  # CPU run, or no memory limit configured
    elif _is_finite(peak_cuda) and _is_finite(max_cuda):
        cuda_ok = float(peak_cuda) <= float(max_cuda)
    else:
        cuda_ok = False
    resource_ok = time_ok and cuda_ok

    balanced_accuracy = summary.get("best_val_balanced_accuracy")
    balanced_accuracy = float(balanced_accuracy) if _is_finite(balanced_accuracy) else math.nan
    learning_signal_ok = (
        math.isfinite(balanced_accuracy)
        and balanced_accuracy > (1.0 / num_classes) + chance_margin
    )

    checks = {
        "finite": finite,
        "no_leakage": no_leakage,
        "same_split": same_split,
        "checkpoint_ok": checkpoint_ok,
        "resume_ok": resume_ok,
        "reload_ok": reload_ok,
        "resource_ok": resource_ok,
        "learning_signal_ok": learning_signal_ok,
    }

    failed = [name for name in checks if not checks[name]]
    reasons: list[str] = []
    for name in failed:
        if name == "learning_signal_ok":
            reasons.append(_learning_reason(num_classes, chance_margin))
        else:
            reasons.append(_REASON[name])

    if any(name in failed for name in _NEEDS_FIX_CHECKS):
        verdict = ViabilityVerdict.NEEDS_FIX.value
    elif any(name in failed for name in _EXCLUDE_CHECKS):
        verdict = ViabilityVerdict.EXCLUDE.value
    else:
        verdict = ViabilityVerdict.PROCEED.value

    return ViabilityResult(verdict=verdict, reasons=tuple(reasons), checks=checks)


def overall_verdict(per_model_verdicts: list[str]) -> str:
    """Aggregate per-model verdicts into a single experiment-level verdict.

    ``needs_fix`` dominates (the protocol must be repaired first); otherwise the
    run is ``exclude`` only when *every* candidate is excluded; otherwise at
    least one candidate is viable and the run is ``proceed``. This is a summary,
    never a ranking.
    """
    if not per_model_verdicts:
        raise ValueError("At least one per-model verdict is required.")
    if any(v == ViabilityVerdict.NEEDS_FIX.value for v in per_model_verdicts):
        return ViabilityVerdict.NEEDS_FIX.value
    if all(v == ViabilityVerdict.EXCLUDE.value for v in per_model_verdicts):
        return ViabilityVerdict.EXCLUDE.value
    return ViabilityVerdict.PROCEED.value
