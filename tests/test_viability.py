"""Unit tests for the P5.2-B viability gate (pure Python, no torch)."""

from __future__ import annotations

import pytest

from topper_perception.neural.viability import assess_viability, overall_verdict


def _summary(**overrides) -> dict:
    summary = {
        "final_train_loss": 0.5,
        "val_loss": 0.6,
        "val_accuracy": 0.8,
        "val_macro_f1": 0.7,
        "val_balanced_accuracy": 0.75,
        "best_val_balanced_accuracy": 0.78,
        "no_leakage": True,
        "same_split": True,
        "checkpoint_ok": True,
        "resume_ok": True,
        "reload_ok": True,
        "total_train_seconds": 10.0,
        "peak_cuda_mb": 100.0,
    }
    summary.update(overrides)
    return summary


def _limits(**overrides) -> dict:
    limits = {"max_train_seconds_per_model": 300, "max_cuda_mb": 8000}
    limits.update(overrides)
    return limits


def test_proceed_when_all_checks_pass() -> None:
    result = assess_viability(_summary(), num_classes=5, resource_limits=_limits())
    assert result.verdict == "proceed"
    assert result.reasons == ()
    assert all(result.checks.values())


def test_exclude_when_learning_signal_below_chance() -> None:
    result = assess_viability(
        _summary(best_val_balanced_accuracy=0.2), num_classes=5, resource_limits=_limits()
    )
    assert result.verdict == "exclude"
    assert result.checks["learning_signal_ok"] is False


def test_exclude_when_non_finite() -> None:
    result = assess_viability(
        _summary(final_train_loss=float("nan")), num_classes=5, resource_limits=_limits()
    )
    assert result.verdict == "exclude"
    assert result.checks["finite"] is False


@pytest.mark.parametrize(
    "override",
    [
        {"no_leakage": False},
        {"same_split": False},
        {"checkpoint_ok": False},
        {"resume_ok": False},
        {"reload_ok": False},
    ],
)
def test_needs_fix_when_protocol_check_fails(override: dict) -> None:
    result = assess_viability(
        _summary(**override), num_classes=5, resource_limits=_limits()
    )
    assert result.verdict == "needs_fix"


def test_needs_fix_when_time_exceeded() -> None:
    result = assess_viability(
        _summary(total_train_seconds=999.0), num_classes=5, resource_limits=_limits()
    )
    assert result.verdict == "needs_fix"
    assert result.checks["resource_ok"] is False


def test_needs_fix_when_cuda_memory_exceeded() -> None:
    result = assess_viability(
        _summary(peak_cuda_mb=9000.0), num_classes=5, resource_limits=_limits()
    )
    assert result.verdict == "needs_fix"
    assert result.checks["resource_ok"] is False


def test_cpu_run_without_cuda_limit_is_resource_ok() -> None:
    # peak_cuda_mb=None (CPU) and no max_cuda_mb configured -> no memory limit.
    result = assess_viability(
        _summary(peak_cuda_mb=None),
        num_classes=5,
        resource_limits={"max_train_seconds_per_model": 300},
    )
    assert result.checks["resource_ok"] is True
    assert result.verdict == "proceed"


def test_needs_fix_when_cuda_missing_peak_memory() -> None:
    # A CUDA run must report its peak memory; a missing value is needs_fix, not
    # a silent pass (and not a CPU fallback).
    result = assess_viability(
        _summary(device="cuda", peak_cuda_mb=None),
        num_classes=5,
        resource_limits=_limits(),
    )
    assert result.checks["resource_ok"] is False
    assert result.verdict == "needs_fix"


def test_cuda_run_with_peak_memory_is_resource_ok() -> None:
    result = assess_viability(
        _summary(device="cuda", peak_cuda_mb=100.0),
        num_classes=5,
        resource_limits=_limits(),
    )
    assert result.checks["resource_ok"] is True


@pytest.mark.parametrize("num_classes", [1, 0, "5", True, 2.5])
def test_invalid_num_classes_raises(num_classes) -> None:
    with pytest.raises(ValueError):
        assess_viability(_summary(), num_classes=num_classes, resource_limits=_limits())


@pytest.mark.parametrize("chance_margin", [-0.1, float("nan"), float("inf")])
def test_invalid_chance_margin_raises(chance_margin) -> None:
    with pytest.raises(ValueError):
        assess_viability(
            _summary(),
            num_classes=5,
            resource_limits=_limits(),
            chance_margin=chance_margin,
        )


def test_invalid_resource_limits_raises() -> None:
    with pytest.raises(ValueError):
        assess_viability(_summary(), num_classes=5, resource_limits=[1, 2])


def test_overall_verdict_needs_fix_dominates() -> None:
    assert overall_verdict(["proceed", "needs_fix", "exclude"]) == "needs_fix"


def test_overall_verdict_exclude_only_if_all_excluded() -> None:
    assert overall_verdict(["exclude", "exclude"]) == "exclude"


def test_overall_verdict_proceed_otherwise() -> None:
    assert overall_verdict(["proceed", "exclude"]) == "proceed"
    assert overall_verdict(["proceed"]) == "proceed"


def test_overall_verdict_empty_raises() -> None:
    with pytest.raises(ValueError):
        overall_verdict([])


def test_overall_verdict_unknown_verdict_raises() -> None:
    with pytest.raises(ValueError, match="Unknown per-model verdict"):
        overall_verdict(["proceed", "bogus"])
