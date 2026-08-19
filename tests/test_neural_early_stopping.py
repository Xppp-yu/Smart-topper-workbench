"""Unit tests for the P5.2-B early-stopping rule (pure Python, no torch)."""

from __future__ import annotations

import math

import pytest

from topper_perception.neural.early_stopping import EarlyStopper, best_checkpoint_rule


def test_first_epoch_is_best() -> None:
    stopper = EarlyStopper(monitor="val_loss", mode="min", patience=2, min_epochs=1)
    step = stopper.step(1, 0.5)
    assert step.is_best is True
    assert step.should_stop is False


def test_stops_after_patience_non_improving() -> None:
    stopper = EarlyStopper(monitor="val_loss", mode="min", patience=2, min_epochs=1)
    stopper.step(1, 1.0)
    assert stopper.step(2, 1.1).should_stop is False
    assert stopper.step(3, 1.2).should_stop is True


def test_min_epochs_prevents_early_stop() -> None:
    stopper = EarlyStopper(monitor="val_loss", mode="min", patience=1, min_epochs=3)
    stopper.step(1, 1.0)
    assert stopper.step(2, 1.1).should_stop is False
    assert stopper.step(3, 1.2).should_stop is True


def test_improvement_resets_patience() -> None:
    stopper = EarlyStopper(monitor="val_loss", mode="min", patience=2, min_epochs=1)
    stopper.step(1, 1.0)
    stopper.step(2, 1.1)
    assert stopper.step(3, 0.8).is_best is True
    assert stopper.step(4, 1.1).should_stop is False


def test_rejects_non_val_loss_monitor() -> None:
    with pytest.raises(ValueError, match="monitor='val_loss'"):
        EarlyStopper(monitor="val_macro_f1")


@pytest.mark.parametrize("min_delta", [float("nan"), float("inf"), -0.1])
def test_rejects_non_finite_or_negative_min_delta(min_delta: float) -> None:
    with pytest.raises(ValueError, match="min_delta"):
        EarlyStopper(min_delta=min_delta)


@pytest.mark.parametrize("metric", [float("nan"), float("inf"), float("-inf")])
def test_rejects_non_finite_metric(metric: float) -> None:
    stopper = EarlyStopper(monitor="val_loss", mode="min")
    with pytest.raises(ValueError, match="finite"):
        stopper.step(1, metric)


def test_best_checkpoint_rule_mentions_monitor() -> None:
    rule = best_checkpoint_rule({"monitor": "val_loss", "mode": "min", "min_delta": 0.0})
    assert rule.startswith("best = argmin(val_loss)")


def test_zero_min_delta_is_finite() -> None:
    # The default 0.0 must remain acceptable.
    assert math.isfinite(EarlyStopper(min_delta=0.0).min_delta)
