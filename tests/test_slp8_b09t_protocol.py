from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.validate_slp8_b09t_protocol import validate


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiments/slp8_pm_b09t_final_test_protocol_v0.1.json"


def _payload() -> dict:
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_frozen_protocol_passes_without_test_access() -> None:
    assert validate(PROTOCOL) == []
    source = (ROOT / "scripts/validate_slp8_b09t_protocol.py").read_text(encoding="utf-8")
    assert "enable_test_access(" not in source
    assert "load_test=True" not in source


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["authorization_gate"].__setitem__("test_authorized", True),
        lambda d: d["authorization_gate"].__setitem__("execution_authorized", True),
        lambda d: d["authorization_gate"].__setitem__("load_test", True),
        lambda d: d["authorization_gate"].__setitem__("required_purpose_literal", "evaluation"),
        lambda d: d["test_cohort"].__setitem__("expected_samples", 494),
        lambda d: d["checkpoints"][0].__setitem__("sha256", "0" * 64),
        lambda d: d["checkpoints"][1].__setitem__("fixed_epochs", 21),
        lambda d: d["prediction_contract"].__setitem__("class_order", list(range(8))),
        lambda d: d["frozen_metrics"].__setitem__("primary", "pixel_accuracy"),
        lambda d: d["frozen_metrics"].__setitem__("empty_foreground_class_policy", "skip"),
        lambda d: d["anti_adaptation"].__setitem__("no_test_driven_rerun", False),
    ],
)
def test_protocol_drift_fails_closed(tmp_path: Path, mutate) -> None:
    payload = copy.deepcopy(_payload())
    mutate(payload)
    assert validate(_write(tmp_path, payload))
