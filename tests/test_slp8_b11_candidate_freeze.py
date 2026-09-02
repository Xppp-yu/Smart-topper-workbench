import json
from pathlib import Path

from scripts.validate_slp8_b11_candidate_freeze import validate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/slp8_pm_research_candidate_v0.1.json"


def test_frozen_candidate_passes():
    assert validate(CONFIG) == []


def test_test_access_fails_closed(tmp_path: Path):
    d = json.loads(CONFIG.read_text(encoding="utf-8"))
    d["development_evidence"]["test_access"] = 0
    path = tmp_path / "bad.json"; path.write_text(json.dumps(d), encoding="utf-8")
    assert "TEST must be strict false" in validate(path)


def test_epoch_drift_fails_closed(tmp_path: Path):
    d = json.loads(CONFIG.read_text(encoding="utf-8"))
    d["final_development_fit"]["fixed_epochs_by_seed"]["42"] = 16
    path = tmp_path / "bad.json"; path.write_text(json.dumps(d), encoding="utf-8")
    assert "epoch freeze mismatch" in validate(path)
