from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import pytest

from scripts.validate_b07_protocol import validate


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiments/slp8_pm_full_protocol_v0.1.json"
FOLDS = ROOT / "configs/experiments/slp8_pm_full_folds_v0.1.json"
B01_FREEZE = ROOT / "data/processed/slp8_training_tables_v0.1/freeze_manifest.json"


def test_frozen_protocol_passes() -> None:
    assert validate(PROTOCOL) == []


def test_folds_cover_each_development_subject_once() -> None:
    payload = json.loads(FOLDS.read_text(encoding="utf-8"))
    ids = [sid for fold in payload["folds"] for sid in fold["val_subject_ids"]]
    assert len(ids) == 91
    assert len(set(ids)) == 91
    assert sum(f["val_sample_count"] for f in payload["folds"]) == 4095


def test_folds_match_b01_train_plus_val_and_exclude_test() -> None:
    payload = json.loads(FOLDS.read_text(encoding="utf-8"))
    freeze = json.loads(B01_FREEZE.read_text(encoding="utf-8"))["core"]["splits"]
    folded = {sid for fold in payload["folds"] for sid in fold["val_subject_ids"]}
    development = set(freeze["train"]["subject_ids"]) | set(freeze["val"]["subject_ids"])
    test_subjects = set(freeze["test"]["subject_ids"])
    assert folded == development
    assert folded.isdisjoint(test_subjects)


def test_test_is_structural_only_and_absent_from_folds() -> None:
    payload = json.loads(FOLDS.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert payload["test_access"] == "DENIED"
    assert payload["invariants"]["test_subjects_in_any_fold"] == 0
    assert protocol["test_access"] == {
        "allowed": False,
        "load_test": False,
        "expected_rows": 0,
        "expected_labels": 0,
        "expected_onehot": 0,
    }


def test_execution_matrix_is_30_units() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    matrix = payload["execution_matrix"]
    assert matrix["total_units"] == matrix["candidates"] * matrix["folds"] * matrix["seeds"] == 30


def test_primary_is_pooled_oof_not_fold_average() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert "pooled" in payload["metrics"]["primary"].lower()
    assert payload["metrics"]["fold_average_is_primary"] is False
    assert payload["metrics"]["failed_units_may_be_dropped"] is False


def test_budget_arithmetic() -> None:
    budget = json.loads(PROTOCOL.read_text(encoding="utf-8"))["resource_budget"]
    assert budget["max_wall_minutes_per_candidate"] == 15 * 5 * 3
    assert budget["max_wall_minutes_total"] == 15 * 5 * 3 * 2
    assert budget["max_peak_cuda_mb"] == 8192


def _variant(
    tmp_path: Path,
    *,
    mutate_protocol: Callable[[dict], None] | None = None,
    mutate_folds: Callable[[dict], None] | None = None,
) -> Path:
    root = tmp_path
    exp = root / "configs/experiments"
    exp.mkdir(parents=True)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))
    if mutate_folds:
        mutate_folds(folds)
    fold_path = exp / FOLDS.name
    fold_path.write_text(json.dumps(folds, indent=2), encoding="utf-8")
    protocol["fold_contract"]["manifest_sha256"] = hashlib.sha256(
        fold_path.read_bytes()
    ).hexdigest()
    if mutate_protocol:
        mutate_protocol(protocol)
    protocol_path = exp / PROTOCOL.name
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    return protocol_path


@pytest.mark.parametrize(
    ("mutate_protocol", "mutate_folds", "expected"),
    [
        (lambda p: p["test_access"].update(allowed=True), None, "TEST must be denied"),
        (None, lambda f: f["folds"][1]["val_subject_ids"].__setitem__(0, f["folds"][0]["val_subject_ids"][0]), "fold subjects must be unique"),
        (lambda p: p["candidates"].reverse(), None, "candidate order/set mismatch"),
        (lambda p: p["resource_budget"].update(max_wall_minutes_total=449), None, "total budget multiplication mismatch"),
        (lambda p: p["metrics"].update(fold_average_is_primary=True), None, "fold average cannot be primary"),
    ],
)
def test_protocol_corruption_fails_closed(
    tmp_path: Path,
    mutate_protocol: Callable[[dict], None] | None,
    mutate_folds: Callable[[dict], None] | None,
    expected: str,
) -> None:
    errors = validate(
        _variant(
            tmp_path,
            mutate_protocol=mutate_protocol,
            mutate_folds=mutate_folds,
        )
    )
    assert expected in errors
