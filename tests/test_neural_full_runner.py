"""Cheap unit tests for P5.2-C Full runner mechanics; never train Full."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from topper_perception.experiments.artifacts import atomic_write_json, sha256_hex
from topper_perception.neural.full import (
    PROBA_COLUMNS,
    aggregate_record_rows,
    validate_candidate_complete,
)


def _snapshot_rows() -> list[dict]:
    probabilities = [0.05, 0.8, 0.05, 0.05, 0.05]
    rows = []
    for index in range(10):
        row = {
            "model": "tiny_cnn",
            "repeat": 0,
            "outer_seed": 11,
            "local_fold": 0,
            "sample_id": f"1/supine1.json#{index}",
            "record_id": "1/supine1.json",
            "subject_id": "1",
            "y_true": "supine",
            "y_pred": "supine",
            "confidence": 0.8,
        }
        row.update(dict(zip(PROBA_COLUMNS, probabilities, strict=True)))
        rows.append(row)
    return rows


def test_record_aggregation_requires_ten_snapshots_and_averages_probabilities() -> None:
    rows = aggregate_record_rows(_snapshot_rows())
    assert len(rows) == 1
    assert rows[0]["record_id"] == "1/supine1.json"
    assert rows[0]["n_snapshots"] == 10
    assert rows[0]["y_pred"] == "supine"
    assert rows[0]["proba__supine"] == pytest.approx(0.8)

    with pytest.raises(ValueError, match="expected 10"):
        aggregate_record_rows(_snapshot_rows()[:-1])


def test_completed_candidate_marker_is_content_hashed(tmp_path: Path) -> None:
    candidate = tmp_path / "folds" / "repeat_0" / "fold_0" / "tiny_cnn"
    candidate.mkdir(parents=True)
    artifact = candidate / "summary.json"
    atomic_write_json(artifact, {"ok": True})
    relative = artifact.relative_to(tmp_path).as_posix()
    marker = {
        "state": "SUCCEEDED",
        "model": "tiny_cnn",
        "repeat": 0,
        "local_fold": 0,
        "split_manifest_sha256": "a" * 64,
        "artifacts": [
            {
                "path": relative,
                "size_bytes": artifact.stat().st_size,
                "sha256": sha256_hex(artifact),
            }
        ],
    }
    atomic_write_json(candidate / "complete.json", marker)

    verified = validate_candidate_complete(
        candidate,
        tmp_path,
        model_name="tiny_cnn",
        repeat=0,
        local_fold=0,
        split_manifest_sha256="a" * 64,
    )
    assert verified == marker

    artifact.write_text(json.dumps({"ok": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity mismatch"):
        validate_candidate_complete(
            candidate,
            tmp_path,
            model_name="tiny_cnn",
            repeat=0,
            local_fold=0,
            split_manifest_sha256="a" * 64,
        )
