from __future__ import annotations

import json
from pathlib import Path

from topper_perception.io.popu_inventory import (
    inventory_tactilus_dataset,
    inventory_tactilus_record,
    summarise_inventory,
)


def _write_record(
    path: Path,
    *,
    readings: list[object],
    position: str = "left",
    subject_id: str = "1",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "volunteer_id": subject_id,
                "position": position,
                "variation": "1",
                "tactilus_rows": 2,
                "tactilus_columns": 3,
                "snapshots": {"0": {"tactilus_readings": readings}},
            }
        ),
        encoding="utf-8",
    )


def test_inventory_reports_a_structurally_valid_record(tmp_path: Path) -> None:
    tactilus_root = tmp_path / "tactilus_data"
    source = tactilus_root / "1" / "left1_0.json"
    _write_record(source, readings=[0, 1, 2, 3, 4, 5])

    row = inventory_tactilus_record(source, tactilus_root, include_sha256=True).as_dict()

    assert row["sample_id"] == "popu-tactilus::1/left1_0.json"
    assert row["status"] == "OK"
    assert row["rows"] == 2
    assert row["columns"] == 3
    assert row["snapshot_count"] == 1
    assert len(str(row["source_sha256"])) == 64


def test_inventory_retains_bad_record_as_an_error_row(tmp_path: Path) -> None:
    tactilus_root = tmp_path / "tactilus_data"
    source = tactilus_root / "1" / "bad.json"
    _write_record(source, readings=[0, 1, 2])

    row = inventory_tactilus_record(source, tactilus_root).as_dict()

    assert row["status"] == "ERROR"
    assert "reading_count_mismatch" in str(row["error_codes"])


def test_dataset_inventory_and_summary_keep_each_source_record(tmp_path: Path) -> None:
    tactilus_root = tmp_path / "tactilus_data"
    _write_record(tactilus_root / "1" / "left1_0.json", readings=[0] * 6)
    _write_record(
        tactilus_root / "2" / "other.json",
        readings=[0] * 6,
        position="other",
        subject_id="2",
    )

    rows = list(inventory_tactilus_dataset(tmp_path))
    summary = summarise_inventory(rows, tactilus_root=tactilus_root)

    assert len(rows) == 2
    assert summary["records"] == 2
    assert summary["unique_subjects"] == 2
    assert summary["status_counts"] == {"OK": 1, "WARN": 1}
    assert summary["posture_counts"] == {"left": 1, "other": 1}
