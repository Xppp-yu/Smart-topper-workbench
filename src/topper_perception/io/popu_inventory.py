"""Streaming structural inventory for PoPu Tactilus JSON records.

This module deliberately inventories one JSON record at a time.  It never
builds an in-memory list of pressure matrices, so a full pass is bounded by
the size of one source JSON plus the compact CSV rows that are eventually
written by the caller.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any


POPU_POSTURES = ("empty", "supine", "prone", "left", "right")

INVENTORY_COLUMNS = (
    "sample_id",
    "source_relative_path",
    "source_file",
    "source_file_bytes",
    "source_sha256",
    "subject_id",
    "subject_directory",
    "posture",
    "variation",
    "rows",
    "columns",
    "declared_cell_count",
    "snapshot_count",
    "valid_snapshot_count",
    "invalid_snapshot_count",
    "minimum_reading_count",
    "maximum_reading_count",
    "status",
    "error_codes",
    "warning_codes",
)


@dataclass(frozen=True, slots=True)
class InventoryRow:
    """One compact, traceable record-level result from the inventory pass."""

    values: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {column: self.values.get(column, "") for column in INVENTORY_COLUMNS}


def resolve_tactilus_root(data_root: Path) -> Path:
    """Accept either the PoPu root or its ``tactilus_data`` child directory."""
    root = data_root.expanduser()
    nested = root / "tactilus_data"
    if nested.is_dir():
        return nested
    if root.name == "tactilus_data" and root.is_dir():
        return root
    raise FileNotFoundError(f"PoPu Tactilus directory was not found under: {data_root}")


def iter_tactilus_record_paths(data_root: Path) -> Iterator[Path]:
    """Yield Tactilus JSON paths in deterministic order without reading them."""
    tactilus_root = resolve_tactilus_root(data_root)
    yield from sorted(
        tactilus_root.rglob("*.json"),
        key=lambda path: path.relative_to(tactilus_root).as_posix().casefold(),
    )


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)  # PoPu metadata may encode values as strings.
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _normalise_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalise_posture(value: object) -> str | None:
    text = _normalise_text(value)
    return text.lower() if text is not None else None


def _status(errors: Sequence[str], warnings: Sequence[str]) -> str:
    if errors:
        return "ERROR"
    if warnings:
        return "WARN"
    return "OK"


def _base_values(source_file: Path, tactilus_root: Path, *, include_sha256: bool) -> dict[str, object]:
    relative_path = source_file.relative_to(tactilus_root).as_posix()
    raw_bytes = source_file.read_bytes()
    return {
        "sample_id": f"popu-tactilus::{relative_path}",
        "source_relative_path": relative_path,
        "source_file": str(source_file.resolve()),
        "source_file_bytes": len(raw_bytes),
        "source_sha256": sha256(raw_bytes).hexdigest() if include_sha256 else "",
        "_raw_bytes": raw_bytes,
    }


def inventory_tactilus_record(
    source_file: Path,
    tactilus_root: Path,
    *,
    include_sha256: bool = False,
) -> InventoryRow:
    """Inspect one source record and return its structural status.

    A malformed record is represented as an ``ERROR`` row instead of aborting
    the full inventory.  This makes missing or damaged records visible in the
    eventual CSV and summary rather than silently dropping them.
    """
    values = _base_values(source_file, tactilus_root, include_sha256=include_sha256)
    raw_bytes = values.pop("_raw_bytes")
    errors: list[str] = []
    warnings: list[str] = []
    values.update({column: "" for column in INVENTORY_COLUMNS if column not in values})
    values["subject_directory"] = source_file.parent.name

    try:
        record = json.loads(bytes(raw_bytes).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        errors.append("malformed_json")
        values["status"] = _status(errors, warnings)
        values["error_codes"] = ";".join(errors)
        values["warning_codes"] = ";".join(warnings)
        return InventoryRow(values)

    if not isinstance(record, dict):
        errors.append("record_not_object")
        values["status"] = _status(errors, warnings)
        values["error_codes"] = ";".join(errors)
        values["warning_codes"] = ";".join(warnings)
        return InventoryRow(values)

    subject_id = _normalise_text(record.get("volunteer_id"))
    posture = _normalise_posture(record.get("position"))
    variation = _normalise_text(record.get("variation"))
    rows = _positive_int(record.get("tactilus_rows"))
    columns = _positive_int(record.get("tactilus_columns"))
    values.update(
        {
            "subject_id": subject_id or "",
            "posture": posture or "",
            "variation": variation or "",
            "rows": rows or "",
            "columns": columns or "",
            "declared_cell_count": rows * columns if rows and columns else "",
        }
    )

    if subject_id is None:
        warnings.append("missing_subject_id")
    elif subject_id != source_file.parent.name:
        warnings.append("subject_directory_mismatch")
    if posture is None:
        warnings.append("missing_posture")
    elif posture not in POPU_POSTURES:
        warnings.append("unexpected_posture")
    if variation is None:
        warnings.append("missing_variation")
    if rows is None or columns is None:
        errors.append("invalid_matrix_shape")

    snapshots = record.get("snapshots")
    if not isinstance(snapshots, dict) or not snapshots:
        errors.append("missing_or_invalid_snapshots")
        values["snapshot_count"] = 0
        values["valid_snapshot_count"] = 0
        values["invalid_snapshot_count"] = 0
    else:
        reading_counts: list[int] = []
        valid_snapshots = 0
        invalid_snapshots = 0
        expected_cells = rows * columns if rows and columns else None

        for snapshot in snapshots.values():
            snapshot_ok = True
            if not isinstance(snapshot, dict):
                errors.append("snapshot_not_object")
                invalid_snapshots += 1
                continue
            readings = snapshot.get("tactilus_readings")
            if not isinstance(readings, list):
                errors.append("missing_or_invalid_readings")
                invalid_snapshots += 1
                continue

            reading_counts.append(len(readings))
            if expected_cells is not None and len(readings) != expected_cells:
                errors.append("reading_count_mismatch")
                snapshot_ok = False
            for value in readings:
                if isinstance(value, bool):
                    errors.append("non_numeric_or_non_finite_reading")
                    snapshot_ok = False
                    break
                try:
                    if not math.isfinite(float(value)):
                        errors.append("non_numeric_or_non_finite_reading")
                        snapshot_ok = False
                        break
                except (TypeError, ValueError):
                    errors.append("non_numeric_or_non_finite_reading")
                    snapshot_ok = False
                    break
            if snapshot_ok:
                valid_snapshots += 1
            else:
                invalid_snapshots += 1

        values.update(
            {
                "snapshot_count": len(snapshots),
                "valid_snapshot_count": valid_snapshots,
                "invalid_snapshot_count": invalid_snapshots,
                "minimum_reading_count": min(reading_counts) if reading_counts else "",
                "maximum_reading_count": max(reading_counts) if reading_counts else "",
            }
        )

    values["status"] = _status(errors, warnings)
    values["error_codes"] = ";".join(sorted(set(errors)))
    values["warning_codes"] = ";".join(sorted(set(warnings)))
    return InventoryRow(values)


def inventory_tactilus_dataset(
    data_root: Path,
    *,
    include_sha256: bool = False,
) -> Iterator[InventoryRow]:
    """Yield inventory rows one-by-one; callers choose where to persist them."""
    tactilus_root = resolve_tactilus_root(data_root)
    for source_file in iter_tactilus_record_paths(tactilus_root):
        yield inventory_tactilus_record(
            source_file,
            tactilus_root,
            include_sha256=include_sha256,
        )


def summarise_inventory(rows: Iterable[InventoryRow], *, tactilus_root: Path) -> dict[str, object]:
    """Build a JSON-safe summary from compact rows, not source pressure maps."""
    row_list = [row.as_dict() for row in rows]
    status_counts = Counter(str(row["status"]) for row in row_list)
    posture_counts = Counter(str(row["posture"] or "missing") for row in row_list)
    variation_counts = Counter(str(row["variation"] or "missing") for row in row_list)
    shape_counts = Counter(
        f"{row['rows']}x{row['columns']}"
        if row["rows"] != "" and row["columns"] != ""
        else "missing_or_invalid"
        for row in row_list
    )
    error_counts = Counter(
        code
        for row in row_list
        for code in str(row["error_codes"]).split(";")
        if code
    )
    sample_ids = [str(row["sample_id"]) for row in row_list]
    duplicate_sample_ids = sorted(
        sample_id for sample_id, count in Counter(sample_ids).items() if count > 1
    )

    return {
        "dataset": "PoPu",
        "sensor_layer": "tactilus",
        "tactilus_root": str(tactilus_root.resolve()),
        "records": len(row_list),
        "unique_subjects": len({str(row["subject_id"]) for row in row_list if row["subject_id"]}),
        "status_counts": dict(sorted(status_counts.items())),
        "posture_counts": dict(sorted(posture_counts.items())),
        "variation_counts": dict(sorted(variation_counts.items())),
        "shape_counts": dict(sorted(shape_counts.items())),
        "error_counts": dict(sorted(error_counts.items())),
        "duplicate_sample_ids": duplicate_sample_ids,
    }
