"""Read PoPu Tactilus JSON records without modifying the source dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


POPU_POSTURES = ("empty", "supine", "prone", "left", "right")


@dataclass(frozen=True, slots=True)
class PopuTactilusFrame:
    """One PoPu Tactilus snapshot restored as a two-dimensional matrix."""

    source_file: Path
    subject_id: str
    posture: str | None
    variation: str | None
    snapshot_key: str
    snapshot_id: str
    values: NDArray[np.float32]

    @property
    def rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def columns(self) -> int:
        return int(self.values.shape[1])

    def metadata(self) -> dict[str, Any]:
        """Return JSON-safe provenance and simple raw-signal statistics."""
        return {
            "dataset": "PoPu",
            "sensor_layer": "tactilus",
            "subject_id": self.subject_id,
            "posture": self.posture,
            "variation": self.variation,
            "snapshot_key": self.snapshot_key,
            "snapshot_id": self.snapshot_id,
            "rows": self.rows,
            "columns": self.columns,
            "source_file": str(self.source_file),
            "minimum": float(np.min(self.values)),
            "maximum": float(np.max(self.values)),
            "mean": float(np.mean(self.values)),
            "total_signal": float(np.sum(self.values)),
            "nonzero_cells": int(np.count_nonzero(self.values)),
        }


def _tactilus_root(data_root: Path) -> Path:
    root = data_root.expanduser()
    nested = root / "tactilus_data"
    if nested.is_dir():
        return nested
    if root.name == "tactilus_data" and root.is_dir():
        return root
    raise FileNotFoundError(
        f"PoPu Tactilus directory was not found under: {data_root}"
    )


def _snapshot_sort_key(item: tuple[str, object]) -> tuple[int, int | str]:
    key = str(item[0])
    return (0, int(key)) if key.isdigit() else (1, key)


def load_tactilus_record(source_file: Path) -> list[PopuTactilusFrame]:
    """Load every snapshot in one PoPu Tactilus JSON record."""
    source_file = source_file.expanduser()
    record = json.loads(source_file.read_text(encoding="utf-8"))

    try:
        rows = int(record["tactilus_rows"])
        columns = int(record["tactilus_columns"])
        subject_id = str(record["volunteer_id"])
        snapshots = record["snapshots"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid PoPu Tactilus metadata: {source_file}") from exc

    if rows <= 0 or columns <= 0:
        raise ValueError(f"Invalid matrix shape ({rows}, {columns}): {source_file}")
    if not isinstance(snapshots, dict) or not snapshots:
        raise ValueError(f"No snapshots found in: {source_file}")

    raw_posture = record.get("position")
    posture = None if raw_posture is None else str(raw_posture).strip().lower()
    raw_variation = record.get("variation")
    variation = None if raw_variation is None else str(raw_variation)

    frames: list[PopuTactilusFrame] = []
    for snapshot_key, snapshot in sorted(snapshots.items(), key=_snapshot_sort_key):
        if not isinstance(snapshot, dict) or "tactilus_readings" not in snapshot:
            raise ValueError(
                f"Snapshot {snapshot_key!r} has no tactilus_readings: {source_file}"
            )

        flat = np.asarray(snapshot["tactilus_readings"], dtype=np.float32)
        expected = rows * columns
        if flat.ndim != 1 or flat.size != expected:
            raise ValueError(
                f"Snapshot {snapshot_key!r} expected {expected} readings, "
                f"got shape {flat.shape}: {source_file}"
            )
        if not np.isfinite(flat).all():
            raise ValueError(
                f"Snapshot {snapshot_key!r} contains NaN or infinity: {source_file}"
            )

        frames.append(
            PopuTactilusFrame(
                source_file=source_file.resolve(),
                subject_id=subject_id,
                posture=posture,
                variation=variation,
                snapshot_key=str(snapshot_key),
                snapshot_id=str(snapshot.get("id", snapshot_key)),
                values=flat.reshape(rows, columns),
            )
        )

    return frames


def find_tactilus_records(
    data_root: Path,
    *,
    subject_id: str | int,
    posture: str | None = None,
    variation: str | int | None = None,
) -> list[Path]:
    """Find matching records in deterministic filename order."""
    subject_dir = _tactilus_root(data_root) / str(subject_id)
    if not subject_dir.is_dir():
        raise FileNotFoundError(f"PoPu subject directory was not found: {subject_dir}")

    wanted_posture = None if posture is None else posture.strip().lower()
    wanted_variation = None if variation is None else str(variation)
    matches: list[Path] = []

    for path in sorted(subject_dir.glob("*.json"), key=lambda item: item.name.casefold()):
        record = json.loads(path.read_text(encoding="utf-8"))
        raw_posture = record.get("position")
        record_posture = None if raw_posture is None else str(raw_posture).lower()
        raw_variation = record.get("variation")
        record_variation = None if raw_variation is None else str(raw_variation)

        if wanted_posture is not None and record_posture != wanted_posture:
            continue
        if wanted_variation is not None and record_variation != wanted_variation:
            continue
        matches.append(path)

    if not matches:
        raise FileNotFoundError(
            "No matching PoPu Tactilus record: "
            f"subject={subject_id}, posture={posture}, variation={variation}"
        )
    return matches


def select_tactilus_frame(
    data_root: Path,
    *,
    subject_id: str | int,
    posture: str,
    variation: str | int | None = None,
    record_index: int = 0,
    frame_index: int = 0,
) -> PopuTactilusFrame:
    """Select one deterministic record and frame for inspection."""
    if record_index < 0 or frame_index < 0:
        raise ValueError("record_index and frame_index must be non-negative.")

    records = find_tactilus_records(
        data_root,
        subject_id=subject_id,
        posture=posture,
        variation=variation,
    )
    if record_index >= len(records):
        raise IndexError(
            f"record_index={record_index} is out of range for {len(records)} matches."
        )

    frames = load_tactilus_record(records[record_index])
    if frame_index >= len(frames):
        raise IndexError(
            f"frame_index={frame_index} is out of range for {len(frames)} frames."
        )
    return frames[frame_index]

