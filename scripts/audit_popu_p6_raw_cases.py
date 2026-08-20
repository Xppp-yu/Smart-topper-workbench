"""Audit P6 priority records against read-only PoPu Tactilus JSON metadata."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def audit_case(row: dict[str, str], raw_root: Path) -> dict[str, object]:
    record_id = row["record_id"]
    path = raw_root / record_id
    result: dict[str, object] = {
        "repeat": int(row["repeat"]), "record_id": record_id,
        "subject_id": row["subject_id"], "oof_y_true": row["y_true"],
        "oof_y_pred": row["y_pred"], "path_exists": path.is_file(),
    }
    if not path.is_file():
        return result
    payload = json.loads(path.read_text(encoding="utf-8"))
    snapshots = payload.get("snapshots", {})
    readings = [item.get("tactilus_readings", []) for item in snapshots.values()]
    flat = [value for values in readings for value in values]
    result.update({
        "raw_position": payload.get("position"),
        "variation": payload.get("variation"),
        "rows": payload.get("tactilus_rows"),
        "columns": payload.get("tactilus_columns"),
        "snapshot_count": len(snapshots),
        "reading_lengths": sorted(set(len(values) for values in readings)),
        "finite_values": all(isinstance(value, (int, float)) and math.isfinite(value) for value in flat),
        "max_pressure": max(flat) if flat else None,
        "nonzero_values": sum(value != 0 for value in flat),
        "label_matches_raw_position": row["y_true"] == payload.get("position"),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.cases.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    unique = {(row["repeat"], row["record_id"]): row for row in rows}
    audited = [audit_case(row, args.raw_root) for row in unique.values()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in audited for key in row})
    with args.output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audited)
    print(json.dumps({
        "cases": len(audited),
        "found": sum(bool(row["path_exists"]) for row in audited),
        "label_matches": sum(bool(row.get("label_matches_raw_position")) for row in audited),
        "shape_ok": sum(row.get("reading_lengths") == [1728] for row in audited),
        "snapshots_10": sum(row.get("snapshot_count") == 10 for row in audited),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
