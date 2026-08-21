"""Build the SLP frame-level master index and integrity summary."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from topper_perception.healthcheck import load_path_config
from topper_perception.io.slp_frame_index import (
    FRAME_INDEX_COLUMNS,
    build_slp_frame_index,
    validate_frame_index_rows,
)
from topper_perception.io.slp_inventory import resolve_slp_root


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "paths.local.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/slp/slp_frame_index_v0.1.csv"),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("outputs/reports/slp_frame_index_summary_v0.1.json"),
    )
    return parser


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FRAME_INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = load_path_config(args.config)
    if "slp_data" not in paths:
        raise KeyError(f"slp_data is missing from path config: {args.config}")

    slp_root = resolve_slp_root(paths["slp_data"])
    frame_rows = list(build_slp_frame_index(slp_root))
    summary = validate_frame_index_rows(frame_rows)
    summary.update(
        {
            "dataset": "SLP",
            "task_id": "TASK-SLP-A03-FRAME-MASTER-INDEX-v0.1",
            "slp_root": str(slp_root.resolve()),
            "grain": "setting x subject_id x cover_condition x frame_index",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )

    output = _project_path(args.output)
    summary_output = _project_path(args.summary_output)
    _write_csv([row.as_dict() for row in frame_rows], output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Fail closed on duplicate master keys or ambiguous modality/frame matches.
    if summary["duplicate_primary_key_count"]:
        return 2
    if summary["ambiguous_modality_frame_counts"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
