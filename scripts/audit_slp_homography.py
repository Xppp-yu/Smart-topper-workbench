from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from topper_perception.io.slp_homography_audit import (
    audit_slp_homographies,
    summarise_homography_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SLP homography audit.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=Path("outputs/analysis/slp_homography_audit_v0.1.csv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/reports/slp_homography_audit_summary_v0.1.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(audit_slp_homographies(args.data_root))
    records = [row.as_dict() for row in rows]

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(records).to_csv(args.csv, index=False)
    with args.summary.open("w", encoding="utf-8") as handle:
        json.dump(
            summarise_homography_audit(rows),
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(json.dumps(summarise_homography_audit(rows), ensure_ascii=False, indent=2))
    print(f"csv={args.csv}")
    print(f"summary={args.summary}")


if __name__ == "__main__":
    main()
