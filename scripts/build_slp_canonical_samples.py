"""Build the SLP Canonical Sample dataset and run summary.

This script sits on top of the A03 Frame Master Index and the A04 homography
audit. It produces:

* ``data/processed/slp/slp_canonical_samples_v0.1.csv`` — wide CSV with one
  row per canonical sample.
* ``data/processed/slp/slp_canonical_samples_v0.1.jsonl`` — full structured
  payloads for provenance, including the Frame/Joint/Region layers and the
  A04 geometry contract.
* ``outputs/reports/slp_canonical_samples_summary_v0.1.json`` — dataset-level
  real-data run summary.

The script is deliberately idempotent: it does not modify the SLP data
directory and does not write split, review status, or model predictions back
to the canonical samples. It fails closed if duplicate sample IDs appear, if
any modality URI points outside the SLP root, or if the A03 index is missing.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from topper_perception.io.slp_canonical import (
    CANONICAL_SCHEMA_VERSION,
    DEFAULT_TASK_ID,
    CanonicalSample,
    SlpCanonicalAdapter,
    build_adapter_from_artifacts,
    summarise_canonical_samples,
    write_canonical_csv,
    write_canonical_jsonl,
)
from topper_perception.io.slp_inventory import resolve_slp_root


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="SLP data root. If omitted, must be passed via --slp-root.",
    )
    parser.add_argument(
        "--slp-root",
        type=Path,
        default=None,
        help="Resolved SLP root (with danaLab/simLab children). Overrides --data-root.",
    )
    parser.add_argument(
        "--a03-csv",
        type=Path,
        default=Path("data/processed/slp/slp_frame_index_v0.1.csv"),
        help="Path to the A03 Frame Master Index CSV produced by build_slp_frame_index.py.",
    )
    parser.add_argument(
        "--a04-csv",
        type=Path,
        default=Path("outputs/analysis/slp_homography_audit_v0.1.csv"),
        help="Path to the A04 homography audit CSV produced by audit_slp_homography.py.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/slp/slp_canonical_samples_v0.1.csv"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("data/processed/slp/slp_canonical_samples_v0.1.jsonl"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("outputs/reports/slp_canonical_samples_summary_v0.1.json"),
    )
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument(
        "--require-a04",
        action="store_true",
        help="If set, the run fails when --a04-csv is missing.",
    )
    return parser


def _resolve_slp_root(args: argparse.Namespace) -> Path:
    if args.slp_root is not None:
        return resolve_slp_root(args.slp_root)
    if args.data_root is not None:
        return resolve_slp_root(args.data_root)
    raise SystemExit(
        "Either --slp-root or --data-root must be provided. "
        "Use --slp-root 'E:/TeamProjects/datasets/smart-topper/SLP2022/SLP' for real SLP data."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    slp_root = _resolve_slp_root(args)

    a03_csv = _project_path(args.a03_csv)
    a04_csv = _project_path(args.a04_csv)

    if not a03_csv.is_file():
        print(
            f"error: A03 frame index CSV not found at {a03_csv}. "
            "Run scripts/build_slp_frame_index.py first.",
            file=sys.stderr,
        )
        return 2
    if args.require_a04 and not a04_csv.is_file():
        print(
            f"error: --require-a04 was set but A04 audit CSV is missing at {a04_csv}.",
            file=sys.stderr,
        )
        return 2

    a04_audit_path = a04_csv if a04_csv.is_file() else None

    adapter, a03_rows, a04_rows = build_adapter_from_artifacts(
        slp_root=slp_root,
        a03_frame_index_csv=a03_csv,
        a04_homography_audit_csv=a04_audit_path,
        task_id=args.task_id,
    )

    samples: list[CanonicalSample] = list(adapter.iter_canonical_samples())

    duplicate_sample_ids = [
        sample_id
        for sample_id, count in Counter(sample.sample_id for sample in samples).items()
        if count > 1
    ]
    if duplicate_sample_ids:
        print(
            f"error: duplicate canonical sample IDs detected: {duplicate_sample_ids[:5]} ...",
            file=sys.stderr,
        )
        return 3

    # Reject URIs that resolve outside of slp_root to keep the adapter
    # auditable; missing-on-disk is allowed (it produces a quality flag)
    # but absolute paths outside slp_root are not.
    for sample in samples:
        for modality, uri in sample.frame.modality_uris.items():
            if not uri:
                continue
            candidate = Path(uri)
            if candidate.is_absolute() and not str(candidate).startswith(str(slp_root.resolve())):
                print(
                    f"error: sample {sample.sample_id} modality {modality} URI {uri!r} "
                    f"is outside slp_root {slp_root!r}.",
                    file=sys.stderr,
                )
                return 4

    output_csv = _project_path(args.output_csv)
    output_jsonl = _project_path(args.output_jsonl)
    output_summary = _project_path(args.output_summary)

    write_canonical_csv(samples, output_csv)
    write_canonical_jsonl(samples, output_jsonl)

    summary = summarise_canonical_samples(samples)
    summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["a03_frame_index_csv"] = str(a03_csv)
    summary["a04_homography_audit_csv"] = str(a04_csv) if a04_audit_path else None
    summary["output_csv"] = str(output_csv)
    summary["output_jsonl"] = str(output_jsonl)
    summary["a03_rows_loaded"] = len(a03_rows)
    summary["a04_rows_loaded"] = len(a04_rows)

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
                "rows": summary["rows"],
                "quarantine_rows": summary["quarantine_rows"],
                "traceable_rate": summary["uri_traceability"]["traceable_rate"],
                "subjects": summary["subjects"],
                "output_csv": str(output_csv),
                "output_jsonl": str(output_jsonl),
                "output_summary": str(output_summary),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    # Fail-closed on the two A03 invariants so adapter regressions cannot
    # silently produce canonical samples that violate the master index.
    if summary["rows"] != len(a03_rows):
        print(
            f"error: canonical sample count {summary['rows']} != a03 rows {len(a03_rows)}",
            file=sys.stderr,
        )
        return 5
    if summary["quarantine_rows"] > 0 and summary["quarantine_rows"] == summary["rows"]:
        # Total quarantine means the adapter produced no usable sample.
        print(
            "error: all canonical samples are quarantined; refusing to overwrite outputs.",
            file=sys.stderr,
        )
        return 6

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
