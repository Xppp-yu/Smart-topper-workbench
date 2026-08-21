"""Run the independent PoPu P7 Full evidence re-verification analysis.

This script is the entry point the Reviewer invokes to reproduce the
``outputs/analysis/EXP-P7-FULL-ANALYSIS-20260821-R01/`` artifact directory.
It accepts either the frozen tar.gz archive or a pre-extracted directory
and runs :func:`topper_perception.neural.p7_full_analysis.analyze_p7_full`
to produce the eight required artifacts.

Per Reviewer Round 2: this CLI does NOT expose flags for the frozen rule
parameters (P6 single threshold, P6.1 temperature / threshold /
require_unanimous). Those values are loaded from the archive's pinned rule
block inside ``condition_comparison.json`` and verified against module-pinned
constants. The CLI's only legitimate role is to point at the archive; it
cannot silently override the frozen contract.

Usage::

    python -m scripts.analyze_popu_p7_full \
        --evidence-path C:\\path\\to\\EXP-P7-FULL-20260820-R02.tar.gz \
        --expected-archive-sha256 cbaffa74878b149e546a42826ae373442c62683af890362684f80963e7fddda1 \
        --output-dir outputs/analysis/EXP-P7-FULL-ANALYSIS-20260821-R01

The script never modifies the input archive and never reaches out to
AutoDL/GPU; it is CPU-only and deterministic from the frozen OOF CSVs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from topper_perception.neural.p7_full_analysis import (
    SCHEMA_VERSION,
    analyze_p7_full,
    verify_evidence_archive,
)

LOGGER = logging.getLogger("analyze_popu_p7_full")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independent re-verification of the PoPu P7 Full evidence pack. "
            "Produces the eight Reviewer-required artifacts under the output "
            "directory. Never modifies the input archive. Frozen rule values "
            "(P6 threshold, P6.1 temperature/threshold/unanimity) are loaded "
            "from the archive's pinned rule block, NOT from CLI flags."
        ),
    )
    parser.add_argument(
        "--evidence-path",
        type=Path,
        required=True,
        help="Path to the EXP-P7-FULL evidence pack (.tar.gz) or a pre-extracted directory.",
    )
    parser.add_argument(
        "--expected-archive-sha256",
        type=str,
        default=None,
        help="Frozen SHA-256 of the .tar.gz archive. Required when --evidence-path is a .tar.gz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the eight Reviewer-required artifacts into.",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory to extract the .tar.gz into. Defaults to a "
            "temporary directory. The original archive is never modified."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG / INFO / WARNING / ERROR).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    evidence_path = args.evidence_path.expanduser()
    if not evidence_path.exists():
        LOGGER.error("Evidence path does not exist: %s", evidence_path)
        return 2
    if evidence_path.is_file() and not args.expected_archive_sha256:
        LOGGER.error(
            "Refusing to verify a .tar.gz without --expected-archive-sha256 (Reviewer point: integrity)."
        )
        return 2

    LOGGER.info("Schema: %s", SCHEMA_VERSION)
    LOGGER.info("Evidence path: %s", evidence_path)
    LOGGER.info(
        "Frozen rule values are loaded from condition_comparison.json rule block; "
        "CLI does NOT expose override flags."
    )

    try:
        if evidence_path.is_file():
            verify_evidence_archive(
                evidence_path,
                expected_sha256=args.expected_archive_sha256,
                extract_dir=args.extract_dir,
            )
        manifest = analyze_p7_full(
            evidence_path,
            args.output_dir,
            expected_archive_sha256=args.expected_archive_sha256,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        LOGGER.exception("P7 Full analysis failed: %s", exc)
        return 1

    summary = {
        "schema_version": SCHEMA_VERSION,
        "manifest": manifest.as_dict(),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))