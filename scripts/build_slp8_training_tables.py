"""Build the SLP 8-region pressure-only training/validation/test tables.

This is the CLI entry point for TASK-SLP-B01.  It reads:

* the SLP8 source dataset (``SLP_8Region_Pressure_VAL_v1.1``), and
* the A06 subject-level split manifest,

then writes the frozen B01 training tables (CSV + JSONL) plus a top-level
freeze manifest, TRAIN-only normalization statistics, and a dataset card.

All input paths are taken from CLI flags; no absolute Windows path is
written into any artifact.  Local machine paths used to invoke the script
may be recorded in the build metadata (``build_command``) but are not
embedded in the deterministic manifest core.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Make ``topper_perception`` importable when run directly via Python.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from topper_perception.io.slp8_training_table_freeze import (  # noqa: E402
    ADAPTER_VERSION,
    FREEZE_VERSION,
    TASK_ID,
    Slp8TrainingTableFreezer,
)


def _resolve_git_sha() -> str | None:
    """Best-effort current HEAD SHA-256 hex (40 chars).  None if unavailable."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return None
    return sha or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the SLP8 pressure-only training tables (TASK-SLP-B01).  "
            "Reads the SLP8 source dataset and the A06 subject split, and "
            "writes the frozen TRAIN/VAL/TEST manifests plus normalization "
            "statistics, freeze manifest, and dataset card."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help=(
            "Path to the SLP_8Region_Pressure_VAL_v1.1 dataset root.  "
            "Pass as a CLI flag or env var; never hard-code this in source."
        ),
    )
    parser.add_argument(
        "--a06-split",
        type=Path,
        required=True,
        help=(
            "Path to the A06 subject split JSON "
            "(slp_subject_split_v0.1.json).  Pass as a CLI flag or env var."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "slp8_training_tables_v0.1",
        help="Output directory for the B01 freeze artifacts.",
    )
    parser.add_argument(
        "--dataset-card",
        type=Path,
        default=None,
        help=(
            "Optional path for the dataset card markdown.  "
            "Defaults to <output-dir>/dataset_card.md."
        ),
    )
    parser.add_argument(
        "--git-sha",
        default=None,
        help=(
            "Optional Git HEAD SHA to record in the freeze manifest.  "
            "If omitted, the script tries to read it from the local repo."
        ),
    )
    parser.add_argument(
        "--no-auto-git-sha",
        action="store_true",
        help="Disable auto-detection of Git HEAD SHA.",
    )
    parser.add_argument(
        "--task-id",
        default=TASK_ID,
        help="TASK-ID to record in the freeze manifest (default: %(default)s).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset_root = Path(args.dataset_root).resolve()
    a06_split = Path(args.a06_split).resolve()
    output_dir = Path(args.output_dir).resolve()
    dataset_card = Path(args.dataset_card).resolve() if args.dataset_card else None

    if not dataset_root.is_dir():
        print(f"error: dataset root not found: {dataset_root}", file=sys.stderr)
        return 2
    if not a06_split.is_file():
        print(f"error: A06 split JSON not found: {a06_split}", file=sys.stderr)
        return 2

    git_sha = args.git_sha
    if not git_sha and not args.no_auto_git_sha:
        git_sha = _resolve_git_sha()

    build_command = (
        f"uv run python scripts/build_slp8_training_tables.py "
        f"--dataset-root {dataset_root} --a06-split {a06_split} "
        f"--output-dir {output_dir}"
        + (f" --git-sha {git_sha}" if git_sha else "")
    )

    print(f"[B01] task-id       : {args.task_id}")
    print(f"[B01] adapter       : {ADAPTER_VERSION}")
    print(f"[B01] freeze version: {FREEZE_VERSION}")
    print(f"[B01] dataset root  : {dataset_root}")
    print(f"[B01] a06 split     : {a06_split}")
    print(f"[B01] output dir    : {output_dir}")
    print(f"[B01] git sha       : {git_sha or '(unknown)'}")

    freezer = Slp8TrainingTableFreezer(
        dataset_root=dataset_root,
        a06_split_path=a06_split,
        output_dir=output_dir,
        dataset_card_path=dataset_card,
        git_sha=git_sha,
        build_command=build_command,
    )

    result = freezer.build()

    print("")
    print(f"[B01] WROTE {result.n_train} train / {result.n_val} val / {result.n_test} test rows")
    print(f"[B01]   train manifest: {result.train_csv}")
    print(f"[B01]   val   manifest: {result.val_csv}")
    print(f"[B01]   test  manifest: {result.test_csv}")
    print(f"[B01]   freeze manifest: {result.freeze_manifest_path}")
    print(f"[B01]   normalization stats: {result.normalization_stats_path}")
    print(f"[B01]   dataset card: {result.dataset_card_path}")
    print("")
    print(f"[B01]   a06 split SHA-256      = {result.a06_split_sha256}")
    print(f"[B01]   source manifest SHA-256= {result.source_manifest_sha256}")
    print(f"[B01]   train manifest SHA-256 = {result.train_manifest_sha256}")
    print(f"[B01]   val   manifest SHA-256 = {result.val_manifest_sha256}")
    print(f"[B01]   test  manifest SHA-256 = {result.test_manifest_sha256}")
    print(f"[B01]   normalization SHA-256  = {result.normalization_stats_sha256}")
    print(f"[B01]   freeze   manifest SHA-256 = {result.freeze_manifest_sha256}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
