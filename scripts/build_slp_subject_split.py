"""Build the SLP Subject-level Train/Val/Test Split Manifest.

This script reads the A05 canonical sample CSV, runs the deterministic
SlpSubjectSplitAdapter, and writes:

* ``data/processed/slp/slp_subject_split_v0.1.json`` — frozen split manifest
* ``outputs/reports/slp_subject_split_summary_v0.1.json`` — human-readable summary

Design rules (mirroring the A06 task contract):

* Deterministic: same seed always produces identical manifest SHA-256.
* Subject-level isolation: no subject appears in more than one split.
* simLab (7 subjects) held entirely in TEST as an out-of-domain held-out set.
* danaLab (102 subjects) split 80/10/10 train/val/test via deterministic
  hash-based assignment.
* Quarantined frames are reported separately and never silently included.
* No model scores influence the split.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from topper_perception.io.slp_subject_split import (
    ADAPTER_VERSION,
    DEFAULT_TASK_ID,
    SPLIT_SCHEMA_VERSION,
    SlpSubjectSplitAdapter,
    load_canonical_samples_from_csv,
    run_isolation_tests,
    verify_reproducibility,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--a05-csv",
        type=Path,
        default=PROJECT_ROOT / "data/processed/slp/slp_canonical_samples_v0.1.csv",
        help="Path to the A05 canonical sample CSV.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=PROJECT_ROOT / "data/processed/slp/slp_subject_split_v0.1.json",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=PROJECT_ROOT / "outputs/reports/slp_subject_split_summary_v0.1.json",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Fixed random seed for reproducibility.",
    )
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    a05_csv = _project_path(args.a05_csv)
    if not a05_csv.is_file():
        print(
            f"error: A05 canonical sample CSV not found at {a05_csv}. "
            "Run scripts/build_slp_canonical_samples.py first.",
            file=sys.stderr,
        )
        return 2

    output_manifest = _project_path(args.output_manifest)
    output_summary = _project_path(args.output_summary)

    print(f"Loading canonical samples from {a05_csv} ...")
    samples = load_canonical_samples_from_csv(a05_csv)
    print(f"  loaded {len(samples)} canonical sample rows")

    # Build manifest
    adapter = SlpSubjectSplitAdapter(
        samples,
        task_id=args.task_id,
        random_seed=args.random_seed,
    )
    manifest = adapter.build_manifest()

    # Write manifest
    manifest.to_json(output_manifest)
    print(f"Manifest written to {output_manifest}")
    print(f"  manifest_sha256 = {manifest.manifest_sha256}")

    # Reproducibility verification
    print("Verifying reproducibility (running split twice with same seed) ...")
    # Reload from CSV to get fresh iterator
    samples2 = load_canonical_samples_from_csv(a05_csv)
    repro = verify_reproducibility(
        samples2,
        seed=args.random_seed,
        task_id=args.task_id,
    )
    if not repro["reproducible"]:
        print(
            f"FATAL: reproducibility check failed.  "
            f"first_sha={repro['first_run_sha256']} "
            f"second_sha={repro['second_run_sha256']}",
            file=sys.stderr,
        )
        return 10
    print(f"  reproducibility PASSED (sha_match={repro['sha_match']}, "
          f"assignment_match={repro['assignment_match']})")

    # Isolation tests
    print("Running subject-level isolation tests ...")
    iso = run_isolation_tests(manifest)
    all_passed = True
    for check_name, result in iso.items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {check_name}: {result['details']}")
        if not result["passed"]:
            all_passed = False

    if not all_passed:
        print("FATAL: isolation tests failed", file=sys.stderr)
        return 11

    # Build summary
    summary = {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "task_id": args.task_id,
        "random_seed": args.random_seed,
        "manifest_sha256": manifest.manifest_sha256,
        "total_subjects": manifest.total_subjects,
        "danaLab_subjects": manifest.danaLab_subjects,
        "simLab_subjects": manifest.simLab_subjects,
        "total_frames": manifest.total_frames,
        "total_quarantined_frames": manifest.total_quarantined_frames,
        "total_usable_frames": manifest.total_usable_frames,
        "split_strategy_summary": manifest.split_strategy_summary,
        "split_rationale": manifest.split_rationale,
        "per_split": {},
    }

    for stat in manifest.split_statistics:
        split = stat.split
        summary["per_split"][split] = {
            "subject_count": stat.subject_count,
            "subject_count_danaLab": stat.subject_count_danaLab,
            "subject_count_simLab": stat.subject_count_simLab,
            "total_frames": stat.total_frames,
            "quarantined_frames": stat.quarantined_frames,
            "usable_frames": stat.usable_frames,
            "frame_count_danaLab": stat.frame_count_danaLab,
            "frame_count_simLab": stat.frame_count_simLab,
        }

    # Add subject lists
    summary["subject_lists"] = {
        "train": sorted(manifest.train_subjects()),
        "val": sorted(manifest.val_subjects()),
        "test": sorted(manifest.test_subjects()),
    }

    summary["isolation_tests"] = {k: {"passed": v["passed"]} for k, v in iso.items()}
    summary["reproducibility"] = {
        "sha_match": repro["sha_match"],
        "assignment_match": repro["assignment_match"],
        "reproducible": repro["reproducible"],
    }

    # DanaLab simLab distribution
    dana_subjects = {e.subject_id for e in manifest.subject_entries if e.setting == "danaLab"}
    sim_subjects = {e.subject_id for e in manifest.subject_entries if e.setting == "simLab"}
    summary["setting_distribution"] = {
        "danaLab": {
            "subjects": sorted(dana_subjects),
            "count": len(dana_subjects),
            "train": sorted(manifest.train_subjects() & dana_subjects),
            "val": sorted(manifest.val_subjects() & dana_subjects),
            "test": sorted(manifest.test_subjects() & dana_subjects),
        },
        "simLab": {
            "subjects": sorted(sim_subjects),
            "count": len(sim_subjects),
            "train": sorted(manifest.train_subjects() & sim_subjects),
            "val": sorted(manifest.val_subjects() & sim_subjects),
            "test": sorted(manifest.test_subjects() & sim_subjects),
        },
    }

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Summary written to {output_summary}")

    # Print headline summary
    print()
    print("=" * 60)
    print("SLP Subject Split Manifest — Summary")
    print("=" * 60)
    print(f"  Total subjects : {manifest.total_subjects} "
          f"(danaLab={manifest.danaLab_subjects}, simLab={manifest.simLab_subjects})")
    print(f"  Total frames   : {manifest.total_frames} "
          f"(quarantined={manifest.total_quarantined_frames}, "
          f"usable={manifest.total_usable_frames})")
    for stat in manifest.split_statistics:
        print(
            f"  [{stat.split.upper():>4}] "
            f"subjects={stat.subject_count:>3} "
            f"(danaLab={stat.subject_count_danaLab}, simLab={stat.subject_count_simLab}), "
            f"frames={stat.total_frames:>5} (usable={stat.usable_frames:>5})"
        )
    print(f"  Manifest SHA-256: {manifest.manifest_sha256}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
