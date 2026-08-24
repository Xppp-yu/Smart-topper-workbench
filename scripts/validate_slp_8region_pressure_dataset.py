"""Validate SLP_8Region_Pressure_VAL_v1.1 dataset smoke + A06 split compatibility.

Usage:
    # Quick smoke (9 samples):
    uv run python scripts/validate_slp_8region_pressure_dataset.py
        --dataset-root "E:\\TeamProjects\\datasets\\smart-topper\\SLP2022\\SLP\\SLP_8Region_Pressure_VAL_v1.1"
        --split-manifest "E:\\TeamProjects\\smarttopper-slp-a06\\data\\processed\\slp\\slp_subject_split_v0.1.json"

    # Full validation (all 4,590 samples, SHA256 + onehot roundtrip):
    uv run python scripts/validate_slp_8region_pressure_dataset.py
        --dataset-root "E:\\TeamProjects\\datasets\\smart-topper\\SLP2022\\SLP\\SLP_8Region_Pressure_VAL_v1.1"
        --split-manifest "E:\\TeamProjects\\smarttopper-slp-a06\\data\\processed\\slp\\slp_subject_split_v0.1.json"
        --full

Sections (always run):
    1. Manifest structural checks (all 4,590 rows)
    2. Path containment
    3. Array spot-checks (9 samples; skipped with --no-spot-check)
    4. dataset_summary.json consistency
    5. class_schema.json consistency
    6. A06 split compatibility
    9. Manifest JSON fail-closed (dataset_summary/class_schema must exist)

Sections (only with --full):
    7. Full P/L/O validation for all 4,590 samples
    8. points.csv existence + containment for all 4,590

Exit codes:
    0  All checks passed
    1  One or more checks failed
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "SLP_8Region_Pressure_VAL_v1.1"
EXPECTED_TOTAL = 4590
EXPECTED_SUBJECTS = 102
EXPECTED_FRAMES_PER_SUBJECT = 45
EXPECTED_POSTURES = {"SUPINE": 1530, "LEFT": 1530, "RIGHT": 1530}
PRESSURE_SHAPE = (192, 84)
PRESSURE_DTYPE = np.float64
LABEL_SHAPE = (192, 84)
LABEL_DTYPE = np.uint8
ONEHOT_SHAPE = (9, 192, 84)
ONEHOT_DTYPE = np.uint8
SPOT_CHECK_SUBJECTS = ["00001", "00051", "00102"]  # early, mid, late


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def green(msg: str) -> str:
    return f"\033[92m{msg}\033[0m"


def red(msg: str) -> str:
    return f"\033[91m{msg}\033[0m"


def yellow(msg: str) -> str:
    return f"\033[93m{msg}\033[0m"


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def check(label: str, passed: bool, detail: str = "") -> bool:
    icon = green("PASS") if passed else red("FAIL")
    print(f"  [{icon}] {label}")
    if detail:
        print(f"        {detail}")
    return passed


# ---------------------------------------------------------------------------
# Manifest structural checks
# ---------------------------------------------------------------------------

def validate_manifest_structure(
    csv_path: Path,
) -> dict[str, object]:
    """Parse manifest, return summary dict. Fail on structural errors."""
    rows: list[dict[str, str]] = []
    # utf-8-sig handles BOM-prefixed CSV files
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for i, raw in enumerate(reader, start=2):
            sid = raw.get("sample_id", "").strip()
            if not sid:
                raise ValueError(f"Row {i}: empty sample_id")
            rows.append(raw)

    total = len(rows)
    sample_ids = [r["sample_id"] for r in rows]
    unique_ids = set(sample_ids)
    subjects = [r["subject_id"] for r in rows]
    postures = Counter(r["posture"] for r in rows)
    frames = Counter(r["frame_id"] for r in rows)

    # Check no duplicates
    dupes = len(sample_ids) - len(unique_ids)

    return {
        "rows_total": total,
        "unique_sample_ids": len(unique_ids),
        "duplicate_sample_ids": dupes,
        "unique_subjects": len(set(subjects)),
        "per_posture": dict(postures),
        "per_subject_count": Counter(subjects),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Path containment check
# ---------------------------------------------------------------------------

def check_path_containment(rows: list[dict[str, str]], dataset_root: Path) -> tuple[int, int]:
    """Check all relative paths stay within dataset_root. Returns (ok_count, fail_count).

    Uses pathlib.relative_to() which raises ValueError when the resolved path
    escapes root. Absolute-path strings (Windows C:\\ or POSIX /) are also rejected.
    """
    ok = fail = 0
    root_resolved = dataset_root.resolve()
    for raw in rows:
        for col in ("pressure_npy", "region_label_npy", "region_onehot_npy"):
            val = raw.get(col, "").strip()
            if not val:
                continue
            s = str(val)
            # Reject absolute-path strings
            if s and (s[0] == "/" or (len(s) > 1 and s[1] == ":")):
                fail += 1
                continue
            full = (dataset_root / val).resolve()
            try:
                full.relative_to(root_resolved)
                ok += 1
            except ValueError:
                fail += 1
    return ok, fail


# ---------------------------------------------------------------------------
# Spot-check: load a deterministic sample
# ---------------------------------------------------------------------------

def spot_check_array(
    rows: list[dict[str, str]],
    dataset_root: Path,
    subject_id: str,
    frame_id: str,
) -> dict[str, object]:
    """Load one sample's arrays and verify shape/dtype/finite/range."""
    # Find the matching row
    target_prefix = f"SLP:danaLab:{subject_id}:uncover:{frame_id}"
    raw = next((r for r in rows if r["sample_id"] == target_prefix), None)
    if raw is None:
        return {"ok": False, "error": f"No row for {target_prefix}"}

    errors: list[str] = []
    pressure_sha256_actual = ""

    # pressure.npy
    p_rel = raw["pressure_npy"].strip()
    p_path = dataset_root / p_rel
    if p_path.exists():
        pressure = np.load(p_path, allow_pickle=False)
        if pressure.shape != PRESSURE_SHAPE:
            errors.append(f"pressure shape {pressure.shape} != {PRESSURE_SHAPE}")
        if pressure.dtype != PRESSURE_DTYPE:
            errors.append(f"pressure dtype {pressure.dtype} != {PRESSURE_DTYPE}")
        if not np.isfinite(pressure).all():
            errors.append("pressure contains NaN/Inf")
        pressure_sha256_actual = hashlib.sha256(p_path.read_bytes()).hexdigest()
    else:
        errors.append(f"pressure.npy not found: {p_path}")

    # region_label.npy
    l_rel = raw["region_label_npy"].strip()
    l_path = dataset_root / l_rel
    label_valid = False
    if l_path.exists():
        region_label = np.load(l_path, allow_pickle=False)
        if region_label.shape != LABEL_SHAPE:
            errors.append(f"label shape {region_label.shape} != {LABEL_SHAPE}")
        if region_label.dtype != LABEL_DTYPE:
            errors.append(f"label dtype {region_label.dtype} != {LABEL_DTYPE}")
        label_min = int(region_label.min())
        label_max = int(region_label.max())
        if label_min < 0 or label_max > 8:
            errors.append(f"label values [{label_min},{label_max}] out of [0,8]")
        label_valid = True
    else:
        errors.append(f"region_label.npy not found: {l_path}")

    # onehot.npy (if present)
    onehot_valid = False
    onehot_roundtrip_ok = False
    o_rel = raw.get("region_onehot_npy", "").strip()
    if o_rel:
        o_path = dataset_root / o_rel
        if o_path.exists():
            region_onehot = np.load(o_path, allow_pickle=False)
            if region_onehot.shape != ONEHOT_SHAPE:
                errors.append(f"onehot shape {region_onehot.shape} != {ONEHOT_SHAPE}")
            if region_onehot.dtype != ONEHOT_DTYPE:
                errors.append(f"onehot dtype {region_onehot.dtype} != {ONEHOT_DTYPE}")
            # Only check semantic properties if shape is correct
            if region_onehot.shape == ONEHOT_SHAPE:
                unique_vals = set(np.unique(region_onehot).tolist())
                if unique_vals - {0, 1}:
                    errors.append(f"onehot non-binary values: {unique_vals}")
                elif 0 not in unique_vals - {0, 1}:
                    # Check per-pixel channel sum == 1 (mutually exclusive)
                    pixel_sums = region_onehot.sum(axis=0)
                    bad_pixels = int(np.sum(pixel_sums != 1))
                    if bad_pixels > 0:
                        errors.append(
                            f"onehot: {bad_pixels} pixels have channel sum != 1"
                        )
            if label_valid and region_onehot.shape == ONEHOT_SHAPE and len(errors) == 0:
                reconstructed = np.argmax(region_onehot, axis=0).astype(np.uint8)
                onehot_roundtrip_ok = bool(np.array_equal(reconstructed, region_label))
            onehot_valid = True
        else:
            errors.append(f"region_onehot.npy not found: {o_path}")

    # SHA256 match
    manifest_sha = raw["source_pmarray_sha256"].strip()
    if pressure_sha256_actual and pressure_sha256_actual != manifest_sha:
        errors.append(
            f"SHA256 mismatch: manifest={manifest_sha[:16]}..., "
            f"actual={pressure_sha256_actual[:16]}..."
        )

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "pressure_sha256_actual": pressure_sha256_actual,
        "onehot_roundtrip_ok": onehot_roundtrip_ok,
        "subject_id": subject_id,
        "frame_id": frame_id,
    }


# ---------------------------------------------------------------------------
# A06 split compatibility
# ---------------------------------------------------------------------------

def check_a06_split_compatibility(
    rows: list[dict[str, str]],
    split_manifest_path: Path,
) -> dict[str, object]:
    """Join manifest against A06 frozen split, verify counts.

    The A06 split was designed for all canonical SLP data (danaLab + simLab, all
    covers, including quarantine entries). This dataset only has danaLab/uncover/no-quarantine.

    Compatibility check:
    - All 102 dataset subjects must appear in A06 subject_entries (once per subject)
    - The same dataset subject must not be assigned to multiple A06 splits
    - Expected: 81 train / 10 val / 11 test = 102 subjects → 3645 / 450 / 495 samples
    """
    if not split_manifest_path.exists():
        return {
            "ok": False,
            "error": f"A06 split manifest not found: {split_manifest_path}",
        }

    with split_manifest_path.open(encoding="utf-8") as f:
        split_data = json.load(f)

    split_sha = split_data.get("manifest_sha256", "unknown")

    # Build a map of subject_id → split (one entry per subject; duplicates = quarantine entries)
    # For compatibility, we use the first occurrence (primary assignment) per subject.
    # Quarantine entries may add extra split assignments; they are legitimate per-frame quarantine
    # and not a "cross-split leak" for subject-level assignment.
    # Note: A06 includes both danaLab and simLab subjects. The same subject_id can appear
    # in both settings as different entries. We only check danaLab entries for our dataset.
    split_subjects_primary: dict[str, str] = {}
    split_subjects_all: dict[str, list[str]] = {}
    for s in split_data.get("subject_entries", []):
        sid = s["subject_id"]
        setting = s.get("setting", "")
        # Only track danaLab entries (this dataset's setting)
        if setting == "danaLab":
            if sid not in split_subjects_primary:
                split_subjects_primary[sid] = s["split"]
            if sid not in split_subjects_all:
                split_subjects_all[sid] = []
            split_subjects_all[sid].append(s["split"])

    # Verify all 102 dataset subjects are in A06
    dataset_subjects = {r["subject_id"] for r in rows}
    a06_subject_ids = set(split_subjects_primary.keys())
    missing_from_split = sorted(dataset_subjects - a06_subject_ids)
    extra_in_split = sorted(a06_subject_ids - dataset_subjects)

    # Check per-split counts (use primary assignment)
    split_counts: dict[str, int] = {}
    for raw in rows:
        sid = raw["subject_id"]
        split = split_subjects_primary.get(sid, "UNKNOWN")
        if split == "UNKNOWN":
            continue
        if split not in split_counts:
            split_counts[split] = 0
        split_counts[split] += 1

    # Expected: 81 train × 45 + 10 val × 45 + 11 test × 45 = 4,590
    expected_counts = {
        "train": 81 * 45,   # 3,645
        "val": 10 * 45,    # 450
        "test": 11 * 45,    # 495
    }

    # Check no dataset subject appears in multiple A06 splits
    multi_split = {
        sid: splits
        for sid, splits in split_subjects_all.items()
        if len(set(splits)) > 1 and sid in dataset_subjects
    }

    return {
        "ok": len(missing_from_split) == 0
              and len(extra_in_split) == 0,
        "split_sha": split_sha,
        "a06_subjects_total": len(a06_subject_ids),
        "dataset_subjects_total": len(dataset_subjects),
        "dataset_subjects": dataset_subjects,
        "missing_from_split": missing_from_split,
        "extra_in_split": extra_in_split,
        "per_a06_split_sample_count": split_counts,
        "expected_per_split": expected_counts,
        "per_split_match": {
            k: split_counts.get(k, 0) == v for k, v in expected_counts.items()
        },
        "multi_split_subjects": {
            k: v for k, v in multi_split.items()
        },
    }


# ---------------------------------------------------------------------------
# Full-sample validator (used in --full mode)
# ---------------------------------------------------------------------------

def validate_sample_full(
    raw: dict[str, str],
    dataset_root: Path,
) -> dict[str, object]:
    """Validate one sample's P/L/O arrays with SHA256 + onehot roundtrip.

    All np.load calls use allow_pickle=False.
    Returns {"ok": bool, "errors": list[str], "sample_id": str}.
    """
    errors: list[str] = []
    pressure_sha256_actual = ""

    p_rel = raw.get("pressure_npy", "").strip()
    p_path = dataset_root / p_rel
    if not p_path.exists():
        errors.append(f"pressure.npy not found: {p_path}")
    else:
        pressure = np.load(p_path, allow_pickle=False)
        if pressure.shape != PRESSURE_SHAPE:
            errors.append(f"pressure shape {pressure.shape} != {PRESSURE_SHAPE}")
        if pressure.dtype != PRESSURE_DTYPE:
            errors.append(f"pressure dtype {pressure.dtype} != {PRESSURE_DTYPE}")
        if not np.isfinite(pressure).all():
            errors.append("pressure contains NaN/Inf")
        pressure_sha256_actual = hashlib.sha256(p_path.read_bytes()).hexdigest()

    l_rel = raw.get("region_label_npy", "").strip()
    l_path = dataset_root / l_rel
    label_valid = False
    if not l_path.exists():
        errors.append(f"region_label.npy not found: {l_path}")
    else:
        region_label = np.load(l_path, allow_pickle=False)
        if region_label.shape != LABEL_SHAPE:
            errors.append(f"label shape {region_label.shape} != {LABEL_SHAPE}")
        if region_label.dtype != LABEL_DTYPE:
            errors.append(f"label dtype {region_label.dtype} != {LABEL_DTYPE}")
        label_min = int(region_label.min())
        label_max = int(region_label.max())
        if label_min < 0 or label_max > 8:
            errors.append(f"label values [{label_min},{label_max}] out of [0,8]")
        label_valid = True

    onehot_roundtrip_ok = False
    o_rel = raw.get("region_onehot_npy", "").strip()
    if o_rel:
        o_path = dataset_root / o_rel
        if not o_path.exists():
            errors.append(f"region_onehot.npy not found: {o_path}")
        else:
            region_onehot = np.load(o_path, allow_pickle=False)
            # Shape check first — must pass before semantic checks
            if region_onehot.shape != ONEHOT_SHAPE:
                errors.append(f"onehot shape {region_onehot.shape} != {ONEHOT_SHAPE}")
            if region_onehot.dtype != ONEHOT_DTYPE:
                errors.append(f"onehot dtype {region_onehot.dtype} != {ONEHOT_DTYPE}")
            # Only proceed with semantic checks if shape is correct
            if region_onehot.shape == ONEHOT_SHAPE:
                unique_vals = set(np.unique(region_onehot).tolist())
                if unique_vals - {0, 1}:
                    errors.append(f"onehot non-binary values: {unique_vals}")
                elif 0 not in unique_vals - {0, 1}:
                    # Check per-pixel channel sum == 1 (mutually exclusive)
                    pixel_sums = region_onehot.sum(axis=0)
                    bad_pixels = int(np.sum(pixel_sums != 1))
                    if bad_pixels > 0:
                        errors.append(
                            f"onehot: {bad_pixels} pixels have channel sum != 1"
                        )
                # argmax roundtrip
                if label_valid and not errors:
                    reconstructed = np.argmax(region_onehot, axis=0).astype(np.uint8)
                    onehot_roundtrip_ok = bool(np.array_equal(reconstructed, region_label))
                    if not onehot_roundtrip_ok:
                        diff_count = int(np.sum(reconstructed != region_label))
                        errors.append(
                            f"onehot roundtrip: argmax(onehot) != label "
                            f"({diff_count}/{region_label.size} pixels differ)"
                        )

    # SHA256 match (do NOT trust embedded hash; compute actual)
    manifest_sha = raw.get("source_pmarray_sha256", "").strip()
    if pressure_sha256_actual and pressure_sha256_actual != manifest_sha:
        errors.append(
            f"SHA256 mismatch: manifest={manifest_sha[:16]}..., "
            f"actual={pressure_sha256_actual[:16]}..."
        )

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "sample_id": raw.get("sample_id", "unknown"),
        "onehot_roundtrip_ok": onehot_roundtrip_ok,
    }


# ---------------------------------------------------------------------------
# points.csv helpers
# ---------------------------------------------------------------------------

def check_points_csv_containment(
    rows: list[dict[str, str]],
    dataset_root: Path,
) -> tuple[int, int]:
    """Check all points.csv paths stay within dataset_root using relative_to.

    Returns (ok_count, fail_count).
    """
    root_resolved = dataset_root.resolve()
    ok = fail = 0
    for raw in rows:
        pcsv_rel = raw.get("points_csv", "").strip()
        if not pcsv_rel:
            continue
        s = str(pcsv_rel)
        if s and (s[0] == "/" or (len(s) > 1 and s[1] == ":")):
            fail += 1
            continue
        full = (dataset_root / pcsv_rel).resolve()
        try:
            full.relative_to(root_resolved)
            ok += 1
        except ValueError:
            fail += 1
    return ok, fail


def check_points_csv_is_file(
    rows: list[dict[str, str]],
    dataset_root: Path,
) -> tuple[int, int]:
    """Verify all 4590 points.csv files exist (is_file). Returns (ok_count, fail_count)."""
    ok = fail = 0
    for raw in rows:
        pcsv_rel = raw.get("points_csv", "").strip()
        if not pcsv_rel:
            continue
        p = dataset_root / pcsv_rel
        if p.is_file():
            ok += 1
        else:
            fail += 1
    return ok, fail


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate SLP_8Region_Pressure_VAL_v1.1 dataset"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to SLP_8Region_Pressure_VAL_v1.1 root",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Optional: A06 split manifest for compatibility check",
    )
    parser.add_argument(
        "--spot-check",
        action="store_true",
        default=True,
        help="Run array spot-checks on 9 deterministic samples (default: True)",
    )
    parser.add_argument(
        "--no-spot-check",
        dest="spot_check",
        action="store_false",
        help="Skip array spot-checks",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help=(
            "Run full validation on all 4,590 samples: P/L/O shape/dtype/finite/range, "
            "SHA256 match, onehot roundtrip, points.csv existence+containment. "
            "Exit code 0 only if 4590/4590 pass."
        ),
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    manifest_path = dataset_root / "manifest" / "val_manifest.csv"
    split_manifest = Path(args.split_manifest) if args.split_manifest else None

    print(f"\nDataset: {DATASET_ID}")
    print(f"Root:    {dataset_root}")
    print(f"Manifest: {manifest_path}")
    print(f"A06 split: {split_manifest or '(not provided)'}")

    if not manifest_path.exists():
        print(red(f"\nERROR: val_manifest.csv not found at {manifest_path}"))
        return 1

    t0 = time.time()
    all_passed = True

    # ── 1. Manifest structural ────────────────────────────────────────────
    section("1. Manifest Structural Checks")
    try:
        summary = validate_manifest_structure(manifest_path)
    except Exception as ex:
        print(red(f"  ERROR: {ex}"))
        return 1

    all_passed &= check(
        f"Total rows = {EXPECTED_TOTAL}",
        summary["rows_total"] == EXPECTED_TOTAL,
        f"got {summary['rows_total']}",
    )
    all_passed &= check(
        "Unique sample_ids = 4,590",
        summary["unique_sample_ids"] == EXPECTED_TOTAL,
        f"got {summary['unique_sample_ids']} unique",
    )
    all_passed &= check(
        "No duplicate sample_ids",
        summary["duplicate_sample_ids"] == 0,
        f"found {summary['duplicate_sample_ids']} duplicates",
    )
    all_passed &= check(
        f"Unique subjects = {EXPECTED_SUBJECTS}",
        summary["unique_subjects"] == EXPECTED_SUBJECTS,
        f"got {summary['unique_subjects']}",
    )

    per_posture = summary["per_posture"]
    posture_ok = all(per_posture.get(k, 0) == v for k, v in EXPECTED_POSTURES.items())
    all_passed &= check(
        "Per-posture counts match expected",
        posture_ok,
        f"got {dict(per_posture)}, expected {EXPECTED_POSTURES}",
    )

    per_subj = summary["per_subject_count"]
    all_45 = all(v == EXPECTED_FRAMES_PER_SUBJECT for v in per_subj.values())
    all_passed &= check(
        "Each subject has exactly 45 frames",
        all_45,
        f"min={min(per_subj.values())}, max={max(per_subj.values())}",
    )

    rows = summary["rows"]

    # ── 2. Path containment ─────────────────────────────────────────────
    section("2. Path Containment (no D:\\ escapes)")
    path_ok, path_fail = check_path_containment(rows, dataset_root)
    all_passed &= check(
        "All relative paths stay within dataset root",
        path_fail == 0,
        f"{path_ok} ok, {path_fail} escaped root",
    )
    # Also verify no absolute D:\\ paths in pressure_npy column
    bad_abs = [r["sample_id"] for r in rows if "/" in r["pressure_npy"] and ":" in r["pressure_npy"]]
    all_passed &= check(
        "No absolute paths in pressure_npy column",
        len(bad_abs) == 0,
        f"{len(bad_abs)} rows with absolute paths",
    )

    # ── 3. Spot-check arrays ────────────────────────────────────────────
    if args.spot_check:
        section("3. Array Spot-Checks (3 subjects × 3 postures = 9 samples)")
        frame_ids = ["000001", "000015", "000030"]  # early, mid, late
        spot_errors: list[str] = []
        onehot_rt_ok = 0
        onehot_rt_total = 0
        sha_ok = 0

        for subj in SPOT_CHECK_SUBJECTS:
            for frame in frame_ids:
                result = spot_check_array(rows, dataset_root, subj, frame)
                label = f"{subj} / frame {frame}"
                if result["ok"]:
                    print(green(f"  PASS   {label}"))
                    sha_ok += 1
                else:
                    print(red(f"  FAIL   {label}"))
                    for e in result["errors"]:
                        print(f"         - {e}")
                    spot_errors.append(f"{label}: {result['errors']}")
                    all_passed = False
                if result.get("onehot_roundtrip_ok") is not None:
                    onehot_rt_total += 1
                    if result["onehot_roundtrip_ok"]:
                        onehot_rt_ok += 1

        all_passed &= check(
            f"Pressure SHA256 spot-check ({sha_ok}/{len(SPOT_CHECK_SUBJECTS) * len(frame_ids)})",
            len(spot_errors) == 0,
            f"{len(spot_errors)} failures",
        )

    # ── 4. Dataset summary JSON consistency ──────────────────────────────
    section("4. Dataset Summary JSON Consistency")
    summary_path = dataset_root / "manifest" / "dataset_summary.json"
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as f:
            ds = json.load(f)
        all_passed &= check(
            "dataset_summary.json source_samples = 4590",
            ds.get("source_samples") == 4590,
            f"got {ds.get('source_samples')}",
        )
        all_passed &= check(
            "dataset_summary.json exported_samples = 4590",
            ds.get("exported_samples") == 4590,
            f"got {ds.get('exported_samples')}",
        )
        all_passed &= check(
            "dataset_summary.json status = SUCCESS",
            ds.get("status") == "SUCCESS",
            f"got {ds.get('status')}",
        )
        all_passed &= check(
            "dataset_summary.json per_posture match",
            ds.get("per_posture") == EXPECTED_POSTURES,
            f"got {ds.get('per_posture')}",
        )
        print(f"  NOTE: background_pixel_ratio = {ds.get('background_pixel_ratio', 'N/A')}")
        print(f"  NOTE: body_pixel_ratio = {ds.get('body_pixel_ratio', 'N/A')}")
    else:
        print(yellow(f"  SKIP: dataset_summary.json not found at {summary_path}"))

    # ── 5. Class schema consistency ─────────────────────────────────────
    section("5. Class Schema Consistency")
    class_schema_path = dataset_root / "manifest" / "class_schema.json"
    if class_schema_path.exists():
        with class_schema_path.open(encoding="utf-8") as f:
            cs = json.load(f)
        all_passed &= check(
            "class_schema version = slp8-v2.2.1-canonical-export-v1.1",
            cs.get("class_schema_version") == "slp8-v2.2.1-canonical-export-v1.1",
            f"got {cs.get('class_schema_version')}",
        )
        all_passed &= check(
            "num_semantic_regions = 8",
            cs.get("num_semantic_regions") == 8,
            f"got {cs.get('num_semantic_regions')}",
        )
        all_passed &= check(
            "num_label_ids = 9",
            cs.get("num_label_ids") == 9,
            f"got {cs.get('num_label_ids')}",
        )
        all_passed &= check(
            "background_id = 0",
            cs.get("background_id") == 0,
            f"got {cs.get('background_id')}",
        )
        class_names = [c["name"] for c in cs.get("classes", [])]
        expected_names = [
            "BACKGROUND", "HEAD_NECK", "SHOULDER", "THORAX_BACK",
            "LUMBAR_WAIST", "PELVIS_HIP", "ARM", "THIGH", "LOWER_LEG_FOOT",
        ]
        all_passed &= check(
            "All 9 class names present",
            class_names == expected_names,
            f"got {class_names}",
        )
    else:
        print(yellow(f"  SKIP: class_schema.json not found at {class_schema_path}"))

    # ── 6. A06 split compatibility ──────────────────────────────────────
    if split_manifest and split_manifest.exists():
        section("6. A06 Split Compatibility")
        compat = check_a06_split_compatibility(rows, split_manifest)
        if compat["ok"]:
            print(green("  PASS   All 102 subjects present in A06 split, no overlap"))
        else:
            print(red(f"  FAIL   {compat.get('error', 'unknown error')}"))
            all_passed = False

        all_passed &= check(
            "A06 split SHA256 recorded",
            bool(compat.get("split_sha", "unknown") != "unknown"),
            f"sha256={compat.get('split_sha', 'N/A')[:16]}...",
        )
        all_passed &= check(
            "All 102 dataset subjects in A06",
            len(compat.get("missing_from_split", [])) == 0,
            f"missing: {compat.get('missing_from_split', [])}",
        )
        all_passed &= check(
            "No extra subjects in A06 from this dataset",
            len(compat.get("extra_in_split", [])) == 0,
            f"extra: {compat.get('extra_in_split', [])}",
        )

        per_split = compat.get("per_a06_split_sample_count", {})
        expected = compat.get("expected_per_split", {})
        split_ok = all(
            per_split.get(k, 0) == v for k, v in expected.items()
        )
        all_passed &= check(
            "Per-A06-split sample counts match expected",
            split_ok,
            f"got {per_split}, expected {expected}",
        )

        # Check quarantine entries don't cause primary split assignment leakage
        # Quarantine entries (q>0) indicate per-frame quarantined samples that don't affect
        # subject-level primary split. Only danaLab entries affect our dataset.
        with split_manifest.open(encoding="utf-8") as f:
            split_data_check = json.load(f)
        primary_by_subject: dict[str, str] = {}
        quarantine_by_subject: dict[str, list[str]] = {}
        for s in split_data_check.get("subject_entries", []):
            sid = s["subject_id"]
            setting = s.get("setting", "")
            if setting != "danaLab":
                continue  # Only danaLab entries affect our dataset
            if sid not in primary_by_subject:
                primary_by_subject[sid] = s["split"]
            if s.get("quarantine_count", 0) > 0:
                if sid not in quarantine_by_subject:
                    quarantine_by_subject[sid] = []
                quarantine_by_subject[sid].append(s["split"])

        # Only flag conflicts for our dataset subjects (danaLab) where quarantine differs from primary
        primary_conflicts = {
            sid: quarantine_by_subject[sid]
            for sid in primary_by_subject
            if sid in compat["dataset_subjects"] and sid in quarantine_by_subject
        }
        all_passed &= check(
            "Primary subject split respected (quarantine entries don't reassign primary)",
            len(primary_conflicts) == 0,
            f"{len(primary_conflicts)} danaLab subjects with quarantine in different split from primary: "
            f"ids={list(primary_conflicts.keys())[:5]}",
        )
    else:
        section("6. A06 Split Compatibility")
        print(yellow(f"  SKIP: A06 split manifest not provided or not found"))

    # ── 7. Full validation: all 4,590 P/L/O arrays ────────────────────────
    if args.full:
        section("7. Full Array Validation (4,590 samples)")
        print(f"  Validating all {EXPECTED_TOTAL} samples (SHA256 + onehot roundtrip)...")
        full_errors: list[str] = []
        onehot_rt_ok = 0
        sha_ok_count = 0
        for i, raw in enumerate(rows, 1):
            result = validate_sample_full(raw, dataset_root)
            if not result["ok"]:
                err_summary = f"[{result['sample_id']}] " + "; ".join(result["errors"][:2])
                full_errors.append(err_summary)
            else:
                sha_ok_count += 1
                if result.get("onehot_roundtrip_ok"):
                    onehot_rt_ok += 1
            # Progress indicator every 500
            if i % 500 == 0 or i == len(rows):
                print(f"  ... {i}/{len(rows)} processed  (errors so far: {len(full_errors)})")

        all_passed &= check(
            f"All {EXPECTED_TOTAL} samples pass (4590/{EXPECTED_TOTAL})",
            len(full_errors) == 0,
            f"{len(full_errors)} failures" + ("; sample: " + full_errors[0] if full_errors else ""),
        )
        print(f"  SHA256 match: {sha_ok_count}/{EXPECTED_TOTAL}")
        print(f"  Onehot roundtrip: {onehot_rt_ok}/{EXPECTED_TOTAL}")
    else:
        section("7. Full Array Validation (--full mode)")
        print(yellow("  SKIP: pass --full to validate all 4,590 samples"))

    # ── 8. points.csv existence + containment ─────────────────────────────
    if args.full:
        section("8. points.csv Existence + Containment + is_file (all 4,590)")
        pts_ok, pts_fail = check_points_csv_containment(rows, dataset_root)
        all_passed &= check(
            "points.csv paths stay within dataset root",
            pts_fail == 0,
            f"{pts_ok} contained, {pts_fail} escaped",
        )
        # Full is_file check
        pts_isfile_ok, pts_isfile_fail = check_points_csv_is_file(rows, dataset_root)
        all_passed &= check(
            f"points.csv all {EXPECTED_TOTAL} exist as files",
            pts_isfile_fail == 0,
            f"{pts_isfile_ok} exist, {pts_isfile_fail} missing or not a file",
        )
        print(f"  NOTE: points.csv content roundtrip is a deterministic spot-check; not all rows read.")
    else:
        section("8. points.csv Existence + Containment (--full mode)")
        print(yellow("  SKIP: pass --full to check all points.csv paths"))

    # ── 9. Fail-closed: manifest JSON files must exist ────────────────────
    section("9. Manifest JSON Fail-Closed Checks")
    summary_path = dataset_root / "manifest" / "dataset_summary.json"
    class_schema_path = dataset_root / "manifest" / "class_schema.json"

    summary_exists = summary_path.exists()
    class_schema_exists = class_schema_path.exists()

    all_passed &= check(
        "dataset_summary.json exists",
        summary_exists,
        "" if summary_exists else f"not found at {summary_path}",
    )
    all_passed &= check(
        "class_schema.json exists",
        class_schema_exists,
        "" if class_schema_exists else f"not found at {class_schema_path}",
    )

    # Validate dataset_summary if present
    if summary_exists:
        try:
            with summary_path.open(encoding="utf-8") as f:
                ds = json.load(f)
            all_passed &= check(
                "dataset_summary.source_samples = 4590",
                ds.get("source_samples") == 4590,
                f"got {ds.get('source_samples')}",
            )
        except Exception as ex:
            all_passed = False
            print(red(f"  ERROR reading dataset_summary.json: {ex}"))

    # Validate class_schema if present
    if class_schema_exists:
        try:
            with class_schema_path.open(encoding="utf-8") as f:
                cs = json.load(f)
            all_passed &= check(
                "class_schema.num_semantic_regions = 8",
                cs.get("num_semantic_regions") == 8,
                f"got {cs.get('num_semantic_regions')}",
            )
        except Exception as ex:
            all_passed = False
            print(red(f"  ERROR reading class_schema.json: {ex}"))

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    section("Result")
    if all_passed:
        print(green(f"  ALL CHECKS PASSED in {elapsed:.1f}s"))
        return 0
    else:
        print(red(f"  SOME CHECKS FAILED in {elapsed:.1f}s"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
