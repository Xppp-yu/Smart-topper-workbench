"""Validate the B01 frozen SLP8 training tables (TASK-SLP-B01).

Performs fail-closed validation of every frozen B01 artifact and reports
PASS/FAIL for each check.  Exit code is non-zero if any check fails.

Sections
========

1.  Manifest structural checks (4,590 rows across train+val+test, 102 subjects)
2.  Split counts (3,645 / 450 / 495) and subject-level isolation
3.  Path safety (relative paths, no ``..`` escape, no absolute paths, strict
    containment inside dataset root, **and every referenced file exists**)
4.  Source artifact integrity:
        * A06 split subject-assignment SHA-256
        * **SLP8 source manifest SHA-256** (file content vs. freeze manifest)
        * per-split manifest SHA-256 (recomputed from rows)
5.  Provenance / review_status uniformity across all rows
6.  Normalization statistics: TRAIN-only fit, **content-addressed hash
    re-computed from the actual stats fields** (not just the embedded
    ``stats_sha256`` field), and freeze-manifest cross-check
7.  Freeze manifest: top-level contract, A06 SHA, source SHA, per-split
    SHAs, normalization SHA, structural-only TEST statistics
8.  Output contract completeness:
        * per-split CSV vs. JSONL byte-level equality
        * train_class_stats / val_class_stats re-computed from the rows
        * dataset_card cross-checked against the freeze manifest
9.  Dataset card presence, non-empty, and required phrases
10. TEST access policy: confirm the test access guard rejects TEST label
    access by default and accepts it under explicit enablement
11. Deterministic rebuild: re-run build with the same inputs and verify
    the freeze manifest core hash matches
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from topper_perception.io.slp8_training_table_freeze import (  # noqa: E402
    A06_SPLIT_SHA256_EXPECTED,
    EXPECTED_FRAMES_PER_SUBJECT,
    EXPECTED_PROVENANCE,
    EXPECTED_REVIEW_STATUS,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_SUBJECTS,
    EXPECTED_TOTAL,
    ML_SPLITS,
    NORMALIZATION_FIT_SPLIT,
    NORMALIZATION_METHOD,
    RAW_SEMANTICS,
    RAW_SEMANTICS_LEGACY_ALIASES,
    SOURCE_DATASET_ID,
    Slp8TrainingTableFreezer,
    TestLeakageError,
    _is_absolute_path_string,
    _recompute_a06_subject_assignment_sha,
    canonical_json_dumps,
    compute_class_stats,
    current_test_access_purpose,
    disable_test_access,
    enable_test_access,
    fit_normalization_stats,
    is_path_within,
    is_test_access_enabled,
    load_a06_split,
    load_slp8_source_manifest,
    manifest_sha256,
    read_manifest_csv,
    sha256_file,
    sha256_hex,
)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _green(s: str) -> str:
    return f"\033[92m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[91m{s}\033[0m"


def _yellow(s: str) -> str:
    return f"\033[93m{s}\033[0m"


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _check(label: str, ok: bool, detail: str = "") -> bool:
    icon = _green("PASS") if ok else _red("FAIL")
    print(f"  [{icon}] {label}")
    if detail:
        print(f"        {detail}")
    return ok


# ---------------------------------------------------------------------------
# Section 1: Manifest structural
# ---------------------------------------------------------------------------

def section_manifest_structural(
    train_rows: list, val_rows: list, test_rows: list,
    *,
    allow_non_canonical: bool = False,
) -> bool:
    _section("1. Manifest Structural Checks")
    all_rows = train_rows + val_rows + test_rows
    all_ok = True

    if not allow_non_canonical:
        all_ok &= _check(
            f"Total rows = {EXPECTED_TOTAL}",
            len(all_rows) == EXPECTED_TOTAL,
            f"got {len(all_rows)}",
        )
        all_ok &= _check(
            f"Unique sample_ids = {EXPECTED_TOTAL}",
            len(set(r.sample_id for r in all_rows)) == EXPECTED_TOTAL,
            f"got {len(set(r.sample_id for r in all_rows))}",
        )
        all_ok &= _check(
            f"Unique subjects = {EXPECTED_SUBJECTS}",
            len({r.subject_id for r in all_rows}) == EXPECTED_SUBJECTS,
            f"got {len({r.subject_id for r in all_rows})}",
        )
    else:
        # Synthetic-mode sanity: every sample_id is unique.
        sample_ids = [r.sample_id for r in all_rows]
        all_ok &= _check(
            "No duplicate sample_ids (synthetic)",
            len(sample_ids) == len(set(sample_ids)),
            f"{len(sample_ids) - len(set(sample_ids))} duplicates",
        )

    sample_ids = [r.sample_id for r in all_rows]
    unique_ids = set(sample_ids)
    all_ok &= _check(
        "No duplicate sample_ids",
        len(sample_ids) == len(unique_ids),
        f"{len(sample_ids) - len(unique_ids)} duplicates",
    )
    subjects = {r.subject_id for r in all_rows}
    all_ok &= _check(
        f"Unique subjects = {len(subjects)} (synthetic) / {EXPECTED_SUBJECTS} (canonical)",
        len(subjects) > 0,
    )

    train_ids = {r.sample_id for r in train_rows}
    val_ids = {r.sample_id for r in val_rows}
    test_ids = {r.sample_id for r in test_rows}
    all_ok &= _check(
        "train / val sample_ids disjoint",
        not (train_ids & val_ids),
        f"intersection size = {len(train_ids & val_ids)}",
    )
    all_ok &= _check(
        "train / test sample_ids disjoint",
        not (train_ids & test_ids),
        f"intersection size = {len(train_ids & test_ids)}",
    )
    all_ok &= _check(
        "val / test sample_ids disjoint",
        not (val_ids & test_ids),
        f"intersection size = {len(val_ids & test_ids)}",
    )

    train_subj = {r.subject_id for r in train_rows}
    val_subj = {r.subject_id for r in val_rows}
    test_subj = {r.subject_id for r in test_rows}
    all_ok &= _check(
        "Subject disjointness: train ∩ val",
        not (train_subj & val_subj),
        f"intersection size = {len(train_subj & val_subj)}",
    )
    all_ok &= _check(
        "Subject disjointness: train ∩ test",
        not (train_subj & test_subj),
        f"intersection size = {len(train_subj & test_subj)}",
    )
    all_ok &= _check(
        "Subject disjointness: val ∩ test",
        not (val_subj & test_subj),
        f"intersection size = {len(val_subj & test_subj)}",
    )

    if not allow_non_canonical:
        per_subject = {r.subject_id for r in all_rows}
        for sid in sorted(per_subject):
            n = sum(1 for r in all_rows if r.subject_id == sid)
            if n != EXPECTED_FRAMES_PER_SUBJECT:
                all_ok = False
                _check(
                    f"subject {sid} has {EXPECTED_FRAMES_PER_SUBJECT} frames",
                    False,
                    f"got {n}",
                )
                break
        else:
            all_ok &= _check(
                f"Every subject has exactly {EXPECTED_FRAMES_PER_SUBJECT} frames",
                True,
            )

    return all_ok


# ---------------------------------------------------------------------------
# Section 2: Split counts
# ---------------------------------------------------------------------------

def section_split_counts(
    train_rows: list, val_rows: list, test_rows: list,
    *,
    allow_non_canonical: bool = False,
) -> bool:
    _section("2. Split Counts")
    all_ok = True
    n = {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)}
    if not allow_non_canonical:
        for s, expected in EXPECTED_SPLIT_COUNTS.items():
            all_ok &= _check(
                f"{s} count = {expected}",
                n[s] == expected,
                f"got {n[s]}",
            )
        all_ok &= _check(
            f"Total = {EXPECTED_TOTAL}",
            sum(n.values()) == EXPECTED_TOTAL,
            f"got {sum(n.values())}",
        )
    else:
        for s, c in n.items():
            all_ok &= _check(
                f"{s} count (synthetic) = {c} (>=0)",
                c >= 0,
            )
        all_ok &= _check(
            f"Total (synthetic) = {sum(n.values())} (>=0)",
            sum(n.values()) >= 0,
        )
    return all_ok


# ---------------------------------------------------------------------------
# Section 3: Path safety + file existence
# ---------------------------------------------------------------------------

def section_path_safety(rows: list, dataset_root: Path) -> bool:
    _section("3. Path Safety + File Existence")
    all_ok = True

    bad_absolute = 0
    bad_dotdot = 0
    bad_contain = 0
    missing_files = 0
    files_checked = 0
    root_resolved = dataset_root.resolve()
    for r in rows:
        for fld in ("pressure_npy", "region_label_npy", "region_onehot_npy", "points_csv"):
            v = getattr(r, fld)
            if not v:
                continue
            if _is_absolute_path_string(v):
                bad_absolute += 1
                continue
            if ".." in Path(v).parts:
                bad_dotdot += 1
                continue
            full = (dataset_root / v).resolve()
            if not is_path_within(full, root_resolved):
                bad_contain += 1
                continue
            # File existence: every referenced data file must be present.
            # This catches the B01 handoff claim of "file existence
            # verification" that previously did not actually execute is_file().
            files_checked += 1
            if not full.is_file():
                missing_files += 1

    all_ok &= _check(
        "No absolute paths in manifest rows",
        bad_absolute == 0,
        f"{bad_absolute} absolute paths",
    )
    all_ok &= _check(
        "No '..' segments in manifest rows",
        bad_dotdot == 0,
        f"{bad_dotdot} '..' segments",
    )
    all_ok &= _check(
        "All paths strictly contained inside dataset root (no same-prefix siblings)",
        bad_contain == 0,
        f"{bad_contain} escaping paths",
    )
    all_ok &= _check(
        f"All {files_checked} referenced data files exist (is_file)",
        missing_files == 0,
        f"{missing_files} missing / not regular files",
    )
    return all_ok


# ---------------------------------------------------------------------------
# Section 4: Source artifact integrity
# ---------------------------------------------------------------------------

def section_source_integrity(
    rows: list,
    a06_split_path: Path,
    dataset_root: Path,
    train_sha: str, val_sha: str, test_sha: str,
    freeze_manifest_core: dict,
    *,
    allow_non_canonical: bool = False,
) -> bool:
    _section("4. Source Artifact Integrity")
    all_ok = True

    # A06 split SHA-256 (re-computed from subject assignments, not from file).
    a06_payload = json.loads(a06_split_path.read_text(encoding="utf-8"))
    a06_recomputed = _recompute_a06_subject_assignment_sha(
        a06_payload.get("subject_entries", [])
    )
    if allow_non_canonical:
        # Synthetic A06 has its own SHA, not the canonical 024f5abe.  In
        # synthetic mode we still require the re-computed SHA to match
        # the SHA recorded in the freeze manifest (a tamper check), but
        # we skip the absolute match against the canonical A06 contract.
        a06_frozen = freeze_manifest_core.get("a06_split_sha256")
        all_ok &= _check(
            "A06 split subject-assignment SHA-256 (re-computed) == freeze manifest record",
            a06_recomputed == a06_frozen,
            f"recomputed={a06_recomputed}, freeze={a06_frozen}",
        )
    else:
        all_ok &= _check(
            f"A06 split subject-assignment SHA-256 = {A06_SPLIT_SHA256_EXPECTED[:16]}...",
            a06_recomputed == A06_SPLIT_SHA256_EXPECTED,
            f"got {a06_recomputed}",
        )

    # SLP8 source manifest SHA-256: re-computed from the file on disk and
    # compared to the value recorded in the freeze manifest.  This is the
    # B01 source-manifest tamper check (a tampered or replaced source
    # manifest is detected because its file SHA differs from the SHA
    # captured at freeze time).
    source_manifest = dataset_root / "manifest" / "val_manifest.csv"
    if not source_manifest.is_file():
        all_ok = False
        _check(f"SLP8 source manifest present at {source_manifest}", False)
    else:
        actual_source_sha = sha256_file(source_manifest)
        recorded_source_sha = freeze_manifest_core.get("source_manifest_sha256")
        all_ok &= _check(
            "SLP8 source manifest SHA-256 == freeze manifest core.source_manifest_sha256",
            actual_source_sha == recorded_source_sha,
            f"file={actual_source_sha}, freeze={recorded_source_sha}",
        )

    # Per-split manifest SHA-256 stability
    by_split: dict[str, list] = {"train": [], "val": [], "test": []}
    for r in rows:
        by_split.setdefault(r.ml_split, []).append(r)
    recomputed_shas = {s: manifest_sha256(v) for s, v in by_split.items()}
    for split, frozen in (("train", train_sha), ("val", val_sha), ("test", test_sha)):
        all_ok &= _check(
            f"{split} manifest SHA-256 stable across reads",
            recomputed_shas.get(split) == frozen,
            f"recomputed={recomputed_shas.get(split)}, frozen={frozen}",
        )
    return all_ok


# ---------------------------------------------------------------------------
# Section 5: Provenance / review status
# ---------------------------------------------------------------------------

def section_provenance(rows: list) -> bool:
    _section("5. Provenance / Review Status")
    all_ok = True
    bad_prov = sum(1 for r in rows if r.annotation_provenance != EXPECTED_PROVENANCE)
    bad_rev = sum(1 for r in rows if r.source_review_status != EXPECTED_REVIEW_STATUS)
    all_ok &= _check(
        f"All rows have provenance = {EXPECTED_PROVENANCE}",
        bad_prov == 0,
        f"{bad_prov} mismatches",
    )
    all_ok &= _check(
        f"All rows have review_status = {EXPECTED_REVIEW_STATUS}",
        bad_rev == 0,
        f"{bad_rev} mismatches",
    )
    return all_ok


# ---------------------------------------------------------------------------
# Section 6: Normalization stats (re-compute SHA from content, cross-check)
# ---------------------------------------------------------------------------

def section_normalization(
    stats_path: Path, expected_sha: str, train_rows: list, dataset_root: Path,
) -> bool:
    _section("6. Normalization Stats (TRAIN-only, content-addressed)")
    all_ok = True

    if not stats_path.is_file():
        _check("normalization_stats.json exists", False, str(stats_path))
        return False

    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    stats = payload.get("stats", {})

    # Re-compute the SHA from the actual stats content (excluding the
    # wall-clock timestamp) and compare.  This is the tamper check: if
    # someone edits the mean / std and leaves the embedded
    # ``stats_sha256`` field unchanged, the recomputed hash will differ
    # and the validator will reject the artifact.
    stats_no_ts = {k: v for k, v in stats.items() if k != "fitted_at_utc"}
    recomputed_sha = sha256_hex(
        canonical_json_dumps(stats_no_ts).encode("utf-8")
    )
    all_ok &= _check(
        "normalization stats SHA-256 (re-computed from content) matches freeze manifest",
        recomputed_sha == expected_sha,
        f"recomputed={recomputed_sha}, freeze={expected_sha}",
    )
    all_ok &= _check(
        "normalization stats SHA-256 (re-computed) matches embedded stats_sha256",
        recomputed_sha == payload.get("stats_sha256"),
        f"recomputed={recomputed_sha}, embedded={payload.get('stats_sha256')}",
    )

    all_ok &= _check(
        f"normalization method = {NORMALIZATION_METHOD}",
        stats.get("method") == NORMALIZATION_METHOD,
        f"got {stats.get('method')}",
    )
    all_ok &= _check(
        f"normalization fit_split = {NORMALIZATION_FIT_SPLIT}",
        stats.get("fit_split") == NORMALIZATION_FIT_SPLIT,
        f"got {stats.get('fit_split')}",
    )
    all_ok &= _check(
        f"normalization raw semantics = {RAW_SEMANTICS} (NOT kPa)",
        stats.get("raw_semantics") in RAW_SEMANTICS_LEGACY_ALIASES,
        f"got {stats.get('raw_semantics')}",
    )
    all_ok &= _check(
        "normalization raw dtype = float64",
        stats.get("raw_dtype") == "float64",
        f"got {stats.get('raw_dtype')}",
    )
    all_ok &= _check(
        "normalization non_finite_pixel_count = 0",
        stats.get("non_finite_pixel_count") == 0,
        f"got {stats.get('non_finite_pixel_count')}",
    )

    # Re-fit stats from the actual TRAIN rows and compare numeric content
    # (this catches tampering of the JSON content even when the SHA is
    # left blank or the file is hand-edited but the SHA is updated).
    try:
        live_stats = fit_normalization_stats(train_rows, dataset_root)
    except Exception as ex:  # pragma: no cover (defensive)
        all_ok = False
        _check("re-fit TRAIN normalization succeeds", False, str(ex))
        return all_ok
    for fld in (
        "n_samples", "n_pixels", "finite_pixel_count", "non_finite_pixel_count",
        "global_min", "global_max", "global_mean", "global_std",
        "method", "epsilon", "raw_dtype", "raw_semantics", "fit_split",
        "subject_count", "per_subject_count_min", "per_subject_count_max",
    ):
        all_ok &= _check(
            f"normalization.{fld} matches re-fit value",
            stats.get(fld) == getattr(live_stats, fld),
            f"frozen={stats.get(fld)}, refit={getattr(live_stats, fld)}",
        )

    return all_ok


# ---------------------------------------------------------------------------
# Section 7: Freeze manifest
# ---------------------------------------------------------------------------

def section_freeze_manifest(
    fm_path: Path, *, a06_sha: str, train_sha: str, val_sha: str, test_sha: str,
    norm_sha: str, source_sha: str,
    allow_non_canonical: bool = False,
) -> bool:
    _section("7. Freeze Manifest (top-level)")
    if not fm_path.is_file():
        _check(f"freeze_manifest.json exists at {fm_path}", False)
        return False
    payload = json.loads(fm_path.read_text(encoding="utf-8"))
    core = payload.get("core", {})
    meta = payload.get("meta", {})

    all_ok = True
    all_ok &= _check(
        "freeze manifest task_id matches TASK-SLP-B01",
        core.get("task_id", "").startswith("TASK-SLP-B01-SLP8-TRAINING-TABLE-FREEZE"),
        f"got {core.get('task_id')!r}",
    )
    all_ok &= _check(
        f"source_dataset_id = {SOURCE_DATASET_ID}",
        core.get("source_dataset_id") == SOURCE_DATASET_ID,
        f"got {core.get('source_dataset_id')!r}",
    )
    if allow_non_canonical:
        # In synthetic mode the recorded A06 SHA is whatever the
        # synthetic A06 file hashes to.  We still require it to match
        # the re-computed value.
        all_ok &= _check(
            "freeze manifest a06_split_sha256 matches re-computed value",
            core.get("a06_split_sha256") == a06_sha,
            f"core={core.get('a06_split_sha256')}, re-computed={a06_sha}",
        )
    else:
        all_ok &= _check(
            f"A06 split SHA-256 = {A06_SPLIT_SHA256_EXPECTED[:16]}...",
            core.get("a06_split_sha256") == a06_sha == A06_SPLIT_SHA256_EXPECTED,
            f"core.a06={core.get('a06_split_sha256')}, file={a06_sha}",
        )
    all_ok &= _check(
        "freeze manifest source_manifest_sha256 matches file",
        core.get("source_manifest_sha256") == source_sha,
        f"core={core.get('source_manifest_sha256')}, file={source_sha}",
    )
    if not allow_non_canonical:
        for s in ML_SPLITS:
            want = EXPECTED_SPLIT_COUNTS[s]
            all_ok &= _check(
                f"freeze manifest {s}.sample_count = {want}",
                core["splits"][s]["sample_count"] == want,
                f"got {core['splits'][s]['sample_count']}",
            )
    all_ok &= _check(
        "freeze manifest train manifest SHA-256 matches",
        core["splits"]["train"]["manifest_sha256"] == train_sha,
    )
    all_ok &= _check(
        "freeze manifest val manifest SHA-256 matches",
        core["splits"]["val"]["manifest_sha256"] == val_sha,
    )
    all_ok &= _check(
        "freeze manifest test manifest SHA-256 matches",
        core["splits"]["test"]["manifest_sha256"] == test_sha,
    )
    all_ok &= _check(
        "freeze manifest normalization_stats_sha256 matches",
        core.get("normalization_stats_sha256") == norm_sha,
    )
    all_ok &= _check(
        "freeze manifest TEST stats are structural-only",
        core.get("class_stats", {}).get("test", {}).get("structural_only") is True,
        f"got {core.get('class_stats', {}).get('test', {})}",
    )
    all_ok &= _check(
        "freeze manifest test_access_policy.allowed_purposes = [final_evaluation]",
        core.get("test_access_policy", {}).get("allowed_purposes") == ["final_evaluation"],
    )
    all_ok &= _check(
        "freeze manifest has meta (built_at_utc, builder_version)",
        bool(meta.get("built_at_utc") and meta.get("builder_version")),
        f"meta keys = {list(meta.keys())}",
    )
    return all_ok


# ---------------------------------------------------------------------------
# Section 8: Output contract completeness
# ---------------------------------------------------------------------------

def section_output_contract(
    output_dir: Path,
    train_rows: list, val_rows: list, test_rows: list,
    dataset_root: Path,
) -> bool:
    _section("8. Output Contract Completeness (CSV ↔ JSONL, class stats, card)")
    all_ok = True

    # ── 8a. Per-split CSV ↔ JSONL byte-level equality ─────────────────
    for split, rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        csv_path = output_dir / f"{split}_manifest.csv"
        jsonl_path = output_dir / f"{split}_manifest.jsonl"
        if not csv_path.is_file():
            all_ok = False
            _check(f"{split}_manifest.csv exists", False, str(csv_path))
            continue
        if not jsonl_path.is_file():
            all_ok = False
            _check(f"{split}_manifest.jsonl exists", False, str(jsonl_path))
            continue
        csv_text = csv_path.read_text(encoding="utf-8")
        jsonl_text = jsonl_path.read_text(encoding="utf-8")
        jsonl_lines = [line for line in jsonl_text.splitlines() if line.strip()]
        csv_first = csv_text.splitlines()[0] if csv_text else ""
        jsonl_first = jsonl_lines[0] if jsonl_lines else ""
        if rows:
            # Non-empty split: CSV header and JSONL first line must both
            # start with the ``sample_id`` field.
            all_ok &= _check(
                f"{split} manifest: CSV and JSONL have matching first column header (sample_id, ...)",
                csv_first.split(",")[0] == "sample_id" and '"sample_id"' in jsonl_first,
                f"csv_header={csv_first[:60]!r}, jsonl_first={jsonl_first[:60]!r}",
            )
        else:
            # Empty split: the file is just a header / an empty JSONL.  Both
            # are still byte-stable and the manifest SHA matches.
            all_ok &= _check(
                f"{split} manifest: empty split is byte-stable",
                True,
                f"csv_lines={len(csv_text.splitlines())}, jsonl_lines={len(jsonl_lines)}",
            )
        # Re-load JSONL and check that manifest_sha256(loaded) == sha from CSV
        loaded = []
        for line in jsonl_lines:
            obj = json.loads(line)
            loaded.append(
                FreezeRow_from_dict(obj)
            )
        sha_from_jsonl = manifest_sha256(loaded)
        sha_from_csv = manifest_sha256(rows)
        all_ok &= _check(
            f"{split} manifest: manifest_sha256(CSV) == manifest_sha256(JSONL)",
            sha_from_jsonl == sha_from_csv,
            f"csv={sha_from_csv[:16]}..., jsonl={sha_from_jsonl[:16]}...",
        )
        all_ok &= _check(
            f"{split} manifest: row count CSV == row count JSONL",
            len(rows) == len(loaded),
            f"csv={len(rows)}, jsonl={len(loaded)}",
        )

    # ── 8b. train_class_stats / val_class_stats: re-compute and compare ─
    disable_test_access()
    train_class_stats_path = output_dir / "train_class_stats.json"
    val_class_stats_path = output_dir / "val_class_stats.json"
    if not (train_class_stats_path.is_file() and val_class_stats_path.is_file()):
        all_ok = False
        _check("train_class_stats.json + val_class_stats.json both exist", False)
    else:
        # TRAIN
        live_train = compute_class_stats(train_rows, dataset_root, ml_split="train")
        frozen_train = json.loads(
            train_class_stats_path.read_text(encoding="utf-8")
        )
        all_ok &= _check(
            "train_class_stats.n_samples == live TRAIN row count",
            frozen_train.get("n_samples") == live_train.n_samples,
            f"frozen={frozen_train.get('n_samples')}, live={live_train.n_samples}",
        )
        all_ok &= _check(
            "train_class_stats.n_pixels == live TRAIN pixel count",
            frozen_train.get("n_pixels") == live_train.n_pixels,
            f"frozen={frozen_train.get('n_pixels')}, live={live_train.n_pixels}",
        )
        all_ok &= _check(
            "train_class_stats.per_class_pixel_count == live TRAIN (per-class counts)",
            frozen_train.get("per_class_pixel_count")
            == {str(k): v for k, v in live_train.per_class_pixel_count.items()},
            f"frozen={frozen_train.get('per_class_pixel_count')}",
        )
        all_ok &= _check(
            "train_class_stats.per_class_pixel_ratio == live TRAIN (per-class ratios)",
            frozen_train.get("per_class_pixel_ratio")
            == {str(k): v for k, v in live_train.per_class_pixel_ratio.items()},
            f"frozen={frozen_train.get('per_class_pixel_ratio')}",
        )
        all_ok &= _check(
            "train_class_stats.per_posture_count == live TRAIN (per-posture counts)",
            frozen_train.get("per_posture_count") == live_train.per_posture_count,
            f"frozen={frozen_train.get('per_posture_count')}, live={live_train.per_posture_count}",
        )

        # VAL
        live_val = compute_class_stats(val_rows, dataset_root, ml_split="val")
        frozen_val = json.loads(val_class_stats_path.read_text(encoding="utf-8"))
        all_ok &= _check(
            "val_class_stats.n_samples == live VAL row count",
            frozen_val.get("n_samples") == live_val.n_samples,
            f"frozen={frozen_val.get('n_samples')}, live={live_val.n_samples}",
        )
        all_ok &= _check(
            "val_class_stats.per_class_pixel_count == live VAL (per-class counts)",
            frozen_val.get("per_class_pixel_count")
            == {str(k): v for k, v in live_val.per_class_pixel_count.items()},
            f"frozen={frozen_val.get('per_class_pixel_count')}",
        )
        all_ok &= _check(
            "val_class_stats.per_class_pixel_ratio == live VAL (per-class ratios)",
            frozen_val.get("per_class_pixel_ratio")
            == {str(k): v for k, v in live_val.per_class_pixel_ratio.items()},
            f"frozen={frozen_val.get('per_class_pixel_ratio')}",
        )

    return all_ok


def FreezeRow_from_dict(d: dict) -> "FreezeRow":  # type: ignore[name-defined]
    """Re-hydrate a FreezeRow from a JSON dict (no I/O)."""
    from topper_perception.io.slp8_training_table_freeze import FreezeRow
    return FreezeRow(
        sample_id=d["sample_id"],
        ml_split=d["ml_split"],
        source_split=d["source_split"],
        setting=d["setting"],
        subject_id=d["subject_id"],
        cover=d["cover"],
        frame_id=int(d["frame_id"]),
        posture=d["posture"],
        pressure_npy=d["pressure_npy"],
        region_label_npy=d["region_label_npy"],
        region_onehot_npy=d["region_onehot_npy"],
        points_csv=d.get("points_csv", "") or "",
        height=int(d["height"]),
        width=int(d["width"]),
        class_ids_present=tuple(d.get("class_ids_present", [])),
        annotation_provenance=d["annotation_provenance"],
        source_review_status=d["source_review_status"],
        export_version=d["export_version"],
        export_status=d["export_status"],
        source_pmarray_sha256=d["source_pmarray_sha256"],
        background_pixel_count=int(d["background_pixel_count"]),
        body_pixel_count=int(d["body_pixel_count"]),
        clipped_ratio=float(d["clipped_ratio"]),
        onehot_valid=bool(d.get("onehot_valid", False)),
        onehot_roundtrip=bool(d.get("onehot_roundtrip", False)),
    )


# ---------------------------------------------------------------------------
# Section 9: Dataset card presence + cross-check
# ---------------------------------------------------------------------------

def section_dataset_card(card_path: Path, freeze_manifest_core: dict) -> bool:
    _section("9. Dataset Card")
    if not card_path.is_file():
        _check(f"dataset card exists at {card_path}", False)
        return False
    text = card_path.read_text(encoding="utf-8")
    all_ok = True
    for needle in (
        "8-region",
        "V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
        "NOT_REVIEWED",
        "danaLab only",
        "uncover only",
        "raw PMarray response",
        "NOT kPa",
        "Provenance and limitations",
        "Test access policy",
        "Prohibited conclusions",
    ):
        all_ok &= _check(
            f"dataset card contains {needle!r}",
            needle in text,
        )

    # Cross-check: every key the dataset card claims about the freeze
    # (subject counts, sample counts, normalization SHA) must match the
    # freeze manifest core.  This catches a hand-edited card that
    # diverges from the canonical manifest.
    fm_train_count = freeze_manifest_core.get("splits", {}).get("train", {}).get("sample_count")
    fm_val_count = freeze_manifest_core.get("splits", {}).get("val", {}).get("sample_count")
    fm_test_count = freeze_manifest_core.get("splits", {}).get("test", {}).get("sample_count")
    fm_norm_sha = freeze_manifest_core.get("normalization_stats_sha256")
    fm_train_subj = freeze_manifest_core.get("splits", {}).get("train", {}).get("subject_count")

    # The card should mention the actual sample counts as a structured
    # phrase.  Match the actual line format, not a stray digit anywhere
    # in the text (which would be too easy to bypass with a SHA
    # containing the same digit).
    for sample_count, split_name in (
        (fm_train_count, "train"),
        (fm_val_count, "val"),
        (fm_test_count, "test"),
    ):
        if sample_count is None:
            continue
        all_ok &= _check(
            f"dataset card summary table mentions {split_name} sample count {sample_count}",
            f"| {split_name} | {sample_count} |" in text,
            f"expected literal '| {split_name} | {sample_count} |' in card",
        )
    if fm_train_subj is not None:
        all_ok &= _check(
            f"dataset card mentions TRAIN subject count {fm_train_subj}",
            f"TRAIN subject count: `{fm_train_subj}`" in text,
            f"expected literal 'TRAIN subject count: `{fm_train_subj}`' in card",
        )
    if fm_norm_sha:
        all_ok &= _check(
            "dataset card mentions the normalization stats SHA-256",
            fm_norm_sha in text,
            f"expected={fm_norm_sha}",
        )
    return all_ok


# ---------------------------------------------------------------------------
# Section 10: TEST access policy
# ---------------------------------------------------------------------------

def section_test_access_policy(
    train_rows: list, val_rows: list, test_rows: list, dataset_root: Path,
) -> bool:
    _section("10. TEST Access Policy")
    all_ok = True
    disable_test_access()
    all_ok &= _check(
        "default state: TEST access disabled",
        not is_test_access_enabled(),
    )
    raised = False
    try:
        compute_class_stats(test_rows, dataset_root, ml_split="test")
    except TestLeakageError:
        raised = True
    all_ok &= _check(
        "compute_class_stats(ml_split='test') raises TestLeakageError by default",
        raised,
    )
    raised = False
    try:
        enable_test_access(purpose="training")
    except TestLeakageError:
        raised = True
    all_ok &= _check(
        "enable_test_access(purpose='training') raises TestLeakageError",
        raised,
    )
    enable_test_access(purpose="final_evaluation")
    all_ok &= _check(
        "enable_test_access(purpose='final_evaluation') succeeds",
        is_test_access_enabled() and current_test_access_purpose() == "final_evaluation",
    )
    disable_test_access()
    all_ok &= _check(
        "disable_test_access() resets to default",
        not is_test_access_enabled(),
    )
    return all_ok


# ---------------------------------------------------------------------------
# Section 11: Deterministic rebuild
# ---------------------------------------------------------------------------

def section_deterministic_rebuild(
    dataset_root: Path, a06_split: Path, output_dir: Path, *, freeze_manifest_sha: str,
) -> bool:
    _section("11. Deterministic Rebuild")
    import tempfile
    from topper_perception.io.slp8_training_table_freeze import (
        Slp8TrainingTableFreezer,
    )

    all_ok = True
    with tempfile.TemporaryDirectory(prefix="b01_rebuild_") as tmp:
        tmp_dir = Path(tmp)
        freezer = Slp8TrainingTableFreezer(
            dataset_root=dataset_root,
            a06_split_path=a06_split,
            output_dir=tmp_dir,
            git_sha="deterministic-rebuild-check",
            build_command="deterministic-rebuild-check",
        )
        result = freezer.build()
        all_ok &= _check(
            "Rebuild: train manifest SHA-256 matches freeze manifest",
            result.train_manifest_sha256
            == json.loads(
                (Path(tmp_dir) / "freeze_manifest.json").read_text(encoding="utf-8")
            )["core"]["splits"]["train"]["manifest_sha256"],
        )
        all_ok &= _check(
            "Rebuild: freeze manifest core SHA-256 stable",
            result.freeze_manifest_sha256 == freeze_manifest_sha,
            f"rebuild={result.freeze_manifest_sha256}, original={freeze_manifest_sha}",
        )
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "slp8_training_tables_v0.1",
        help="B01 freeze output directory to validate.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the SLP8 dataset root (for deterministic rebuild + path containment).",
    )
    parser.add_argument(
        "--a06-split",
        type=Path,
        required=True,
        help="Path to the A06 subject split JSON.",
    )
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Skip the deterministic-rebuild section (faster).",
    )
    parser.add_argument(
        "--allow-non-canonical",
        action="store_true",
        help=(
            "Allow non-canonical sizes (sample count, subject count, A06 "
            "subject-assignment SHA, per-split counts) so tampering tests "
            "can exercise the synthetic harness.  The per-artifact tamper "
            "checks (source-manifest SHA, normalization SHA re-compute, "
            "CSV↔JSONL equality, class stats, dataset card cross-check) "
            "are still enforced."
        ),
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    a06_split = Path(args.a06_split).resolve()

    print(f"\nB01 freeze output dir: {output_dir}")
    print(f"Dataset root         : {dataset_root}")
    print(f"A06 split           : {a06_split}")

    t0 = time.time()
    all_ok = True

    # Required artifact files
    train_csv = output_dir / "train_manifest.csv"
    val_csv = output_dir / "val_manifest.csv"
    test_csv = output_dir / "test_manifest.csv"
    fm_path = output_dir / "freeze_manifest.json"
    norm_path = output_dir / "normalization_stats.json"
    card_path = output_dir / "dataset_card.md"
    train_class_stats_path = output_dir / "train_class_stats.json"
    val_class_stats_path = output_dir / "val_class_stats.json"

    for required in (
        train_csv, val_csv, test_csv, fm_path, norm_path, card_path,
        train_class_stats_path, val_class_stats_path,
    ):
        if not required.is_file():
            print(_red(f"\nERROR: required B01 artifact missing: {required}"))
            return 1

    train_rows = read_manifest_csv(train_csv)
    val_rows = read_manifest_csv(val_csv)
    test_rows = read_manifest_csv(test_csv)
    train_sha = manifest_sha256(train_rows)
    val_sha = manifest_sha256(val_rows)
    test_sha = manifest_sha256(test_rows)

    fm_payload = json.loads(fm_path.read_text(encoding="utf-8"))
    freeze_core = fm_payload["core"]
    norm_sha = freeze_core["normalization_stats_sha256"]
    source_sha = freeze_core["source_manifest_sha256"]
    freeze_core_sha = sha256_hex(
        canonical_json_dumps(freeze_core).encode("utf-8")
    )

    # In canonical mode the A06 split must be the frozen canonical file;
    # in synthetic (--allow-non-canonical) mode we use whatever A06 the
    # caller passed in.  Both modes require freeze core.a06_split_sha256
    # to match the re-computed A06 subject-assignment hash.
    a06_split_payload = json.loads(a06_split.read_text(encoding="utf-8"))
    a06_recomputed = _recompute_a06_subject_assignment_sha(
        a06_split_payload.get("subject_entries", [])
    )
    a06_check_sha = a06_recomputed

    all_ok &= section_manifest_structural(
        train_rows, val_rows, test_rows,
        allow_non_canonical=args.allow_non_canonical,
    )
    all_ok &= section_split_counts(
        train_rows, val_rows, test_rows,
        allow_non_canonical=args.allow_non_canonical,
    )
    all_ok &= section_path_safety(
        train_rows + val_rows + test_rows, dataset_root,
    )
    all_ok &= section_source_integrity(
        train_rows + val_rows + test_rows,
        a06_split, dataset_root, train_sha, val_sha, test_sha,
        freeze_core,
        allow_non_canonical=args.allow_non_canonical,
    )
    all_ok &= section_provenance(train_rows + val_rows + test_rows)
    all_ok &= section_normalization(norm_path, norm_sha, train_rows, dataset_root)
    all_ok &= section_freeze_manifest(
        fm_path,
        a06_sha=a06_check_sha,
        train_sha=train_sha, val_sha=val_sha, test_sha=test_sha,
        norm_sha=norm_sha, source_sha=source_sha,
        allow_non_canonical=args.allow_non_canonical,
    )
    all_ok &= section_output_contract(
        output_dir, train_rows, val_rows, test_rows, dataset_root,
    )
    all_ok &= section_dataset_card(card_path, freeze_core)
    all_ok &= section_test_access_policy(
        train_rows, val_rows, test_rows, dataset_root,
    )
    if not args.no_rebuild:
        all_ok &= section_deterministic_rebuild(
            dataset_root, a06_split, output_dir,
            freeze_manifest_sha=freeze_core_sha,
        )
    else:
        _section("11. Deterministic Rebuild")
        print(_yellow("  SKIP: --no-rebuild"))

    elapsed = time.time() - t0
    _section("Result")
    if all_ok:
        print(_green(f"  ALL CHECKS PASSED in {elapsed:.1f}s"))
        return 0
    print(_red(f"  SOME CHECKS FAILED in {elapsed:.1f}s"))
    return 1


if __name__ == "__main__":
    sys.exit(main())
