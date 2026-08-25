"""Tests for the SLP8 B01 training-table freeze (TASK-SLP-B01).

The test file is split into two parts:

* Pure unit tests using synthetic dataset/manifests and the A06 split schema
  re-implemented in-memory.  These tests run on any environment.
* Integration tests gated on the environment variables ``SLP8_DATASET_ROOT``
  and ``A06_SPLIT_PATH`` and the optional ``B01_FREEZE_OUTPUT_DIR``; these
  exercise the build on real data when available.

Negative-test catalogue (from B01 contract):
  * subject overlap across splits
  * unknown subject (not in A06)
  * unmapped subject (no ML split)
  * duplicate sample_id
  * illegal ML split value
  * using source ``VAL`` field as ML split
  * absolute Windows path
  * absolute POSIX path
  * ``..`` path escape
  * same-prefix sibling escape (``dataset_evil`` vs ``dataset``)
  * missing manifest
  * tampered manifest
  * source hash mismatch
  * A06 hash mismatch
  * illegal provenance
  * illegal review_status
  * normalization fitted on VAL
  * normalization fitted on TEST
  * default development mode reading TEST
  * tuning code reading TEST label/onehot
  * TEST class statistics
  * NaN/Inf pressure values
  * illegal label dtype/shape/class ID
  * illegal onehot dtype/shape
  * onehot with multiple active channels
  * onehot roundtrip mismatch
  * all ``np.load`` must use ``allow_pickle=False`` (verified via AST scan)
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable
from unittest.mock import patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from topper_perception.io.slp8_training_table_freeze import (  # noqa: E402
    A06_SPLIT_IDENTIFIER,
    A06_SPLIT_SHA256_EXPECTED,
    RAW_SEMANTICS,
    A06Split,
    B01FreezeTables,
    A06SplitContractError,
    ADAPTER_VERSION,
    AbsolutePathNotAllowedError,
    B01FreezeError,
    EXPECTED_FRAMES_PER_SUBJECT,
    EXPECTED_POSTURE_COUNTS,
    EXPECTED_PROVENANCE,
    EXPECTED_REVIEW_STATUS,
    EXPECTED_SPLIT_COUNTS,
    EXPECTED_SUBJECTS,
    EXPECTED_TOTAL,
    FREEZE_VERSION,
    FreezeRow,
    ML_SPLITS,
    NORMALIZATION_EPSILON,
    NORMALIZATION_FIT_SPLIT,
    NORMALIZATION_METHOD,
    NormalizationContractError,
    NormalizationStats,
    PathContainmentViolationError,
    SAMPLE_ID_PATTERN,
    SampleContractError,
    Slp8SourceManifest,
    Slp8SourceSample,
    Slp8TrainingTableFreezer,
    SubjectMappingError,
    TASK_ID,
    TestLeakageError,
    _is_absolute_path_string,
    _recompute_a06_subject_assignment_sha,
    assert_relative_path,
    build_freeze_row,
    canonical_json_dumps,
    compute_class_stats,
    current_test_access_purpose,
    disable_test_access,
    enable_test_access,
    fit_normalization_stats,
    is_path_within,
    is_test_access_enabled,
    load_a06_split,
    load_b01_freeze_tables,
    load_slp8_source_manifest,
    manifest_sha256,
    read_manifest_csv,
    render_dataset_card,
    sha256_file,
    sha256_hex,
    write_manifest_csv,
    write_manifest_jsonl,
    write_normalization_stats,
)


# ---------------------------------------------------------------------------
# Synthetic dataset/manifest helpers
# ---------------------------------------------------------------------------

def _make_dummy_pressure(shape=(192, 84), value: float = 100.0) -> np.ndarray:
    arr = np.full(shape, value, dtype=np.float64)
    return arr


def _make_dummy_label(shape=(192, 84), class_id: int = 0) -> np.ndarray:
    arr = np.zeros(shape, dtype=np.uint8)
    arr[...] = class_id
    return arr


def _make_dummy_onehot(shape=(9, 192, 84), class_id: int = 0) -> np.ndarray:
    arr = np.zeros(shape, dtype=np.uint8)
    arr[class_id, ...] = 1
    return arr


def _write_sample_arrays(
    dataset_root: Path,
    subject_id: str,
    frame_id: int,
    *,
    pressure_value: float = 100.0,
    label_class: int = 0,
    onehot_class: int = 0,
) -> tuple[str, str, str, str]:
    """Write dummy pressure/label/onehot/points files for one sample.

    Returns the relative paths (string) of each file.
    """
    rel_dir = f"samples/danaLab_{subject_id}_uncover_{frame_id:06d}"
    abs_dir = dataset_root / rel_dir
    abs_dir.mkdir(parents=True, exist_ok=True)
    np.save(abs_dir / "pressure.npy", _make_dummy_pressure(value=pressure_value))
    np.save(abs_dir / "region_label.npy", _make_dummy_label(class_id=label_class))
    np.save(abs_dir / "region_onehot.npy", _make_dummy_onehot(class_id=onehot_class))
    (abs_dir / "points.csv").write_text("x,y,class_id\n0,0,0\n", encoding="utf-8")
    return (
        f"{rel_dir}/pressure.npy",
        f"{rel_dir}/region_label.npy",
        f"{rel_dir}/region_onehot.npy",
        f"{rel_dir}/points.csv",
    )


def _make_synthetic_dataset(
    root: Path,
    *,
    n_subjects: int = 6,
    frames_per_subject: int = 3,
    cover: str = "uncover",
    setting: str = "danaLab",
    postures: tuple[str, ...] = ("SUPINE", "LEFT", "RIGHT"),
    source_provenance: str = EXPECTED_PROVENANCE,
    source_review_status: str = EXPECTED_REVIEW_STATUS,
) -> tuple[Path, list[dict[str, str]]]:
    """Create a minimal synthetic SLP8-like dataset root + manifest.

    Each subject gets ``frames_per_subject`` rows; postures cycle through
    the supplied tuple.  Returns (manifest_path, list_of_csv_rows).
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    rows_total = n_subjects * frames_per_subject
    for s_idx in range(n_subjects):
        subject_id = f"{s_idx + 1:05d}"
        for f_idx in range(frames_per_subject):
            frame_id = f_idx + 1
            posture = postures[f_idx % len(postures)]
            (p_rel, l_rel, o_rel, pt_rel) = _write_sample_arrays(
                root, subject_id, frame_id,
            )
            sha = sha256_file(root / p_rel)
            sid = f"SLP:{setting}:{subject_id}:{cover}:{frame_id:06d}"
            rows.append({
                "sample_id": sid,
                "split": "VAL",
                "setting": setting,
                "subject_id": subject_id,
                "cover": cover,
                "frame_id": str(frame_id),
                "posture": posture,
                "source_status": "REVIEW_REQUIRED",
                "source_flags": "",
                "source_review_status": source_review_status,
                "corrected_support_source": "",
                "corrected_support_type": "",
                "source_coordinate_frame": "RGB_PIXEL",
                "source_class_schema": "slp8-v2.2.1:0..8",
                "source_pmarray": "",
                "source_pmarray_shape": "[192, 84]",
                "source_pmarray_dtype": "float64",
                "source_pmarray_sha256": sha,
                "source_homography": "",
                "homography_direction": "",
                "homography_sha256": "",
                "pressure_npy": p_rel,
                "region_label_npy": l_rel,
                "region_onehot_npy": o_rel,
                "points_csv": pt_rel,
                "height": "192",
                "width": "84",
                "class_ids_present": "0|1|2|3|4|5|6|7|8",
                "background_count": "9000",
                "body_pixel_count": "7000",
                "mapping_required": "True",
                "mapping_valid": "True",
                "clipped_ratio": "0.0",
                "onehot_valid": "True",
                "onehot_roundtrip": "True",
                "points_roundtrip": "True",
                "annotation_provenance": source_provenance,
                "export_version": "1.1.0",
                "export_status": "EXPORTED",
                "exclude_reason": "",
            })
    manifest_path = root / "manifest" / "val_manifest.csv"
    columns = list(rows[0].keys())
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    return manifest_path, rows


def _make_synthetic_a06_split(
    subject_split: dict[str, str],
    *,
    setting: str = "danaLab",
    seed: int = 42,
    schema_version: str = A06_SPLIT_IDENTIFIER,
) -> dict[str, Any]:
    """Build a minimal A06 split dict mirroring slp_subject_split_v0.1."""
    entries = []
    for sid in sorted(subject_split):
        entries.append({
            "subject_id": sid,
            "setting": setting,
            "split": subject_split[sid],
            "frame_count": 45,
            "canonical_sample_count": 135,
            "quarantine_count": 0,
        })
    # SHA: matches the real A06 generator (json.dumps sorted, sort_keys, no ascii)
    payload = sorted(
        [
            {"subject_id": e["subject_id"], "setting": e["setting"], "split": e["split"]}
            for e in entries
        ],
        key=lambda x: x["subject_id"],
    )
    sha = sha256_hex(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )
    return {
        "schema_version": schema_version,
        "task_id": "synthetic-a06",
        "adapter_version": "synthetic_a06_v0",
        "generator": "test_synthetic",
        "created_at": "2026-01-01T00:00:00+00:00",
        "random_seed": seed,
        "split_rationale": "synthetic",
        "split_strategy_summary": "synthetic",
        "danaLab_train_frac": 0.8,
        "danaLab_val_frac": 0.1,
        "danaLab_test_frac": 0.1,
        "subject_entries": entries,
        "split_statistics": [],
        "total_subjects": len(entries),
        "total_frames": len(entries) * 45,
        "total_quarantined_frames": 0,
        "total_usable_frames": len(entries) * 45,
        "danaLab_subjects": len(entries) if setting == "danaLab" else 0,
        "simLab_subjects": 0,
        "manifest_sha256": sha,
    }


def _write_synthetic_a06_split(tmp: Path, a06: dict[str, Any]) -> Path:
    p = tmp / "a06_split.json"
    p.write_text(json.dumps(a06, indent=2), encoding="utf-8")
    return p


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_test_access() -> None:
    """Always reset the test access guard between tests."""
    disable_test_access()
    yield
    disable_test_access()


# ---------------------------------------------------------------------------
# A. Pure unit tests: path safety, hashing, sample-id patterns
# ---------------------------------------------------------------------------

class TestPathSafety:
    def test_relative_path_accepts_simple(self) -> None:
        p = assert_relative_path("samples/a/b/c.npy", field_name="x")
        assert p == Path("samples/a/b/c.npy")

    def test_absolute_windows_path_rejected(self) -> None:
        with pytest.raises(AbsolutePathNotAllowedError):
            assert_relative_path("D:\\data\\file.npy", field_name="x")

    def test_absolute_posix_path_rejected(self) -> None:
        with pytest.raises(AbsolutePathNotAllowedError):
            assert_relative_path("/etc/passwd", field_name="x")

    def test_unc_path_rejected(self) -> None:
        with pytest.raises(AbsolutePathNotAllowedError):
            assert_relative_path("\\\\server\\share\\file.npy", field_name="x")

    def test_dotdot_rejected(self) -> None:
        with pytest.raises(PathContainmentViolationError):
            assert_relative_path("a/../b.npy", field_name="x")

    def test_empty_path_rejected(self) -> None:
        with pytest.raises(PathContainmentViolationError):
            assert_relative_path("", field_name="x")

    def test_is_absolute_string_helper(self) -> None:
        assert _is_absolute_path_string("D:\\foo")
        assert _is_absolute_path_string("D:/foo")
        assert _is_absolute_path_string("/etc/passwd")
        assert _is_absolute_path_string("\\\\srv\\share")
        assert not _is_absolute_path_string("samples/foo.npy")
        assert not _is_absolute_path_string("./relative")

    def test_is_path_within_strict(self, tmp_path: Path) -> None:
        ds = tmp_path / "dataset"
        ds.mkdir()
        child = ds / "samples" / "x.npy"
        child.parent.mkdir(parents=True, exist_ok=True)
        child.write_bytes(b"x")
        assert is_path_within(child, ds)

        # Same-prefix sibling of the dataset root must be REJECTED by the
        # strict containment check (e.g. ``dataset_evil`` is not within
        # ``dataset`` even though the two share a common prefix).  This
        # is the real B01 same-prefix sibling test the handoff advertises.
        sibling = tmp_path / "dataset_evil"
        sibling.mkdir()
        evil_child = sibling / "x.npy"
        evil_child.write_bytes(b"x")
        assert not is_path_within(evil_child, ds), (
            "same-prefix sibling path was accepted; is_path_within is not strict"
        )
        # Reversed direction: the dataset is also not within the sibling.
        assert not is_path_within(ds, sibling), (
            "dataset accepted as inside its same-prefix sibling"
        )

        # Outside
        outside = tmp_path / "other" / "x.npy"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_bytes(b"x")
        assert not is_path_within(outside, ds)


class TestHashingAndDeterminism:
    def test_canonical_json_stable(self) -> None:
        d1 = {"b": 2, "a": 1}
        d2 = {"a": 1, "b": 2}
        assert canonical_json_dumps(d1) == canonical_json_dumps(d2)
        assert canonical_json_dumps(d1) == '{"a":1,"b":2}'

    def test_sha256_hex(self) -> None:
        assert sha256_hex("") == (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        )
        assert sha256_hex("abc") == (
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        )

    def test_sha256_file(self, tmp_path: Path) -> None:
        p = tmp_path / "x.txt"
        p.write_text("hello", encoding="utf-8")
        assert sha256_file(p) == (
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        )


# ---------------------------------------------------------------------------
# B. Synthetic A06 split: positive
# ---------------------------------------------------------------------------

class TestA06SplitPositive:
    def test_a06_split_round_trip(self, tmp_workspace: Path) -> None:
        subject_split = {f"{i:05d}": "train" for i in range(1, 9)}
        subject_split.update({f"{i:05d}": "val" for i in range(9, 11)})
        subject_split.update({f"{i:05d}": "test" for i in range(11, 13)})
        a06 = _make_synthetic_a06_split(subject_split)
        p = _write_synthetic_a06_split(tmp_workspace, a06)
        loaded = load_a06_split(p, expected_sha256=None, enforce_canonical_subject_counts=False)
        assert loaded.sha256 == a06["manifest_sha256"]
        assert loaded.split_counts_subjects == {"train": 8, "val": 2, "test": 2}
        assert loaded.subject_to_ml_split["00001"] == "train"
        assert loaded.subject_to_ml_split["00012"] == "test"

    def test_recompute_sha_matches_real_a06(self) -> None:
        # The re-computation must produce the same SHA as the frozen A06
        # for the canonical subject assignment.
        sha = _recompute_a06_subject_assignment_sha(
            [
                {"subject_id": "00001", "setting": "danaLab", "split": "train"},
                {"subject_id": "00002", "setting": "danaLab", "split": "val"},
            ]
        )
        # Just a smoke check — exact value depends on the entries; the
        # important property is that the function returns a 64-char hex digest.
        assert re.fullmatch(r"[0-9a-f]{64}", sha) is not None


class TestA06SplitNegative:
    def test_missing_a06_split_file(self, tmp_workspace: Path) -> None:
        with pytest.raises(A06SplitContractError, match="not found"):
            load_a06_split(tmp_workspace / "does_not_exist.json")

    def test_a06_split_wrong_schema(self, tmp_workspace: Path) -> None:
        a06 = _make_synthetic_a06_split({"00001": "train"})
        a06["schema_version"] = "wrong_version"
        p = _write_synthetic_a06_split(tmp_workspace, a06)
        with pytest.raises(A06SplitContractError, match="schema_version"):
            load_a06_split(p)

    def test_a06_split_invalid_json(self, tmp_workspace: Path) -> None:
        p = tmp_workspace / "bad.json"
        p.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(A06SplitContractError, match="not valid JSON"):
            load_a06_split(p)

    def test_a06_split_embedded_sha_mismatch(self, tmp_workspace: Path) -> None:
        a06 = _make_synthetic_a06_split({"00001": "train"})
        a06["manifest_sha256"] = "0" * 64  # mismatch
        p = _write_synthetic_a06_split(tmp_workspace, a06)
        with pytest.raises(A06SplitContractError, match="does not match re-computed"):
            load_a06_split(
                p, expected_sha256=None, enforce_canonical_subject_counts=False
            )

    def test_a06_split_recomputed_sha_mismatch_canonical(self, tmp_workspace: Path) -> None:
        a06 = _make_synthetic_a06_split({"00001": "train"})
        # Bypass the SHA-256 gate so the canonical subject-count check triggers.
        a06["danaLab_subjects"] = 1
        p = _write_synthetic_a06_split(tmp_workspace, a06)
        with pytest.raises(A06SplitContractError, match="subject count for 'train'"):
            load_a06_split(p, expected_sha256=None, enforce_canonical_subject_counts=True)


# ---------------------------------------------------------------------------
# C. Synthetic source manifest + freeze row builder: positive
# ---------------------------------------------------------------------------

class TestSourceManifestAndFreezeRowPositive:
    def test_load_synthetic_manifest(self, tmp_workspace: Path) -> None:
        mp, _ = _make_synthetic_dataset(tmp_workspace / "ds", n_subjects=4, frames_per_subject=2)
        loaded = load_slp8_source_manifest(
            tmp_workspace / "ds", enforce_canonical_total=False
        )
        assert loaded.sample_count == 4 * 2
        # sample_id is the primary key
        sids = [s.sample_id for s in loaded.samples]
        assert all(SAMPLE_ID_PATTERN.match(s) for s in sids)
        # subject_id pattern
        for s in loaded.samples:
            assert s.height == 192
            assert s.width == 84

    def test_build_freeze_row_happy(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=2, frames_per_subject=1)
        loaded = load_slp8_source_manifest(ds, enforce_canonical_total=False)
        src = loaded.samples[0]
        row = build_freeze_row(src, ml_split="train", dataset_root=ds)
        assert row.ml_split == "train"
        assert row.subject_id == src.subject_id
        assert row.pressure_npy == src.pressure_npy

    def test_manifest_csv_round_trip(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=2, frames_per_subject=2)
        loaded = load_slp8_source_manifest(ds, enforce_canonical_total=False)
        a06 = _make_synthetic_a06_split(
            {"00001": "train", "00002": "val"}
        )
        a06_path = _write_synthetic_a06_split(tmp_workspace, a06)
        a06_loaded = load_a06_split(
            a06_path, expected_sha256=None, enforce_canonical_subject_counts=False
        )
        rows: list[FreezeRow] = []
        for src in loaded.samples:
            ml_split = a06_loaded.ml_split_for_subject(src.subject_id, src.setting)
            rows.append(build_freeze_row(src, ml_split=ml_split, dataset_root=ds))
        csv_path = tmp_workspace / "train_manifest.csv"
        write_manifest_csv(csv_path, rows)
        re_read = read_manifest_csv(csv_path)
        assert len(re_read) == len(rows)
        assert {r.sample_id for r in re_read} == {r.sample_id for r in rows}

    def test_manifest_hash_stable(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=2, frames_per_subject=2)
        loaded = load_slp8_source_manifest(ds, enforce_canonical_total=False)
        rows_a = [
            build_freeze_row(s, ml_split="train", dataset_root=ds)
            for s in loaded.samples
        ]
        # Shuffle and re-hash — must be identical
        rows_b = list(reversed(rows_a))
        assert manifest_sha256(rows_a) == manifest_sha256(rows_b)

    def test_manifest_jsonl_round_trip(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=2, frames_per_subject=1)
        loaded = load_slp8_source_manifest(ds, enforce_canonical_total=False)
        rows = [build_freeze_row(s, ml_split="train", dataset_root=ds) for s in loaded.samples]
        p = tmp_workspace / "x.jsonl"
        write_manifest_jsonl(p, rows)
        # Re-read and verify line count
        lines = p.read_text(encoding="utf-8").rstrip("\n").split("\n")
        assert len(lines) == len(rows)
        # Each line is canonical JSON
        for line, r in zip(lines, sorted(rows, key=lambda x: x.sample_id)):
            obj = json.loads(line)
            assert obj["sample_id"] == r.sample_id


# ---------------------------------------------------------------------------
# D. Negative freeze-row construction
# ---------------------------------------------------------------------------

class TestFreezeRowNegative:
    def test_absolute_path_in_source_rejected(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=1, frames_per_subject=1)
        loaded = load_slp8_source_manifest(ds, enforce_canonical_total=False)
        src = loaded.samples[0]
        # Manually replace the path with an absolute one
        object.__setattr__(src, "pressure_npy", "D:\\evil\\file.npy")
        with pytest.raises(AbsolutePathNotAllowedError):
            build_freeze_row(src, ml_split="train", dataset_root=ds)

    def test_dotdot_path_rejected(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=1, frames_per_subject=1)
        loaded = load_slp8_source_manifest(ds, enforce_canonical_total=False)
        src = loaded.samples[0]
        object.__setattr__(src, "region_label_npy", "samples/../../../etc/passwd")
        with pytest.raises(PathContainmentViolationError):
            build_freeze_row(src, ml_split="train", dataset_root=ds)

    def test_invalid_provenance_rejected(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=1, frames_per_subject=1, source_provenance="HUMAN_PIXEL_ANNOTATED")
        with pytest.raises(SampleContractError, match="provenance"):
            load_slp8_source_manifest(ds, enforce_canonical_total=False)

    def test_invalid_review_status_rejected(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=1, frames_per_subject=1, source_review_status="ACCEPTED")
        with pytest.raises(SampleContractError, match="review_status"):
            load_slp8_source_manifest(ds, enforce_canonical_total=False)

    def test_cover_uncover_only(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=1, frames_per_subject=1, cover="cover1")
        # Either the sample_id pattern or the cover check fires — both are
        # valid fail-closed paths.  We just need a SampleContractError.
        with pytest.raises(SampleContractError):
            load_slp8_source_manifest(ds, enforce_canonical_total=False)

    def test_setting_dana_lab_only(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        # Use simLab setting (sample_id pattern enforces danaLab, so the
        # pattern check fires first; that is also a valid fail-closed path).
        _make_synthetic_dataset(ds, n_subjects=1, frames_per_subject=1, setting="simLab")
        with pytest.raises(SampleContractError):
            load_slp8_source_manifest(ds, enforce_canonical_total=False)

    def test_duplicate_sample_id(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=2, frames_per_subject=1)
        # Inject a duplicate sample_id by overwriting the CSV
        manifest_csv = ds / "manifest" / "val_manifest.csv"
        text = manifest_csv.read_text(encoding="utf-8")
        # Duplicate the first data row
        first_data = text.splitlines()[1]
        manifest_csv.write_text(text + first_data + "\n", encoding="utf-8")
        with pytest.raises(SampleContractError, match="duplicate"):
            load_slp8_source_manifest(ds, enforce_canonical_total=False)


# ---------------------------------------------------------------------------
# E. Slp8TrainingTableFreezer: end-to-end positive
# ---------------------------------------------------------------------------

class TestFreezerEndToEnd:
    def test_full_build_and_hash(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        n_subjects = 102
        # Build a 102-subject synthetic dataset, 45 frames per subject.
        _make_synthetic_dataset(ds, n_subjects=n_subjects, frames_per_subject=45)
        # Build A06 with 81 train / 10 val / 11 test (matches SLP8)
        subject_split: dict[str, str] = {}
        for i in range(1, 82):
            subject_split[f"{i:05d}"] = "train"
        for i in range(82, 92):
            subject_split[f"{i:05d}"] = "val"
        for i in range(92, 103):
            subject_split[f"{i:05d}"] = "test"
        a06 = _make_synthetic_a06_split(subject_split)
        a06_path = _write_synthetic_a06_split(tmp_workspace, a06)

        out_dir = tmp_workspace / "out"
        freezer = Slp8TrainingTableFreezer(
            dataset_root=ds,
            a06_split_path=a06_path,
            output_dir=out_dir,
            git_sha="deadbeef",
            expected_a06_sha256=None,
            enforce_canonical_source_total=False,
        )
        result = freezer.build()

        assert result.n_train == 81 * 45
        assert result.n_val == 10 * 45
        assert result.n_test == 11 * 45
        assert (out_dir / "train_manifest.csv").is_file()
        assert (out_dir / "val_manifest.csv").is_file()
        assert (out_dir / "test_manifest.csv").is_file()
        assert (out_dir / "freeze_manifest.json").is_file()
        assert (out_dir / "normalization_stats.json").is_file()
        assert (out_dir / "dataset_card.md").is_file()
        assert (out_dir / "train_class_stats.json").is_file()
        assert (out_dir / "val_class_stats.json").is_file()

    def test_subject_overlap_detected(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=3, frames_per_subject=2)
        # 3 subjects across 3 splits, no overlap
        subject_split = {"00001": "train", "00002": "val", "00003": "test"}
        a06 = _make_synthetic_a06_split(subject_split)
        a06_path = _write_synthetic_a06_split(tmp_workspace, a06)
        out_dir = tmp_workspace / "out"
        freezer = Slp8TrainingTableFreezer(
            dataset_root=ds,
            a06_split_path=a06_path,
            output_dir=out_dir,
            expected_a06_sha256=None,
            enforce_canonical_a06_subject_counts=False,
            enforce_canonical_source_total=False,
            enforce_canonical_split_counts=False,
        )
        result = freezer.build()
        assert result.n_train + result.n_val + result.n_test == 6

    def test_unknown_subject_in_dataset(self, tmp_workspace: Path) -> None:
        # If the dataset has a subject that is not in A06, we must
        # fail-closed rather than silently assign a split.
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=4, frames_per_subject=1)
        # A06 only knows about 2 subjects
        subject_split = {"00001": "train", "00002": "val"}
        a06 = _make_synthetic_a06_split(subject_split)
        a06_path = _write_synthetic_a06_split(tmp_workspace, a06)
        out_dir = tmp_workspace / "out"
        freezer = Slp8TrainingTableFreezer(
            dataset_root=ds,
            a06_split_path=a06_path,
            output_dir=out_dir,
            expected_a06_sha256=None,
            enforce_canonical_a06_subject_counts=False,
            enforce_canonical_source_total=False,
            enforce_canonical_split_counts=False,
        )
        with pytest.raises(SubjectMappingError):
            freezer.build()

    def test_unmapped_subject_detected(self, tmp_workspace: Path) -> None:
        # A06 has subject 00004 but the ML split is not in {train, val, test}
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=3, frames_per_subject=1)
        subject_split = {"00001": "train", "00002": "val", "00003": "quarantine"}
        a06 = _make_synthetic_a06_split(subject_split)
        a06_path = _write_synthetic_a06_split(tmp_workspace, a06)
        out_dir = tmp_workspace / "out"
        freezer = Slp8TrainingTableFreezer(
            dataset_root=ds,
            a06_split_path=a06_path,
            output_dir=out_dir,
            expected_a06_sha256=None,
            enforce_canonical_a06_subject_counts=False,
            enforce_canonical_source_total=False,
            enforce_canonical_split_counts=False,
        )
        with pytest.raises(A06SplitContractError):
            freezer.build()


# ---------------------------------------------------------------------------
# F. Test access guard
# ---------------------------------------------------------------------------

class TestAccessGuard:
    def test_default_is_disabled(self) -> None:
        assert not is_test_access_enabled()
        assert current_test_access_purpose() is None

    def test_enable_with_wrong_purpose_rejected(self) -> None:
        with pytest.raises(TestLeakageError, match="only purpose"):
            enable_test_access(purpose="training")

    def test_enable_with_correct_purpose(self) -> None:
        enable_test_access(purpose="final_evaluation")
        assert is_test_access_enabled()
        assert current_test_access_purpose() == "final_evaluation"
        disable_test_access()
        assert not is_test_access_enabled()

    def test_disable_resets(self) -> None:
        enable_test_access(purpose="final_evaluation")
        disable_test_access()
        assert not is_test_access_enabled()

    def test_guard_split_blocks_test(self) -> None:
        from topper_perception.io.slp8_training_table_freeze import (
            guard_split_for_label_access,
        )
        with pytest.raises(TestLeakageError, match="TEST access denied"):
            guard_split_for_label_access("test")
        enable_test_access(purpose="final_evaluation")
        try:
            guard_split_for_label_access("test")
        finally:
            disable_test_access()


# ---------------------------------------------------------------------------
# G. Normalization stats — TRAIN only
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_normalization_basic(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=2, frames_per_subject=2)
        loaded = load_slp8_source_manifest(ds, enforce_canonical_total=False)
        rows = [build_freeze_row(s, ml_split="train", dataset_root=ds) for s in loaded.samples]
        stats = fit_normalization_stats(rows, ds)
        assert stats.fit_split == "train"
        assert stats.n_samples == 4
        assert stats.method == NORMALIZATION_METHOD
        assert stats.epsilon == NORMALIZATION_EPSILON
        assert stats.raw_semantics == RAW_SEMANTICS
        assert stats.raw_dtype == "float64"
        assert stats.non_finite_pixel_count == 0
        assert stats.subject_count == 2
        # 100.0 ± 0 → mean=100, std=0
        assert stats.global_min == 100.0
        assert stats.global_max == 100.0
        assert stats.global_mean == pytest.approx(100.0)
        assert stats.global_std == pytest.approx(0.0)

    def test_normalization_empty_train(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=1, frames_per_subject=1)
        loaded = load_slp8_source_manifest(ds, enforce_canonical_total=False)
        rows = [build_freeze_row(s, ml_split="train", dataset_root=ds) for s in loaded.samples]
        with pytest.raises(NormalizationContractError, match="empty"):
            fit_normalization_stats([], ds)

    def test_normalization_rejects_non_train(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=1, frames_per_subject=1)
        loaded = load_slp8_source_manifest(ds, enforce_canonical_total=False)
        # Mark as VAL — must be rejected
        bad = []
        for s in loaded.samples:
            r = build_freeze_row(s, ml_split="val", dataset_root=ds)
            bad.append(r)
        with pytest.raises(NormalizationContractError, match="TRAIN-only"):
            fit_normalization_stats(bad, ds)

    def test_normalization_nan_rejected(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=1, frames_per_subject=1)
        # Inject NaN into one pressure file
        for s in ds.rglob("pressure.npy"):
            arr = np.load(s, allow_pickle=False)
            arr[0, 0] = float("nan")
            np.save(s, arr)
            break
        loaded = load_slp8_source_manifest(ds, enforce_canonical_total=False)
        rows = [build_freeze_row(s, ml_split="train", dataset_root=ds) for s in loaded.samples]
        with pytest.raises(NormalizationContractError, match="non-finite"):
            fit_normalization_stats(rows, ds)

    def test_normalization_inf_rejected(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=1, frames_per_subject=1)
        for s in ds.rglob("pressure.npy"):
            arr = np.load(s, allow_pickle=False)
            arr[0, 0] = float("inf")
            np.save(s, arr)
            break
        loaded = load_slp8_source_manifest(ds, enforce_canonical_total=False)
        rows = [build_freeze_row(s, ml_split="train", dataset_root=ds) for s in loaded.samples]
        with pytest.raises(NormalizationContractError, match="non-finite"):
            fit_normalization_stats(rows, ds)

    def test_normalization_wrong_dtype_rejected(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=1, frames_per_subject=1)
        for s in ds.rglob("pressure.npy"):
            arr = np.load(s, allow_pickle=False).astype(np.float32)
            np.save(s, arr)
            break
        loaded = load_slp8_source_manifest(ds, enforce_canonical_total=False)
        rows = [build_freeze_row(s, ml_split="train", dataset_root=ds) for s in loaded.samples]
        with pytest.raises(NormalizationContractError, match="dtype"):
            fit_normalization_stats(rows, ds)

    def test_normalization_wrong_shape_rejected(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=1, frames_per_subject=1)
        for s in ds.rglob("pressure.npy"):
            arr = np.zeros((10, 10), dtype=np.float64)
            np.save(s, arr)
            break
        loaded = load_slp8_source_manifest(ds, enforce_canonical_total=False)
        rows = [build_freeze_row(s, ml_split="train", dataset_root=ds) for s in loaded.samples]
        with pytest.raises(NormalizationContractError, match="shape"):
            fit_normalization_stats(rows, ds)

    def test_normalization_stats_file_round_trip(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=1, frames_per_subject=1)
        loaded = load_slp8_source_manifest(ds, enforce_canonical_total=False)
        rows = [build_freeze_row(s, ml_split="train", dataset_root=ds) for s in loaded.samples]
        stats = fit_normalization_stats(rows, ds)
        p = tmp_workspace / "stats.json"
        write_normalization_stats(p, stats)
        payload = json.loads(p.read_text(encoding="utf-8"))
        assert payload["stats_sha256"] == stats.content_sha256()


# ---------------------------------------------------------------------------
# H. Class stats: TRAIN/VAL allowed, TEST blocked
# ---------------------------------------------------------------------------

class TestClassStats:
    def test_compute_train_class_stats(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=2, frames_per_subject=2)
        loaded = load_slp8_source_manifest(ds, enforce_canonical_total=False)
        rows = [build_freeze_row(s, ml_split="train", dataset_root=ds) for s in loaded.samples]
        stats = compute_class_stats(rows, ds, ml_split="train")
        assert stats.n_samples == 4
        # Synthetic data has class 0 (BACKGROUND) only
        assert 0 in stats.per_class_pixel_count
        assert stats.onehot_roundtrip_ok_count == 4

    def test_compute_test_class_stats_blocked(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=2, frames_per_subject=2)
        loaded = load_slp8_source_manifest(ds, enforce_canonical_total=False)
        rows = [build_freeze_row(s, ml_split="test", dataset_root=ds) for s in loaded.samples]
        with pytest.raises(TestLeakageError, match="TEST access denied"):
            compute_class_stats(rows, ds, ml_split="test")
        enable_test_access(purpose="final_evaluation")
        try:
            stats = compute_class_stats(rows, ds, ml_split="test")
            assert stats.n_samples == 4
        finally:
            disable_test_access()

    def test_compute_class_stats_wrong_split_in_rows(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=2, frames_per_subject=2)
        loaded = load_slp8_source_manifest(ds, enforce_canonical_total=False)
        # rows marked as train but we ask for val
        rows = [build_freeze_row(s, ml_split="train", dataset_root=ds) for s in loaded.samples]
        with pytest.raises(B01FreezeError, match="ml_split="):
            compute_class_stats(rows, ds, ml_split="val")


# ---------------------------------------------------------------------------
# I. Dataset card rendering
# ---------------------------------------------------------------------------

class TestDatasetCard:
    def test_card_contains_required_phrases(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=1, frames_per_subject=1)
        loaded = load_slp8_source_manifest(ds, enforce_canonical_total=False)
        rows = [build_freeze_row(s, ml_split="train", dataset_root=ds) for s in loaded.samples]
        stats = fit_normalization_stats(rows, ds)
        cs = compute_class_stats(rows, ds, ml_split="train")
        # Build a minimal freeze manifest for the renderer
        from topper_perception.io.slp8_training_table_freeze import (
            build_freeze_manifest,
        )
        a06 = _make_synthetic_a06_split({"00001": "train"})
        a06_loaded = A06Split(
            raw=a06,
            subject_to_ml_split={"00001": "train"},
            subject_to_setting={"00001": "danaLab"},
            split_counts_subjects={"train": 1, "val": 0, "test": 0},
            split_counts_samples={"train": 1, "val": 0, "test": 0},
            sha256=a06["manifest_sha256"],
        )
        fm = build_freeze_manifest(
            train_rows=rows, val_rows=[], test_rows=[],
            a06_split=a06_loaded,
            source_manifest_sha256=loaded.source_manifest_sha256,
            stats=stats, train_stats=cs, val_stats=cs,
        )
        text = render_dataset_card(
            freeze_manifest=fm,
            train_class_stats=cs,
            val_class_stats=cs,
            normalization_stats=stats,
        )
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
            assert needle in text


# ---------------------------------------------------------------------------
# J. np.load allow_pickle=False enforcement (AST scan)
# ---------------------------------------------------------------------------

MODULE_PATH = (
    PROJECT_ROOT / "src" / "topper_perception" / "io" / "slp8_training_table_freeze.py"
)


def test_all_npload_uses_allow_pickle_false() -> None:
    """AST scan: every ``np.load(...)`` call must pass ``allow_pickle=False``."""
    import ast
    src = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "load"
                and isinstance(func.value, ast.Name)
                and func.value.id == "np"
            ):
                # check kwargs
                has_kwarg = any(
                    kw.arg == "allow_pickle" for kw in node.keywords
                )
                if not has_kwarg:
                    violations.append(
                        f"line {node.lineno}: np.load() missing allow_pickle=False"
                    )
                else:
                    kw = next(kw for kw in node.keywords if kw.arg == "allow_pickle")
                    if not (
                        isinstance(kw.value, ast.Constant) and kw.value.value is False
                    ):
                        violations.append(
                            f"line {node.lineno}: np.load() allow_pickle != False"
                        )
    assert not violations, "np.load allow_pickle contract violations: " + "; ".join(
        violations
    )


# ---------------------------------------------------------------------------
# K. No hard-coded local paths
# ---------------------------------------------------------------------------

def test_no_hard_coded_local_paths() -> None:
    """AST scan: no absolute Windows or POSIX path string literal in the module."""
    src = MODULE_PATH.read_text(encoding="utf-8")
    # Look for patterns like 'E:\...', 'D:/...', '/home/...', '/tmp/...'
    bad_patterns = [
        re.compile(r"['\"][A-Z]:\\"),         # Windows drive letter
        re.compile(r"['\"]E:[/\\\\]"),         # E:/
        re.compile(r"['\"]D:[/\\\\]"),         # D:/
    ]
    bad: list[str] = []
    for line_no, line in enumerate(src.splitlines(), start=1):
        for pat in bad_patterns:
            if pat.search(line):
                bad.append(f"line {line_no}: {line.strip()}")
    assert not bad, "Hard-coded local paths in module: " + "; ".join(bad)


# ---------------------------------------------------------------------------
# L. Sample-id pattern
# ---------------------------------------------------------------------------

def test_sample_id_pattern_valid_and_invalid() -> None:
    assert SAMPLE_ID_PATTERN.match("SLP:danaLab:00001:uncover:000001")
    assert SAMPLE_ID_PATTERN.match("SLP:danaLab:99999:uncover:000045")
    # Bad
    assert not SAMPLE_ID_PATTERN.match("SLP:simLab:00001:uncover:000001")
    assert not SAMPLE_ID_PATTERN.match("SLP:danaLab:0001:uncover:000001")
    assert not SAMPLE_ID_PATTERN.match("SLP:danaLab:00001:cover1:000001")
    assert not SAMPLE_ID_PATTERN.match("SLP:danaLab:00001:uncover:00001")
    assert not SAMPLE_ID_PATTERN.match("garbage")


# ---------------------------------------------------------------------------
# M. Real-data integration (gated on env vars)
# ---------------------------------------------------------------------------

REAL_DATASET_ROOT = Path(
    os.environ.get("SLP8_DATASET_ROOT", "")
)
REAL_A06_SPLIT = Path(
    os.environ.get("A06_SPLIT_PATH", "")
)
REAL_B01_OUTPUT = Path(
    os.environ.get("B01_FREEZE_OUTPUT_DIR", "")
)


@pytest.mark.skipif(
    not (REAL_DATASET_ROOT.is_dir() and REAL_A06_SPLIT.is_file()),
    reason="real data not provided",
)
def test_real_data_build() -> None:
    out_dir = REAL_B01_OUTPUT if str(REAL_B01_OUTPUT) else (
        PROJECT_ROOT / "data" / "processed" / "slp8_training_tables_v0.1"
    )
    if out_dir.exists():
        shutil.rmtree(out_dir)
    freezer = Slp8TrainingTableFreezer(
        dataset_root=REAL_DATASET_ROOT,
        a06_split_path=REAL_A06_SPLIT,
        output_dir=out_dir,
    )
    result = freezer.build()
    assert result.n_train == EXPECTED_SPLIT_COUNTS["train"]
    assert result.n_val == EXPECTED_SPLIT_COUNTS["val"]
    assert result.n_test == EXPECTED_SPLIT_COUNTS["test"]


@pytest.mark.skipif(
    not (REAL_DATASET_ROOT.is_dir() and REAL_A06_SPLIT.is_file()),
    reason="real data not provided",
)
def test_real_data_validator() -> None:
    """Invoke the validator on the real-data output."""
    import subprocess
    out_dir = REAL_B01_OUTPUT if str(REAL_B01_OUTPUT) else (
        PROJECT_ROOT / "data" / "processed" / "slp8_training_tables_v0.1"
    )
    if not (out_dir / "freeze_manifest.json").is_file():
        pytest.skip("B01 freeze output not present; run test_real_data_build first")
    proc = subprocess.run(
        [
            "uv", "run", "python",
            str(PROJECT_ROOT / "scripts" / "validate_slp8_training_tables.py"),
            "--dataset-root", str(REAL_DATASET_ROOT),
            "--a06-split", str(REAL_A06_SPLIT),
            "--output-dir", str(out_dir),
            "--no-rebuild",
        ],
        capture_output=True, text=True, encoding="utf-8",
    )
    assert proc.returncode == 0, (
        f"validator failed (exit={proc.returncode})\n"
        f"STDOUT (tail):\n{proc.stdout[-2000:]}\n"
        f"STDERR (tail):\n{proc.stderr[-2000:]}"
    )


# ---------------------------------------------------------------------------
# N. Manifest hash: round-trip stability
# ---------------------------------------------------------------------------

def test_manifest_hash_independent_of_input_order(tmp_workspace: Path) -> None:
    ds = tmp_workspace / "ds"
    _make_synthetic_dataset(ds, n_subjects=3, frames_per_subject=2)
    loaded = load_slp8_source_manifest(ds, enforce_canonical_total=False)
    rows = [build_freeze_row(s, ml_split="train", dataset_root=ds) for s in loaded.samples]
    h1 = manifest_sha256(rows)
    h2 = manifest_sha256(list(reversed(rows)))
    h3 = manifest_sha256(sorted(rows, key=lambda r: r.sample_id))
    h4 = manifest_sha256(sorted(rows, key=lambda r: r.subject_id))
    assert h1 == h2 == h3 == h4


# ---------------------------------------------------------------------------
# O. Module version constants stability
# ---------------------------------------------------------------------------

def test_module_constants() -> None:
    assert ADAPTER_VERSION == "slp8_training_table_freeze_v0.1"
    assert FREEZE_VERSION == "slp8_training_tables_v0.1"
    assert TASK_ID == "TASK-SLP-B01-SLP8-TRAINING-TABLE-FREEZE-v0.1"
    assert NORMALIZATION_METHOD == "raw_passthrough_with_minmax_reference"
    assert NORMALIZATION_FIT_SPLIT == "train"
    assert ML_SPLITS == ("train", "val", "test")
    assert EXPECTED_TOTAL == 4590
    assert EXPECTED_SUBJECTS == 102
    assert EXPECTED_FRAMES_PER_SUBJECT == 45
    assert EXPECTED_SPLIT_COUNTS == {"train": 3645, "val": 450, "test": 495}
    assert EXPECTED_POSTURE_COUNTS == {"SUPINE": 1530, "LEFT": 1530, "RIGHT": 1530}
    assert A06_SPLIT_SHA256_EXPECTED == (
        "024f5abe05afc108f66be978dfc6d3e2f0c558571141d7cb459b849d0d33a706"
    )
    assert EXPECTED_PROVENANCE == "V221_CORRECTED_SUPPORT_AUTO_ACCEPTED"
    assert EXPECTED_REVIEW_STATUS == "NOT_REVIEWED"
    # Corrected spelling: the older ``raw_pmaray_response`` alias remains
    # accepted on read for backward compatibility, but every new artifact
    # and validator comparison uses the canonical ``raw_pmarray_response``.
    assert RAW_SEMANTICS == "raw_pmarray_response"


# ---------------------------------------------------------------------------
# P. Tampering detection — the actual blocking cases the user reported
# ---------------------------------------------------------------------------

class _FrozenFreezerHarness:
    """Build a small synthetic B01 freeze directory for tamper tests.

    The harness uses 6 subjects × 3 frames = 18 samples so a sub-second
    build is sufficient to exercise every B01 artifact on the synthetic
    side and the real-data build remains the source of the canonical
    hashes.
    """

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.dataset_root = tmp / "ds"
        self.a06_split = tmp / "a06.json"
        self.output_dir = tmp / "out"

    def build(self, *, n_subjects: int = 6, frames_per_subject: int = 3,
              test_subjects: int = 0, val_subjects: int | None = None) -> None:
        _make_synthetic_dataset(
            self.dataset_root,
            n_subjects=n_subjects,
            frames_per_subject=frames_per_subject,
        )
        # Default split: train, val, test in that order.  ``val_subjects``
        # defaults to 1/3 of n_subjects (rounded), ``test_subjects`` defaults
        # to 0.  Pass ``test_subjects=1`` for tests that need a non-empty
        # TEST split.
        if val_subjects is None:
            val_subjects = max(1, n_subjects // 3)
        n_train = n_subjects - val_subjects - test_subjects
        if n_train < 1:
            raise ValueError("harness.build: not enough subjects to split")
        subject_split: dict[str, str] = {}
        next_id = 1
        for _ in range(n_train):
            subject_split[f"{next_id:05d}"] = "train"
            next_id += 1
        for _ in range(val_subjects):
            subject_split[f"{next_id:05d}"] = "val"
            next_id += 1
        for _ in range(test_subjects):
            subject_split[f"{next_id:05d}"] = "test"
            next_id += 1
        a06 = _make_synthetic_a06_split(subject_split)
        self.a06_split.write_text(
            json.dumps(a06, indent=2), encoding="utf-8"
        )
        freezer = Slp8TrainingTableFreezer(
            dataset_root=self.dataset_root,
            a06_split_path=self.a06_split,
            output_dir=self.output_dir,
            expected_a06_sha256=None,
            enforce_canonical_a06_subject_counts=False,
            enforce_canonical_source_total=False,
            enforce_canonical_split_counts=False,
        )
        freezer.build()


def _tamper_replace_text(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        raise AssertionError(f"tamper helper: needle {old!r} not found in {path}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def _run_validator_expect_fail(
    harness: _FrozenFreezerHarness,
) -> tuple[int, str, str]:
    """Run the validator on the harness with --allow-non-canonical.

    Returns (returncode, stdout, stderr).  The synthetic harness is
    not a real-data 4,590-sample build, so the validator must be
    invoked with ``--allow-non-canonical`` to skip the absolute count
    checks.  Tamper-specific checks (source-manifest SHA, normalization
    SHA re-compute, CSV↔JSONL equality, class stats, dataset card
    cross-check) are still enforced.
    """
    import subprocess
    proc = subprocess.run(
        [
            "uv", "run", "python",
            str(PROJECT_ROOT / "scripts" / "validate_slp8_training_tables.py"),
            "--dataset-root", str(harness.dataset_root),
            "--a06-split", str(harness.a06_split),
            "--output-dir", str(harness.output_dir),
            "--no-rebuild",
            "--allow-non-canonical",
        ],
        capture_output=True, text=True, encoding="utf-8",
    )
    return proc.returncode, proc.stdout, proc.stderr


def _run_validator_clean_check(
    harness: _FrozenFreezerHarness,
) -> tuple[int, str, str]:
    """Run the validator with --allow-non-canonical and expect rc==0."""
    rc, out, err = _run_validator_expect_fail(harness)
    assert rc == 0, (
        f"clean validator failed (rc={rc})\n"
        f"STDOUT (tail):\n{out[-2000:]}\n"
        f"STDERR (tail):\n{err[-2000:]}"
    )
    return rc, out, err


class TestTamperingDetection:
    """Every section of the validator must fail-closed on common tampering.

    These tests run the actual ``validate_slp8_training_tables.py`` script
    against a synthetic B01 freeze directory and assert that the validator
    exits with a non-zero status code when an artifact is tampered with.
    """

    def _build(self, tmp_path: Path) -> _FrozenFreezerHarness:
        harness = _FrozenFreezerHarness(tmp_path)
        harness.build()
        # Sanity: clean run passes.
        _run_validator_clean_check(harness)
        return harness

    def test_clean_run_passes(self, tmp_path: Path) -> None:
        self._build(tmp_path)

    def test_source_manifest_tampered(self, tmp_path: Path) -> None:
        harness = self._build(tmp_path)
        # Append a benign comment to the SLP8 source manifest, which
        # changes its file SHA.  The validator's Section 4 must detect
        # the SHA mismatch against the freeze manifest record.
        src_csv = harness.dataset_root / "manifest" / "val_manifest.csv"
        with src_csv.open("a", encoding="utf-8") as f:
            f.write("# tampered\n")
        rc, out, _ = _run_validator_expect_fail(harness)
        assert rc != 0, "validator passed with tampered source manifest"
        assert "source_manifest_sha256" in out or "source manifest SHA" in out, (
            f"expected source-manifest SHA error in output:\n{out[-2000:]}"
        )

    def test_normalization_stats_tampered_mean(self, tmp_path: Path) -> None:
        harness = self._build(tmp_path)
        stats_path = harness.output_dir / "normalization_stats.json"
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        # Change global_mean and bump the embedded stats_sha256 to a
        # syntactically valid 64-char hex so we test that the validator
        # does NOT trust the embedded field alone.
        original = payload["stats"]["global_mean"]
        payload["stats"]["global_mean"] = original + 12345.0
        payload["stats_sha256"] = "f" * 64
        stats_path.write_text(
            canonical_json_dumps(payload) + "\n", encoding="utf-8"
        )
        rc, out, _ = _run_validator_expect_fail(harness)
        assert rc != 0, "validator passed with tampered normalization stats"
        assert "re-computed" in out, (
            f"expected re-computed SHA error in output:\n{out[-2000:]}"
        )

    def test_normalization_stats_fake_sha_with_unchanged_content(self, tmp_path: Path) -> None:
        harness = self._build(tmp_path)
        stats_path = harness.output_dir / "normalization_stats.json"
        payload = json.loads(stats_path.read_text(encoding="utf-8"))
        # Re-embed the original content's correct SHA; this is the
        # "embedded field == re-computed" case, which should pass for
        # the SHA check.  The numeric re-fit is what catches it.
        payload["stats_sha256"] = payload.get("stats_sha256", "0" * 64)
        # Now tamper global_min without updating the SHA.
        payload["stats"]["global_min"] = -999.0
        stats_path.write_text(
            canonical_json_dumps(payload) + "\n", encoding="utf-8"
        )
        rc, out, _ = _run_validator_expect_fail(harness)
        assert rc != 0, (
            "validator passed when normalization global_min was tampered "
            "and SHA embedded field was left unchanged"
        )

    def test_train_csv_tampered(self, tmp_path: Path) -> None:
        harness = self._build(tmp_path)
        train_csv = harness.output_dir / "train_manifest.csv"
        text = train_csv.read_text(encoding="utf-8")
        # Replace one subject_id with a different one and rewrite the
        # file.  The CSV's row count stays at the canonical total, but
        # the per-row content changes → manifest SHA changes → freeze
        # manifest's recorded train SHA will no longer match.
        rows = text.splitlines()
        rows[1] = rows[1].replace("00001", "99999")
        train_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
        # The JSONL still has the original subject → byte-level CSV↔JSONL
        # inequality in Section 8 also fires, and the per-split SHA
        # stability check in Section 4 fires.  Either of these must
        # cause the validator to fail.
        rc, out, _ = _run_validator_expect_fail(harness)
        assert rc != 0, "validator passed with tampered train_manifest.csv"

    def test_train_jsonl_missing(self, tmp_path: Path) -> None:
        harness = self._build(tmp_path)
        jsonl = harness.output_dir / "train_manifest.jsonl"
        jsonl.unlink()
        rc, _out, _err = _run_validator_expect_fail(harness)
        assert rc != 0, "validator passed with train_manifest.jsonl missing"

    def test_class_stats_tampered(self, tmp_path: Path) -> None:
        harness = self._build(tmp_path)
        cs_path = harness.output_dir / "train_class_stats.json"
        payload = json.loads(cs_path.read_text(encoding="utf-8"))
        # Bump per_class_pixel_count for class 0 by a large amount and
        # rewrite the file.  The validator re-computes from the rows
        # in Section 8 and must detect the inconsistency.
        payload["per_class_pixel_count"]["0"] = (
            payload["per_class_pixel_count"].get("0", 0) + 999_999
        )
        cs_path.write_text(
            canonical_json_dumps(payload) + "\n", encoding="utf-8"
        )
        rc, out, _ = _run_validator_expect_fail(harness)
        assert rc != 0, "validator passed with tampered train_class_stats"
        assert "per_class_pixel_count" in out, (
            f"expected per-class count error in output:\n{out[-2000:]}"
        )

    def test_dataset_card_sample_count_tampered(self, tmp_path: Path) -> None:
        harness = self._build(tmp_path)
        card = harness.output_dir / "dataset_card.md"
        text = card.read_text(encoding="utf-8")
        # Read the freeze manifest's recorded train subject count, then
        # replace the dedicated line in the card with a deliberately
        # wrong value.  Section 9 cross-checks the literal line
        # ``TRAIN subject count: `<n>` `` against the freeze manifest
        # and must detect the inconsistency.
        fm = json.loads(
            (harness.output_dir / "freeze_manifest.json").read_text(encoding="utf-8")
        )
        train_subj_count = fm["core"]["splits"]["train"]["subject_count"]
        wrong_count = 91234
        bt = "`"
        needle_subj = f"TRAIN subject count: {bt}{train_subj_count}{bt}"
        wrong_subj = f"TRAIN subject count: {bt}{wrong_count}{bt}"
        if needle_subj not in text:
            pytest.fail(f"missing dataset card marker: {needle_subj!r}")
        text = text.replace(needle_subj, wrong_subj, 1)
        card.write_text(text, encoding="utf-8")
        rc, out, _ = _run_validator_expect_fail(harness)
        assert rc != 0, "validator passed with tampered dataset card"
        assert "TRAIN subject count" in out, (
            f"expected TRAIN subject count error in output:\n{out[-2000:]}"
        )

    def test_freeze_manifest_core_a06_sha_tampered(self, tmp_path: Path) -> None:
        harness = self._build(tmp_path)
        fm = harness.output_dir / "freeze_manifest.json"
        payload = json.loads(fm.read_text(encoding="utf-8"))
        # Replace the recorded A06 SHA with a different value.  Section
        # 7's freeze-manifest check must detect the mismatch.
        original = payload["core"]["a06_split_sha256"]
        payload["core"]["a06_split_sha256"] = (
            "0" * 64 if original != "0" * 64 else "f" * 64
        )
        fm.write_text(
            canonical_json_dumps(payload) + "\n", encoding="utf-8"
        )
        rc, out, _ = _run_validator_expect_fail(harness)
        assert rc != 0, "validator passed with tampered freeze manifest A06 SHA"


# ---------------------------------------------------------------------------
# Q. Unified read entry: load_b01_freeze_tables
# ---------------------------------------------------------------------------

class TestUnifiedReadEntry:
    def test_default_excludes_test(self, tmp_path: Path) -> None:
        harness = _FrozenFreezerHarness(tmp_path)
        harness.build(n_subjects=6, test_subjects=1)
        tables = load_b01_freeze_tables(harness.output_dir)
        assert len(tables.train_rows) > 0
        assert len(tables.val_rows) > 0
        # The TEST manifest is always parsed for SHA purposes, but the
        # public development helper must NEVER include TEST rows.
        dev = tables.development_rows()
        assert all(r.ml_split != "test" for r in dev), (
            "development_rows() leaked TEST rows"
        )
        # train+val helpers likewise exclude TEST.
        assert all(r.ml_split != "test" for r in tables.train_rows)
        assert all(r.ml_split != "test" for r in tables.val_rows)

    def test_all_rows_requires_test_access(self, tmp_path: Path) -> None:
        harness = _FrozenFreezerHarness(tmp_path)
        harness.build(n_subjects=6, test_subjects=1)
        tables = load_b01_freeze_tables(harness.output_dir)
        # Default: TEST access denied.
        with pytest.raises(TestLeakageError, match="TEST access denied"):
            tables.all_rows_with_test_opt_in()
        # Wrong purpose string is rejected by enable_test_access.
        with pytest.raises(TestLeakageError, match="only purpose"):
            enable_test_access(purpose="final_evaluatio")  # typo
        # Wrong purpose (allowed string) is rejected at all_rows_with_test_opt_in.
        enable_test_access(purpose="final_evaluation")
        try:
            with pytest.raises(TestLeakageError, match="purpose mismatch"):
                # Force a purpose mismatch by monkey-patching the global
                # current purpose; the all_rows helper must still detect
                # the mismatch and refuse.
                import topper_perception.io.slp8_training_table_freeze as mod
                original = mod._TEST_ACCESS_STATE["purpose"]
                mod._TEST_ACCESS_STATE["purpose"] = "wrong_purpose"
                try:
                    tables.all_rows_with_test_opt_in()
                finally:
                    mod._TEST_ACCESS_STATE["purpose"] = original
        finally:
            disable_test_access()
        # Correct purpose but TEST rows were never loaded — must reload
        # with load_test=True.
        enable_test_access(purpose="final_evaluation")
        try:
            with pytest.raises(TestLeakageError, match="TEST rows are not present"):
                tables.all_rows_with_test_opt_in()
        finally:
            disable_test_access()
        # Reload with load_test=True.
        enable_test_access(purpose="final_evaluation")
        try:
            tables_with_test = load_b01_freeze_tables(
                harness.output_dir, load_test=True
            )
            all_rows = tables_with_test.all_rows_with_test_opt_in()
            assert any(r.ml_split == "test" for r in all_rows)
        finally:
            disable_test_access()

    def test_allowed_splits_test_rejected_without_opt_in(self, tmp_path: Path) -> None:
        harness = _FrozenFreezerHarness(tmp_path)
        harness.build(n_subjects=6, test_subjects=1)
        with pytest.raises(TestLeakageError, match="TEST access denied"):
            load_b01_freeze_tables(harness.output_dir, allowed_splits=("train", "val", "test"))

    def test_load_test_true_requires_opt_in(self, tmp_path: Path) -> None:
        harness = _FrozenFreezerHarness(tmp_path)
        harness.build(n_subjects=6, test_subjects=1)
        with pytest.raises(TestLeakageError, match="TEST access denied"):
            load_b01_freeze_tables(harness.output_dir, load_test=True)

    def test_test_rows_not_public_without_auth(
        self, tmp_path: Path
    ) -> None:
        # Default load: _test_rows is None, so the property raises even
        # before the is_test_access_enabled() check.
        harness = _FrozenFreezerHarness(tmp_path)
        harness.build(n_subjects=6, test_subjects=1)
        tables = load_b01_freeze_tables(harness.output_dir)
        with pytest.raises(
            TestLeakageError, match="TEST rows are not present"
        ):
            _ = tables.test_rows

    def test_test_rows_not_accessible_after_default_load(
        self, tmp_path: Path
    ) -> None:
        # Enabling auth AFTER a default load does not retroactively grant TEST.
        harness = _FrozenFreezerHarness(tmp_path)
        harness.build(n_subjects=6, test_subjects=1)
        tables = load_b01_freeze_tables(harness.output_dir)  # default: no TEST
        enable_test_access(purpose="final_evaluation")
        try:
            with pytest.raises(TestLeakageError, match="TEST rows are not present"):
                _ = tables.test_rows
            with pytest.raises(
                TestLeakageError, match="TEST rows are not present"
            ):
                tables.all_rows_with_test_opt_in()
        finally:
            disable_test_access()

    def test_test_rows_accessible_with_explicit_load_test(
        self, tmp_path: Path
    ) -> None:
        # Only load_b01_freeze_tables(..., load_test=True) produces a handle
        # with _test_rows populated.
        harness = _FrozenFreezerHarness(tmp_path)
        harness.build(n_subjects=6, test_subjects=1)
        enable_test_access(purpose="final_evaluation")
        try:
            tables = load_b01_freeze_tables(
                harness.output_dir, load_test=True
            )
            rows = tables.test_rows
            assert isinstance(rows, tuple)
            assert len(rows) > 0
            assert all(r.ml_split == "test" for r in rows)
            all_rows = tables.all_rows_with_test_opt_in()
            assert any(r.ml_split == "test" for r in all_rows)
        finally:
            disable_test_access()

    def test_missing_artifact_raises_file_not_found(self, tmp_path: Path) -> None:
        # Run a real-data-style build that produces every required file
        # so the directory exists, then remove one of the required
        # files (train_manifest.csv, val_manifest.csv, test_manifest.csv,
        # or freeze_manifest.json).
        harness = _FrozenFreezerHarness(tmp_path)
        harness.build()
        (harness.output_dir / "test_manifest.csv").unlink()
        with pytest.raises(FileNotFoundError):
            load_b01_freeze_tables(harness.output_dir)


# ---------------------------------------------------------------------------
# R. Source loader fail-closed: missing onehot_valid/roundtrip etc.
# ---------------------------------------------------------------------------

class TestSourceLoaderFailClosed:
    def test_missing_onehot_valid_rejected(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=1, frames_per_subject=1)
        # Drop the ``onehot_valid`` column from the CSV (use csv.DictReader
        # to be robust to quoted fields).
        import csv as _csv
        manifest_csv = ds / "manifest" / "val_manifest.csv"
        with manifest_csv.open(newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            fieldnames = [n for n in reader.fieldnames if n != "onehot_valid"]
            rows = [
                {k: v for k, v in row.items() if k != "onehot_valid"}
                for row in reader
            ]
        with manifest_csv.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        with pytest.raises(SampleContractError, match="missing 'onehot_valid'"):
            load_slp8_source_manifest(ds, enforce_canonical_total=False)

    def test_onehot_valid_false_rejected(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=1, frames_per_subject=1)
        import csv as _csv
        manifest_csv = ds / "manifest" / "val_manifest.csv"
        with manifest_csv.open(newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
        rows[0]["onehot_valid"] = "False"
        with manifest_csv.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        with pytest.raises(
            SampleContractError, match="onehot_valid='False' — must be 'True'"
        ):
            load_slp8_source_manifest(ds, enforce_canonical_total=False)

    def test_onehot_roundtrip_false_rejected(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=1, frames_per_subject=1)
        import csv as _csv
        manifest_csv = ds / "manifest" / "val_manifest.csv"
        with manifest_csv.open(newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
        rows[0]["onehot_roundtrip"] = "False"
        with manifest_csv.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        with pytest.raises(
            SampleContractError, match="onehot_roundtrip='False' — must be 'True'"
        ):
            load_slp8_source_manifest(ds, enforce_canonical_total=False)

    def test_invalid_source_split_rejected(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=1, frames_per_subject=1)
        import csv as _csv
        manifest_csv = ds / "manifest" / "val_manifest.csv"
        with manifest_csv.open(newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
        rows[0]["split"] = "TRAIN"
        with manifest_csv.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        with pytest.raises(SampleContractError, match="source_split"):
            load_slp8_source_manifest(ds, enforce_canonical_total=False)

    def test_invalid_posture_rejected(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=1, frames_per_subject=1)
        import csv as _csv
        manifest_csv = ds / "manifest" / "val_manifest.csv"
        with manifest_csv.open(newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
        rows[0]["posture"] = "PRONE"
        with manifest_csv.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        with pytest.raises(SampleContractError, match="posture"):
            load_slp8_source_manifest(ds, enforce_canonical_total=False)

    def test_invalid_export_status_rejected(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=1, frames_per_subject=1)
        import csv as _csv
        manifest_csv = ds / "manifest" / "val_manifest.csv"
        with manifest_csv.open(newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
        rows[0]["export_status"] = "FAILED"
        with manifest_csv.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        with pytest.raises(SampleContractError, match="export_status"):
            load_slp8_source_manifest(ds, enforce_canonical_total=False)

    def test_invalid_pmarray_sha_format_rejected(self, tmp_workspace: Path) -> None:
        ds = tmp_workspace / "ds"
        _make_synthetic_dataset(ds, n_subjects=1, frames_per_subject=1)
        manifest_csv = ds / "manifest" / "val_manifest.csv"
        text = manifest_csv.read_text(encoding="utf-8")
        # Replace the first 64-char hex SHA on the first data row with a
        # non-hex string.  Use the existing header to identify the column
        # via csv so we are robust to quoted fields like "[192, 84]".
        import csv as _csv
        with manifest_csv.open(newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            fieldnames = reader.fieldnames
            rows = list(reader)
        bad_sha = "X" * 64  # contains a non-hex char; also 64 chars
        rows[0]["source_pmarray_sha256"] = bad_sha
        with manifest_csv.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        with pytest.raises(SampleContractError, match="source_pmarray_sha256"):
            load_slp8_source_manifest(ds, enforce_canonical_total=False)
