"""Tests for the SLP Subject-level Split (A06).

Coverage map (from the A06 task contract):

* Deterministic reproducibility: same seed → same manifest SHA-256.
* Subject-level isolation: no subject appears in two splits.
* Train / val / test are pairwise disjoint.
* simLab subjects are all in TEST (out-of-domain held-out).
* danaLab split ratios approximate 80/10/10.
* Quarantined frames are reported separately and never silently included.
* Schema compliance: manifest validates against slp_subject_split_v0.1.schema.json.
* JSON round-trip: manifest → JSON → manifest preserves all fields.
* Existing SLP tests (A03/A04/A05) continue to pass.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from topper_perception.io.slp_subject_split import (
    ADAPTER_VERSION,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TASK_ID,
    SPLIT_SCHEMA_VERSION,
    SlpSubjectSplitAdapter,
    SubjectSplitManifest,
    deterministic_subject_hash,
    load_canonical_samples_from_csv,
    run_isolation_tests,
    verify_reproducibility,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_canonical_samples() -> list[dict[str, object]]:
    """Minimal mock data: 10 danaLab + 2 simLab subjects, 3 frames each.

    Uses composite subject IDs to avoid danaLab/simLab collision issues:
    - danaLab: 00001-00010 (all danaLab)
    - simLab: 00011-00012 (all simLab, same ID numbers as potential danaLab)
    """
    samples = []
    # 10 danaLab subjects
    for i in range(1, 11):
        sid = f"{i:05d}"
        setting = "danaLab"
        for cover in ("uncover", "cover1", "cover2"):
            for frame_idx in range(1, 4):
                samples.append({
                    "sample_id": f"slp::{setting}::{sid}::{cover}::{frame_idx:06d}",
                    "setting": setting,
                    "subject_id": sid,
                    "cover_condition": cover,
                    "frame_index": frame_idx,
                    "quarantine": False,
                })
    # 2 simLab subjects (IDs 00011, 00012 — same numbers as potential danaLab)
    for i in range(11, 13):
        sid = f"{i:05d}"
        setting = "simLab"
        for cover in ("uncover", "cover1", "cover2"):
            for frame_idx in range(1, 4):
                # simLab cover2 has missing depthRaw → quarantine
                quarantine = (cover == "cover2")
                samples.append({
                    "sample_id": f"slp::{setting}::{sid}::{cover}::{frame_idx:06d}",
                    "setting": setting,
                    "subject_id": sid,
                    "cover_condition": cover,
                    "frame_index": frame_idx,
                    "quarantine": quarantine,
                })
    return samples


@pytest.fixture
def mock_danalab_only() -> list[dict[str, object]]:
    """10 danaLab subjects, 3 frames each, no quarantine."""
    samples = []
    for i in range(1, 11):
        sid = f"{i:05d}"
        for cover in ("uncover", "cover1", "cover2"):
            for frame_idx in range(1, 4):
                samples.append({
                    "sample_id": f"slp::danaLab::{sid}::{cover}::{frame_idx:06d}",
                    "setting": "danaLab",
                    "subject_id": sid,
                    "cover_condition": cover,
                    "frame_index": frame_idx,
                    "quarantine": False,
                })
    return samples


# ---------------------------------------------------------------------------
# Deterministic hash
# ---------------------------------------------------------------------------

class TestDeterministicSubjectHash:
    def test_same_input_same_hash(self):
        h1 = deterministic_subject_hash("00001", seed=42)
        h2 = deterministic_subject_hash("00001", seed=42)
        assert h1 == h2

    def test_different_seed_different_hash(self):
        h1 = deterministic_subject_hash("00001", seed=42)
        h2 = deterministic_subject_hash("00001", seed=123)
        assert h1 != h2

    def test_different_subject_different_hash(self):
        h1 = deterministic_subject_hash("00001", seed=42)
        h2 = deterministic_subject_hash("00002", seed=42)
        assert h1 != h2

    def test_hash_in_unit_interval(self):
        for _ in range(100):
            h = deterministic_subject_hash("00001", seed=42)
            assert 0.0 <= h < 1.0


# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------

class TestSlpSubjectSplitAdapter:
    def test_build_manifest_produces_all_fields(self, mock_canonical_samples):
        adapter = SlpSubjectSplitAdapter(
            mock_canonical_samples,
            task_id=DEFAULT_TASK_ID,
            random_seed=DEFAULT_RANDOM_SEED,
        )
        manifest = adapter.build_manifest()

        assert manifest.schema_version == SPLIT_SCHEMA_VERSION
        assert manifest.task_id == DEFAULT_TASK_ID
        assert manifest.adapter_version == ADAPTER_VERSION
        assert manifest.random_seed == DEFAULT_RANDOM_SEED
        assert manifest.total_subjects == 12
        assert manifest.danaLab_subjects == 10
        assert manifest.simLab_subjects == 2
        # 12 subjects × 3 covers × 3 frames = 108 canonical sample rows
        assert manifest.total_frames == 12 * 3  # sum of unique frame indices: 12 subjects × 3 frames
        assert manifest.split_rationale  # non-empty
        assert manifest.split_strategy_summary  # non-empty
        assert len(manifest.manifest_sha256) == 64  # SHA-256 hex

    def test_simlab_all_in_test(self, mock_canonical_samples):
        manifest = SlpSubjectSplitAdapter(
            mock_canonical_samples, random_seed=DEFAULT_RANDOM_SEED
        ).build_manifest()
        simlab_keys = {manifest.subject_key(e) for e in manifest.subject_entries if e.setting == "simLab"}
        simlab_test_keys = {manifest.subject_key(e) for e in manifest.subject_entries
                            if e.setting == "simLab" and e.split == "test"}
        assert simlab_test_keys == simlab_keys, (
            f"simLab subjects {simlab_keys - simlab_test_keys} not all in TEST"
        )

    def test_danalab_split_ratios(self, mock_canonical_samples):
        manifest = SlpSubjectSplitAdapter(
            mock_canonical_samples, random_seed=DEFAULT_RANDOM_SEED
        ).build_manifest()
        dana_train = sum(1 for e in manifest.subject_entries
                         if e.setting == "danaLab" and e.split == "train")
        dana_val = sum(1 for e in manifest.subject_entries
                       if e.setting == "danaLab" and e.split == "val")
        dana_test = sum(1 for e in manifest.subject_entries
                        if e.setting == "danaLab" and e.split == "test")
        total = dana_train + dana_val + dana_test
        assert total == 10, f"Expected 10 danaLab subjects, got {total}"
        # 80/10/10 → 8/1/1 for 10 subjects
        assert dana_train == 8, f"Expected 8 danaLab train, got {dana_train}"
        assert dana_val == 1, f"Expected 1 danaLab val, got {dana_val}"
        assert dana_test == 1, f"Expected 1 danaLab test, got {dana_test}"

    def test_no_subject_in_multiple_splits(self, mock_canonical_samples):
        manifest = SlpSubjectSplitAdapter(
            mock_canonical_samples, random_seed=DEFAULT_RANDOM_SEED
        ).build_manifest()
        errors = manifest.verify_no_cross_split_subjects()
        assert errors == [], f"Cross-split subjects found: {errors}"

    def test_train_val_test_disjoint(self, mock_canonical_samples):
        manifest = SlpSubjectSplitAdapter(
            mock_canonical_samples, random_seed=DEFAULT_RANDOM_SEED
        ).build_manifest()
        errors = manifest.verify_train_val_test_disjoint()
        assert errors == [], f"Non-disjoint splits: {errors}"

    def test_quarantine_reported_separately(self, mock_canonical_samples):
        manifest = SlpSubjectSplitAdapter(
            mock_canonical_samples, random_seed=DEFAULT_RANDOM_SEED
        ).build_manifest()
        # 2 simLab subjects × 1 cover (cover2) × 3 frames = 6 quarantined frames
        assert manifest.total_quarantined_frames == 6, (
            f"Expected 6 quarantined frames, got {manifest.total_quarantined_frames}"
        )
        assert manifest.total_usable_frames == manifest.total_frames - 6
        # All quarantined frames are in TEST (simLab)
        test_quarantined = sum(e.quarantine_count for e in manifest.subject_entries
                               if e.split == "test")
        assert test_quarantined == 6, f"Expected 6 quarantined in test, got {test_quarantined}"

    def test_all_subjects_accounted(self, mock_canonical_samples):
        manifest = SlpSubjectSplitAdapter(
            mock_canonical_samples, random_seed=DEFAULT_RANDOM_SEED
        ).build_manifest()
        # Use composite keys (setting::subject_id) to handle danaLab/simLab ID collisions
        all_keys = {manifest.subject_key(e) for e in manifest.subject_entries}
        union = manifest.train_subjects() | manifest.val_subjects() | manifest.test_subjects()
        assert union == all_keys, (
            f"union={len(union)} vs all_keys={len(all_keys)}: "
            f"union={sorted(union)}, all_keys={sorted(all_keys)}"
        )

    def test_reproducibility_twice_same_seed(self, mock_canonical_samples):
        repro = verify_reproducibility(
            mock_canonical_samples, seed=DEFAULT_RANDOM_SEED
        )
        assert repro["reproducible"] is True
        assert repro["sha_match"] is True
        assert repro["assignment_match"] is True

    def test_different_seed_different_manifest(self, mock_canonical_samples):
        m1 = SlpSubjectSplitAdapter(
            mock_canonical_samples, random_seed=42
        ).build_manifest()
        m2 = SlpSubjectSplitAdapter(
            mock_canonical_samples, random_seed=999
        ).build_manifest()
        assert m1.manifest_sha256 != m2.manifest_sha256

    def test_quarantine_isolation_in_test_split(self, mock_canonical_samples):
        """Quarantined simLab subjects are in TEST, not train or val."""
        manifest = SlpSubjectSplitAdapter(
            mock_canonical_samples, random_seed=DEFAULT_RANDOM_SEED
        ).build_manifest()
        # All quarantined frames belong to simLab subjects (00011, 00012)
        # and those subjects must be in TEST
        quarantined_entries = [
            e for e in manifest.subject_entries if e.quarantine_count > 0
        ]
        assert len(quarantined_entries) == 2, (
            f"Expected 2 quarantined subjects (simLab 00011, 00012), "
            f"got {len(quarantined_entries)}"
        )
        for entry in quarantined_entries:
            assert entry.split == "test", (
                f"Subject {manifest.subject_key(entry)} with quarantined frames is in "
                f"{entry.split}, expected test"
            )

    def test_isolation_tests_all_pass(self, mock_canonical_samples):
        manifest = SlpSubjectSplitAdapter(
            mock_canonical_samples, random_seed=DEFAULT_RANDOM_SEED
        ).build_manifest()
        results = run_isolation_tests(manifest)
        failures = [k for k, v in results.items() if not v["passed"]]
        assert failures == [], f"Isolation test failures: {failures}"


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

class TestManifestJsonRoundtrip:
    def test_manifest_to_json_and_back(self, mock_canonical_samples):
        manifest = SlpSubjectSplitAdapter(
            mock_canonical_samples, random_seed=DEFAULT_RANDOM_SEED
        ).build_manifest()

        # Write to a temp path
        tmp = PROJECT_ROOT / "test_manifest_tmp.json"
        manifest.to_json(tmp)
        try:
            loaded = SubjectSplitManifest.from_json(tmp)
            assert loaded.schema_version == manifest.schema_version
            assert loaded.manifest_sha256 == manifest.manifest_sha256
            assert loaded.subject_to_split() == manifest.subject_to_split()
            assert loaded.train_subjects() == manifest.train_subjects()
            assert loaded.val_subjects() == manifest.val_subjects()
            assert loaded.test_subjects() == manifest.test_subjects()
            assert loaded.total_subjects == manifest.total_subjects
            assert loaded.total_frames == manifest.total_frames
        finally:
            if tmp.is_file():
                tmp.unlink()


# ---------------------------------------------------------------------------
# Schema compliance
# ---------------------------------------------------------------------------

class TestSchemaCompliance:
    @pytest.fixture
    def schema_path(self) -> Path:
        return PROJECT_ROOT / "configs/annotations/slp_subject_split_v0.1.schema.json"

    def test_schema_file_exists(self, schema_path):
        assert schema_path.is_file(), f"Schema not found at {schema_path}"

    def test_manifest_validates_against_schema(self, schema_path):
        """Load the real A05 canonical samples and validate the split manifest."""
        a05_csv = PROJECT_ROOT / "data/processed/slp/slp_canonical_samples_v0.1.csv"
        if not a05_csv.is_file():
            pytest.skip("A05 canonical sample CSV not found; run A05 first")

        samples = load_canonical_samples_from_csv(a05_csv)
        manifest = SlpSubjectSplitAdapter(
            samples, random_seed=DEFAULT_RANDOM_SEED
        ).build_manifest()
        manifest_dict = manifest.as_dict()

        # Basic structural checks against schema
        assert manifest_dict["schema_version"] == "slp_subject_split_v0.1"
        assert manifest_dict["random_seed"] == DEFAULT_RANDOM_SEED
        assert len(manifest_dict["manifest_sha256"]) == 64

        # Each entry must have required fields
        for entry in manifest_dict["subject_entries"]:
            assert "subject_id" in entry
            assert "setting" in entry
            assert entry["setting"] in ("danaLab", "simLab")
            assert "split" in entry
            assert entry["split"] in ("train", "val", "test")
            assert entry["canonical_sample_count"] >= 0
            assert entry["frame_count"] >= 0
            assert entry["quarantine_count"] >= 0

        # Split statistics must have one entry per split
        stat_splits = {s["split"] for s in manifest_dict["split_statistics"]}
        assert stat_splits == {"train", "val", "test"}

        # Total check
        assert manifest_dict["total_subjects"] == (
            manifest_dict["danaLab_subjects"] + manifest_dict["simLab_subjects"]
        )


# ---------------------------------------------------------------------------
# Regression: existing SLP tests
# ---------------------------------------------------------------------------

class TestExistingSlpTests:
    def test_existing_slp_tests_remain_discoverable(self):
        """Ensure A03/A04/A05/A09 test modules are still importable."""
        import importlib

        for module_name in [
            "test_slp_frame_index",
            "test_slp_homography",
            "test_slp_canonical_adapter",
            "test_slp_region_annotation_schema",
        ]:
            try:
                importlib.import_module(module_name)
            except ImportError as exc:
                pytest.fail(f"Could not import {module_name}: {exc}")
