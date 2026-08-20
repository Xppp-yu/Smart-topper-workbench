"""Deterministic Full fold-manifest + seed-derivation tests (no torch)."""

from __future__ import annotations

import pytest

from topper_perception.neural.full_protocol import N_SPLITS, OUTER_SEEDS
from topper_perception.neural.full_splits import (
    _canonical_sha256,
    build_full_fold_manifest,
    derive_inner_seed,
    inner_validation_fold,
    outer_seed_for_repeat,
    validate_full_fold_manifest,
)

#: 60 subjects mirror the frozen full-cohort boundary.
SUBJECTS = [str(i) for i in range(1, 61)]


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------


def test_outer_seed_for_repeat_matches_frozen_seeds() -> None:
    assert outer_seed_for_repeat(0) == 11
    assert outer_seed_for_repeat(1) == 22
    assert outer_seed_for_repeat(2) == 33


@pytest.mark.parametrize("bad", [-1, 3, True, 0.0])
def test_outer_seed_for_repeat_rejects_invalid(bad) -> None:
    with pytest.raises(ValueError):
        outer_seed_for_repeat(bad)


def test_derive_inner_seed_formula() -> None:
    assert derive_inner_seed(11, 0) == 1_000_000 + 11 * 100 + 0
    assert derive_inner_seed(11, 1) == 1_000_000 + 11 * 100 + 1
    assert derive_inner_seed(22, 0) == 1_000_000 + 22 * 100 + 0
    assert derive_inner_seed(33, 4) == 1_000_000 + 33 * 100 + 4


def test_derive_inner_seed_is_deterministic_no_process_random() -> None:
    first = derive_inner_seed(11, 3)
    for _ in range(5):
        assert derive_inner_seed(11, 3) == first


def test_inner_validation_fold_rule() -> None:
    for local_fold in range(12):
        assert inner_validation_fold(local_fold) == local_fold % 4


# ---------------------------------------------------------------------------
# Fold manifest construction + isolation
# ---------------------------------------------------------------------------


def test_manifest_shape_and_validate() -> None:
    manifest = build_full_fold_manifest(SUBJECTS)
    assert manifest["n_subjects"] == 60
    assert manifest["n_repeats"] == len(OUTER_SEEDS) == 3
    assert manifest["n_splits"] == N_SPLITS == 5
    assert manifest["outer_seeds"] == [11, 22, 33]
    assert len(manifest["folds"]) == 3 * 5
    validate_full_fold_manifest(manifest, SUBJECTS)  # must not raise


def test_manifest_sha_is_deterministic() -> None:
    a = build_full_fold_manifest(SUBJECTS)
    b = build_full_fold_manifest(SUBJECTS)
    assert a["sha256"] == b["sha256"]


def test_manifest_outer_test_partitions_subjects_per_repeat() -> None:
    manifest = build_full_fold_manifest(SUBJECTS)
    all_subjects = set(SUBJECTS)
    for repeat in range(3):
        folds = [f for f in manifest["folds"] if f["repeat"] == repeat]
        assert len(folds) == 5
        covered: set[str] = set()
        for fold in folds:
            test = set(fold["outer_test_subjects"])
            assert len(test) == 12  # 60 / 5
            assert not (covered & test)
            covered |= test
        assert covered == all_subjects


def test_manifest_isolation_invariants() -> None:
    manifest = build_full_fold_manifest(SUBJECTS)
    all_subjects = set(SUBJECTS)
    for fold in manifest["folds"]:
        outer_train = set(fold["outer_train_subjects"])
        outer_test = set(fold["outer_test_subjects"])
        inner_train = set(fold["inner_train_subjects"])
        inner_validation = set(fold["inner_validation_subjects"])
        assert outer_train.isdisjoint(outer_test)
        assert outer_train | outer_test == all_subjects
        assert (inner_train | inner_validation) <= outer_train
        assert inner_train.isdisjoint(inner_validation)
        assert inner_train | inner_validation == outer_train


def test_manifest_records_inner_seed_from_formula() -> None:
    manifest = build_full_fold_manifest(SUBJECTS)
    for fold in manifest["folds"]:
        expected = derive_inner_seed(fold["outer_seed"], fold["local_fold"])
        assert fold["inner_seed"] == expected
        assert fold["inner_validation_fold"] == inner_validation_fold(fold["local_fold"])


# ---------------------------------------------------------------------------
# Fail-closed validation
# ---------------------------------------------------------------------------


def test_manifest_tamper_breaks_sha() -> None:
    manifest = build_full_fold_manifest(SUBJECTS)
    manifest["folds"][0]["outer_test_subjects"] = manifest["folds"][0][
        "outer_test_subjects"
    ][:-1] + ["999"]
    with pytest.raises(ValueError, match="SHA-256"):
        validate_full_fold_manifest(manifest, SUBJECTS)


def _rehash(manifest: dict) -> dict:
    manifest["sha256"] = _canonical_sha256(
        {key: value for key, value in manifest.items() if key != "sha256"}
    )
    return manifest


def test_manifest_rejects_outer_train_test_overlap() -> None:
    manifest = build_full_fold_manifest(SUBJECTS)
    fold = manifest["folds"][0]
    escapee = fold["outer_test_subjects"][0]
    fold["outer_train_subjects"] = sorted(set(fold["outer_train_subjects"]) | {escapee})
    with pytest.raises(ValueError, match="overlap"):
        validate_full_fold_manifest(_rehash(manifest), SUBJECTS)


def test_manifest_rejects_inner_escape_outside_outer_train() -> None:
    manifest = build_full_fold_manifest(SUBJECTS)
    fold = manifest["folds"][0]
    escapee = fold["outer_test_subjects"][0]
    fold["inner_train_subjects"] = sorted(set(fold["inner_train_subjects"]) | {escapee})
    with pytest.raises(ValueError, match="escape"):
        validate_full_fold_manifest(_rehash(manifest), SUBJECTS)
