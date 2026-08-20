"""Deterministic fold construction for the frozen P5.2-C Full protocol.

This module builds the auditable split manifest — outer subject-grouped folds
(reusing P5.1) plus the inner subject-grouped epoch-selection folds — from a
sorted list of subject IDs. It is pure NumPy + stdlib (no ``torch``), and it
**never reads PoPu data**: callers pass the subject IDs, this module only turns
them into reproducible folds and validates isolation.

Seed derivation is deterministic arithmetic (no Python ``hash()``, no
process-global random state), so the manifest reproduces identically on any
machine and can be audited line by line:

- ``outer_seed(repeat)      = OUTER_SEEDS[repeat]``
- ``inner_seed(repeat, fold) = 1_000_000 + outer_seed(repeat) * 100 + local_fold``
- ``inner_validation_fold(local_fold) = local_fold % INNER_N_SPLITS``
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from topper_perception.evaluation.grouped import generate_group_folds
from topper_perception.neural.full_protocol import (
    GROUP_KEY,
    INNER_N_SPLITS,
    N_SPLITS,
    OUTER_SEEDS,
)

#: Base and stride for the inner seed. Kept as named constants so the formula is
#: explicit and auditable rather than magic.
INNER_SEED_BASE = 1_000_000
INNER_SEED_STRIDE = 100


def outer_seed_for_repeat(repeat: int, outer_seeds: Sequence[int] = OUTER_SEEDS) -> int:
    """Return the outer seed for ``repeat`` (0-based index into ``outer_seeds``)."""
    if not isinstance(repeat, int) or isinstance(repeat, bool):
        raise ValueError("repeat must be an integer.")
    if not 0 <= repeat < len(outer_seeds):
        raise ValueError(f"repeat {repeat} out of range [0, {len(outer_seeds)}).")
    return int(outer_seeds[repeat])


def derive_inner_seed(
    outer_seed: int,
    local_fold: int,
    *,
    inner_seed_base: int = INNER_SEED_BASE,
    inner_seed_stride: int = INNER_SEED_STRIDE,
) -> int:
    """Derive the inner fold-set seed from the outer seed and local fold.

    Deterministic arithmetic only — no ``hash()``, no process-global RNG. The
    outer seed and local fold are both plain integers, so the mapping is
    transparent and stable.
    """
    if not isinstance(outer_seed, int) or isinstance(outer_seed, bool):
        raise ValueError("outer_seed must be an integer.")
    if not isinstance(local_fold, int) or isinstance(local_fold, bool):
        raise ValueError("local_fold must be an integer.")
    if outer_seed < 0 or local_fold < 0:
        raise ValueError("outer_seed and local_fold must be non-negative.")
    return inner_seed_base + outer_seed * inner_seed_stride + local_fold


def inner_validation_fold(local_fold: int, inner_n_splits: int = INNER_N_SPLITS) -> int:
    """Return the inner validation fold index for a given outer local fold."""
    if not isinstance(local_fold, int) or isinstance(local_fold, bool):
        raise ValueError("local_fold must be an integer.")
    if local_fold < 0:
        raise ValueError("local_fold must be non-negative.")
    return local_fold % inner_n_splits


def _canonical_sha256(data: Any) -> str:
    """Canonical-JSON SHA-256 of ``data`` (sorted keys, compact separators)."""
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _subjects_in(order: Sequence[str], indices: Sequence[int]) -> list[str]:
    """Return the subject IDs at ``indices`` in a stable, sorted manner."""
    return sorted(str(order[i]) for i in indices)


def build_full_fold_manifest(
    subject_ids: Sequence[str],
    *,
    n_splits: int = N_SPLITS,
    outer_seeds: Sequence[int] = OUTER_SEEDS,
    inner_n_splits: int = INNER_N_SPLITS,
) -> dict[str, Any]:
    """Build the full auditable split manifest for the frozen Full protocol.

    For every repeat and local outer fold: the outer train/test subjects, the
    deterministic inner seed, the inner validation-fold index, and the inner
    train/validation subjects. The returned mapping includes a
    ``sha256`` field (canonical-JSON digest over the content *excluding* the
    digest itself), so a governed run can pin and later re-verify the exact
    split schedule.
    """
    subjects = sorted({str(s) for s in subject_ids})
    if len(subjects) < n_splits:
        raise ValueError(
            f"Need at least n_splits={n_splits} subjects, got {len(subjects)}."
        )

    folds: list[dict[str, Any]] = []
    for repeat, outer_seed in enumerate(outer_seeds):
        outer = generate_group_folds(
            subjects, n_splits=n_splits, shuffle=True, seed=int(outer_seed)
        )
        for local_fold, (train_idx, val_idx) in enumerate(outer.folds):
            outer_train = _subjects_in(subjects, train_idx)
            outer_test = _subjects_in(subjects, val_idx)

            inner_seed = derive_inner_seed(int(outer_seed), local_fold)
            validation_fold = inner_validation_fold(local_fold, inner_n_splits)
            inner = generate_group_folds(
                outer_train,
                n_splits=inner_n_splits,
                shuffle=True,
                seed=inner_seed,
            )
            inner_train_idx, inner_val_idx = inner.folds[validation_fold]
            inner_train = _subjects_in(outer_train, inner_train_idx)
            inner_validation = _subjects_in(outer_train, inner_val_idx)

            folds.append(
                {
                    "repeat": repeat,
                    "outer_seed": int(outer_seed),
                    "local_fold": local_fold,
                    "outer_train_subjects": outer_train,
                    "outer_test_subjects": outer_test,
                    "inner_seed": inner_seed,
                    "inner_validation_fold": validation_fold,
                    "inner_train_subjects": inner_train,
                    "inner_validation_subjects": inner_validation,
                }
            )

    manifest: dict[str, Any] = {
        "protocol": "popu_neural_full_v0.1",
        "group_key": GROUP_KEY,
        "n_subjects": len(subjects),
        "n_splits": n_splits,
        "n_repeats": len(outer_seeds),
        "outer_seeds": [int(seed) for seed in outer_seeds],
        "inner_n_splits": inner_n_splits,
        "seed_derivation": {
            "outer_seed": "OUTER_SEEDS[repeat]",
            "inner_seed": f"{INNER_SEED_BASE} + outer_seed * {INNER_SEED_STRIDE} + local_fold",
            "inner_validation_fold": f"local_fold % {inner_n_splits}",
            "note": "Deterministic arithmetic; no Python hash(), no process-random.",
        },
        "folds": folds,
    }
    manifest["sha256"] = _canonical_sha256(manifest)
    return manifest


def validate_full_fold_manifest(
    manifest: Mapping[str, Any],
    subject_ids: Sequence[str],
) -> None:
    """Fail closed when a full fold manifest violates isolation or determinism.

    Checks: the SHA-256 recomputes identically; every expected (repeat, fold)
    is present; outer train/test partition all subjects disjointly; inner
    subjects are a subset of outer train; and inner train/validation partition
    the outer-train subjects disjointly. Any violation raises :class:`ValueError`.
    """
    if not isinstance(manifest, Mapping):
        raise ValueError("Fold manifest must be a mapping.")

    subjects = sorted({str(s) for s in subject_ids})
    all_subjects = set(subjects)

    expected_folds = len(OUTER_SEEDS) * N_SPLITS
    folds = manifest.get("folds")
    if not isinstance(folds, list) or len(folds) != expected_folds:
        raise ValueError(
            f"Fold manifest must contain {expected_folds} folds; "
            f"got {len(folds) if isinstance(folds, list) else 'not-a-list'}."
        )

    # Determinism: the stored digest must match a recomputation over content.
    content = dict(manifest)
    stored_sha = content.pop("sha256", None)
    if stored_sha != _canonical_sha256(content):
        raise ValueError("Fold manifest SHA-256 does not match its content.")

    # Isolation + coverage.
    per_repeat_test: dict[int, set[str]] = {}
    for fold in folds:
        repeat = int(fold["repeat"])
        outer_train = set(fold["outer_train_subjects"])
        outer_test = set(fold["outer_test_subjects"])
        inner_train = set(fold["inner_train_subjects"])
        inner_validation = set(fold["inner_validation_subjects"])

        if outer_train & outer_test:
            raise ValueError(f"Outer train/test overlap in repeat {repeat}.")
        if outer_train | outer_test != all_subjects:
            raise ValueError(
                f"Outer train+test does not cover all subjects in repeat {repeat}."
            )
        if len(outer_train) + len(outer_test) != len(all_subjects):
            raise ValueError(f"Outer fold not disjoint in repeat {repeat}.")

        if not (inner_train | inner_validation).issubset(outer_train):
            raise ValueError(
                f"Inner subjects escape outer train in repeat {repeat}."
            )
        if inner_train & inner_validation:
            raise ValueError(f"Inner train/validation overlap in repeat {repeat}.")
        if inner_train | inner_validation != outer_train:
            raise ValueError(
                f"Inner train+validation != outer train in repeat {repeat}."
            )

        per_repeat_test.setdefault(repeat, set())
        if per_repeat_test[repeat] & outer_test:
            raise ValueError(f"Outer test subjects repeat across folds in repeat {repeat}.")
        per_repeat_test[repeat] |= outer_test

    # Every subject validated exactly once per repeat (out-of-fold).
    for repeat, covered in per_repeat_test.items():
        if covered != all_subjects:
            raise ValueError(
                f"Outer test folds do not cover every subject exactly once "
                f"in repeat {repeat}."
            )
