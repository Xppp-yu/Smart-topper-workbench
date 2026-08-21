"""Tests for the PoPu P7 Full independent analysis module.

The Full P7 evidence pack (``EXP-P7-FULL-20260820-R02.tar.gz``) is the artifact
under independent re-verification by the Reviewer. These tests pin the
contract of ``topper_perception.neural.p7_full_analysis``:

- integrity verification (archive SHA-256, file structure, strict-JSON parsing);
- stitching-then-metric (the forbidden per-fold-then-mean path must NEVER
  collapse to a different answer than pooling first);
- per-(condition, seed) and per-condition aggregation rules (5 seeds must be
  reported as mean / std / worst, never merged into more independent samples);
- P6 single-checkpoint threshold 0.94 and P6.1 ensemble T=0.75 / thr=0.5 /
  require_unanimous=true must be applied to the *stitched* 15-fold OOF;
- the four worst-subject criteria, high-confidence error filtering, and the
  full artifact-writing contract (``summary.json``,
  ``condition_metrics.csv``, ``per_class_metrics.csv``,
  ``per_subject_metrics.csv``, ``worst_subjects.csv``, ``error_cases.csv``,
  ``high_confidence_errors.csv``, ``evidence_manifest.json``).

These tests use small synthetic fixtures so they run on CPU-only CI without
loading the full evidence pack. A single integration test points at the real
archive via ``--run-full-integration`` and is opt-in.
"""

from __future__ import annotations

import csv
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pytest

from topper_perception.neural.data import FROZEN_LABELS, LABEL_TO_INDEX
from topper_perception.neural.p7_full_analysis import (
    ArchiveIntegrityError,
    EvidenceManifest,
    StitchingError,
    analyze_p7_full,
    assert_rule_values_match_frozen,
    compute_condition_summary,
    compute_high_confidence_errors,
    compute_p6_1_ensemble,
    compute_p6_single,
    compute_per_class_breakdown,
    compute_per_subject_breakdown,
    compute_seed_summary,
    compute_stitched_classification_metrics,
    compute_worst_subjects,
    load_clean_oof,
    load_condition_seed_oof,
    load_p6_1_ensemble_rule_from_archive,
    load_p6_single_rule_from_archive,
    strict_json_load,
    verify_evidence_archive,
)
from topper_perception.neural.p6_reject import PROBA_COLUMNS

# Frozen constants the analysis module MUST honor regardless of upstream drift.
FROZEN_P6_SINGLE_THRESHOLD = 0.94
FROZEN_P6_1_TEMPERATURE = 0.75
FROZEN_P6_1_THRESHOLD = 0.5
FROZEN_P6_1_REQUIRE_UNANIMOUS = True
FROZEN_HIGH_CONFIDENCE_THRESHOLD = 0.90
FROZEN_CONDITION_NAMES = (
    "density_stride_2_2",
    "density_stride_4_4",
    "noise_p95_0.01",
    "noise_p95_0.05",
    "noise_p95_0.10",
    "bad_cell_0.01",
    "bad_cell_0.05",
    "bad_cell_0.10",
    "bad_rows_1",
    "bad_rows_2",
    "bad_rows_4",
    "bad_columns_1",
    "bad_columns_2",
    "bad_columns_4",
)
FROZEN_SEEDS = (701, 702, 703, 704, 705)


# ---------------------------------------------------------------------------
# Helpers — small synthetic OOF CSV builders
# ---------------------------------------------------------------------------


def _make_record_row(
    record_id: str,
    subject_id: str,
    y_true: str,
    y_pred: str,
    *,
    repeat: int,
    local_fold: int,
    confidence: float = 0.95,
    proba: dict[str, float] | None = None,
) -> dict[str, Any]:
    """One synthetic record OOF row matching the schema on disk."""
    row = {
        "model": "small_resnet",
        "repeat": int(repeat),
        "outer_seed": 11,
        "local_fold": int(local_fold),
        "record_id": record_id,
        "subject_id": subject_id,
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": float(confidence),
        "n_snapshots": 10,
    }
    for label in FROZEN_LABELS:
        row[f"proba__{label}"] = 0.0
    if proba is not None:
        for label, value in proba.items():
            row[f"proba__{label}"] = float(value)
    else:
        # Default: spike on y_pred, low confidence elsewhere
        row[f"proba__{y_pred}"] = float(confidence)
        # Distribute the remaining mass across non-y_pred labels
        remaining = 1.0 - float(confidence)
        other_labels = [label for label in FROZEN_LABELS if label != y_pred]
        share = remaining / max(1, len(other_labels))
        for label in other_labels:
            row[f"proba__{label}"] = float(share)
    return row


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _build_full_oof_fixture(
    root: Path,
    *,
    repeats: tuple[int, ...] = (0, 1, 2),
    local_folds: tuple[int, ...] = (0, 1, 2, 3, 4),
    conditions: tuple[str, ...] = FROZEN_CONDITION_NAMES,
    seeds: tuple[int, ...] = FROZEN_SEEDS,
    records_per_fold: int = 6,
    subjects_per_fold: int = 2,
    # Per-condition accuracy override (label index → accuracy 0..1)
    condition_accuracy: dict[str, float] | None = None,
    seed_accuracy_jitter: dict[int, float] | None = None,
    include_condition_comparison: bool = True,
) -> None:
    """Build a small synthetic Full P7 evidence pack under ``root``.

    The structure mirrors the real evidence pack: ``folds/repeat_{r}/fold_{f}``
    each contain a ``clean/`` block plus one ``<condition>/seed_{s}/`` block
    per (condition, seed). A top-level ``condition_comparison.json``,
    ``config_used.json``, ``scope.json`` are also emitted.
    """
    condition_accuracy = dict(condition_accuracy or {})
    seed_accuracy_jitter = dict(seed_accuracy_jitter or {})
    subjects = [f"S{idx}" for idx in range(subjects_per_fold)]
    labels = list(FROZEN_LABELS)

    for repeat in repeats:
        for local_fold in local_folds:
            base = root / "folds" / f"repeat_{repeat}" / f"fold_{local_fold}"
            clean_dir = base / "clean"
            clean_records: list[dict[str, Any]] = []
            for idx in range(records_per_fold):
                true_label = labels[idx % len(labels)]
                clean_records.append(
                    _make_record_row(
                        record_id=f"R-clean-r{repeat}f{local_fold}-{idx:03d}",
                        subject_id=subjects[idx % subjects_per_fold],
                        y_true=true_label,
                        y_pred=true_label,
                        repeat=repeat,
                        local_fold=local_fold,
                        confidence=0.99,
                    )
                )
            _write_csv(clean_dir / "record_predictions.csv", clean_records)
            snapshot_count = records_per_fold * 10
            _write_csv(
                clean_dir / "snapshot_predictions.csv",
                _make_snapshot_rows(clean_records, snapshot_count),
            )
            (clean_dir / "summary.json").write_text(
                json.dumps({"repeat": repeat, "local_fold": local_fold, "n_records": records_per_fold}),
                encoding="utf-8",
            )
            for condition in conditions:
                base_accuracy = condition_accuracy.get(condition, 1.0)
                for seed in seeds:
                    jitter = seed_accuracy_jitter.get(int(seed), 0.0)
                    accuracy = max(0.0, min(1.0, base_accuracy + jitter))
                    seed_dir = base / condition / f"seed_{seed}"
                    seed_records: list[dict[str, Any]] = []
                    for idx in range(records_per_fold):
                        true_label = labels[idx % len(labels)]
                        pred_label = true_label if (idx / max(1, records_per_fold - 1)) < accuracy else labels[(idx + 1) % len(labels)]
                        seed_records.append(
                            _make_record_row(
                                record_id=f"R-{condition}-r{repeat}f{local_fold}-s{seed}-{idx:03d}",
                                subject_id=subjects[idx % subjects_per_fold],
                                y_true=true_label,
                                y_pred=pred_label,
                                repeat=repeat,
                                local_fold=local_fold,
                                confidence=0.95,
                            )
                        )
                    _write_csv(seed_dir / "record_predictions.csv", seed_records)
                    _write_csv(
                        seed_dir / "snapshot_predictions.csv",
                        _make_snapshot_rows(seed_records, snapshot_count),
                    )
    if include_condition_comparison:
        # A minimal condition_comparison.json so any code that strict-parses
        # the top-level summary keeps its contract even when reading the fixture.
        # Round 2: must carry the pinned P6 / P6.1 rule blocks with the
        # module-frozen SHA pair so the fail-closed rule loader passes.
        from topper_perception.neural.p7_full_analysis import (
            FROZEN_P6_1_ACTUAL_SHA256,
            FROZEN_P6_1_EXPECTED_SHA256,
            FROZEN_P6_1_RULE_POINTER,
            FROZEN_P6_1_SOURCE_PATH,
            FROZEN_P6_1_TEMPERATURE_POINTER,
            FROZEN_P6_1_THRESHOLD,
            FROZEN_P6_1_UNANIMITY_FIELD_POINTER,
            FROZEN_P6_1_REQUIRE_UNANIMOUS,
            FROZEN_P6_1_TEMPERATURE,
            FROZEN_P6_ACTUAL_SHA256,
            FROZEN_P6_EXPECTED_SHA256,
            FROZEN_P6_FALLBACK_THRESHOLD_POINTER,
            FROZEN_P6_SINGLE_THRESHOLD,
            FROZEN_P6_SOURCE_PATH,
            FROZEN_P6_THRESHOLD_POINTER,
        )
        (root / "condition_comparison.json").write_text(
            json.dumps({
                "fixture": True,
                "n_folds_resolved": len(repeats) * len(local_folds),
                "clean_n_records_total": records_per_fold * len(repeats) * len(local_folds),
                "condition_summaries": [],
                "p6_single_rule": {
                    "rule_kind": "p6_single",
                    "threshold": FROZEN_P6_SINGLE_THRESHOLD,
                    "source_path": FROZEN_P6_SOURCE_PATH,
                    "source_expected_sha256": FROZEN_P6_EXPECTED_SHA256,
                    "source_actual_sha256": FROZEN_P6_ACTUAL_SHA256,
                    "threshold_pointer": FROZEN_P6_THRESHOLD_POINTER,
                    "fallback_threshold_pointer": FROZEN_P6_FALLBACK_THRESHOLD_POINTER,
                },
                "p6_1_ensemble_rule": {
                    "rule_kind": "p6_1_ensemble",
                    "temperature": FROZEN_P6_1_TEMPERATURE,
                    "threshold": FROZEN_P6_1_THRESHOLD,
                    "require_unanimous": FROZEN_P6_1_REQUIRE_UNANIMOUS,
                    "source_path": FROZEN_P6_1_SOURCE_PATH,
                    "source_expected_sha256": FROZEN_P6_1_EXPECTED_SHA256,
                    "source_actual_sha256": FROZEN_P6_1_ACTUAL_SHA256,
                    "temperature_pointer": FROZEN_P6_1_TEMPERATURE_POINTER,
                    "rule_pointer": FROZEN_P6_1_RULE_POINTER,
                    "unanimity_field_pointer": FROZEN_P6_1_UNANIMITY_FIELD_POINTER,
                },
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        (root / "config_used.json").write_text(
            json.dumps({"fixture": True}, ensure_ascii=False),
            encoding="utf-8",
        )
        (root / "scope.json").write_text(
            json.dumps({"fixture": True}, ensure_ascii=False),
            encoding="utf-8",
        )


def _make_snapshot_rows(record_rows: list[dict[str, Any]], total: int) -> list[dict[str, Any]]:
    """Emit snapshot rows that match the expected schema count.

    Snapshot count is purely structural here — we only need the CSV to read.
    """
    rows: list[dict[str, Any]] = []
    snapshot_index = 0
    for record in record_rows:
        for _ in range(10):
            row = {
                "model": "small_resnet",
                "repeat": int(record["repeat"]),
                "outer_seed": int(record["outer_seed"]),
                "local_fold": int(record["local_fold"]),
                "sample_id": f"{record['record_id']}-s{snapshot_index % 10}",
                "record_id": record["record_id"],
                "subject_id": record["subject_id"],
                "y_true": record["y_true"],
                "y_pred": record["y_pred"],
                "confidence": float(record["confidence"]),
            }
            for label in FROZEN_LABELS:
                row[f"proba__{label}"] = float(record[f"proba__{label}"])
            rows.append(row)
            snapshot_index += 1
            if len(rows) >= total:
                return rows
    # Pad to ``total`` with copies so file counts match if caller expected more.
    while len(rows) < total:
        rows.append(dict(rows[-1]))
    return rows


def _pack_tar_gz(src_root: Path, archive_path: Path) -> str:
    """Tar+gzip ``src_root`` into ``archive_path``; return SHA-256 hex."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as handle:
        handle.add(src_root, arcname=src_root.name)
    digest = hashlib.sha256()
    with archive_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Strict-JSON parsing (NaN / Infinity rejection)
# ---------------------------------------------------------------------------


def test_strict_json_load_accepts_well_formed_object(tmp_path: Path) -> None:
    path = tmp_path / "ok.json"
    _write_json(path, {"a": 1, "b": [1, 2, 3], "c": "x"})
    payload = strict_json_load(path)
    assert payload == {"a": 1, "b": [1, 2, 3], "c": "x"}


def test_strict_json_load_rejects_nan_constant(tmp_path: Path) -> None:
    path = tmp_path / "nan.json"
    path.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        strict_json_load(path)


def test_strict_json_load_rejects_positive_infinity(tmp_path: Path) -> None:
    path = tmp_path / "posinf.json"
    path.write_text('{"value": Infinity}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        strict_json_load(path)


def test_strict_json_load_rejects_negative_infinity(tmp_path: Path) -> None:
    path = tmp_path / "neginf.json"
    path.write_text('{"value": -Infinity}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        strict_json_load(path)


def test_strict_json_load_rejects_non_object_root(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        strict_json_load(path)


# ---------------------------------------------------------------------------
# Archive integrity / structure verification
# ---------------------------------------------------------------------------


def test_verify_evidence_archive_rejects_sha_drift(tmp_path: Path) -> None:
    """If the supplied expected SHA-256 does not match the archive, fail closed."""
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src)
    archive = tmp_path / "pack.tar.gz"
    actual_sha = _pack_tar_gz(src, archive)
    wrong_sha = "0" * 64 if actual_sha != "0" * 64 else "1" * 64
    with pytest.raises(ArchiveIntegrityError):
        verify_evidence_archive(archive, expected_sha256=wrong_sha)


def test_verify_evidence_archive_accepts_matching_sha(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src)
    archive = tmp_path / "pack.tar.gz"
    actual_sha = _pack_tar_gz(src, archive)
    manifest = verify_evidence_archive(archive, expected_sha256=actual_sha)
    assert isinstance(manifest, EvidenceManifest)
    assert manifest.archive_sha256 == actual_sha
    assert manifest.n_folds_resolved == 15
    assert manifest.n_conditions_resolved == len(FROZEN_CONDITION_NAMES)
    assert manifest.n_seeds_resolved == len(FROZEN_SEEDS)
    assert manifest.evidence_root.is_dir()
    # Manifest must enumerate the per-file SHA-256 for every file in the pack
    # (excluding the archive itself) so reviewers can spot any drift.
    assert manifest.file_sha256s
    assert any(name.endswith("condition_comparison.json") for name in manifest.file_sha256s)


def test_verify_evidence_archive_accepts_directory_input(tmp_path: Path) -> None:
    """A pre-extracted directory must also satisfy the contract."""
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src)
    manifest = verify_evidence_archive(src, expected_sha256=None)
    assert manifest.n_folds_resolved == 15
    assert manifest.archive_sha256 is None  # no archive was supplied


def test_verify_evidence_archive_rejects_missing_fold(tmp_path: Path) -> None:
    """If a fold is missing the verifier must fail closed (Reviewer point: completeness)."""
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src)
    # Remove one entire fold directory to simulate a missing fold.
    missing = src / "folds" / "repeat_0" / "fold_0"
    for path in missing.rglob("*"):
        if path.is_file():
            path.unlink()
    for path in sorted(missing.rglob("*"), reverse=True):
        if path.is_dir():
            path.rmdir()
    missing.rmdir()
    with pytest.raises(ArchiveIntegrityError, match="fold"):
        verify_evidence_archive(src, expected_sha256=None)


def test_verify_evidence_archive_rejects_short_seed_set(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src, seeds=(701, 702, 703))  # only 3 seeds, need 5
    with pytest.raises(ArchiveIntegrityError, match="seed"):
        verify_evidence_archive(src, expected_sha256=None)


def test_verify_evidence_archive_rejects_missing_condition(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src, conditions=("noise_p95_0.05",))  # only 1 condition
    with pytest.raises(ArchiveIntegrityError, match="condition"):
        verify_evidence_archive(src, expected_sha256=None)


def test_verify_evidence_archive_rejects_nonfinite_json_constant(tmp_path: Path) -> None:
    """If any JSON in the pack contains a NaN / Infinity constant, fail closed."""
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src)
    (src / "condition_comparison.json").write_text(
        '{"accuracy": NaN}', encoding="utf-8"
    )
    with pytest.raises(ArchiveIntegrityError, match="non-finite"):
        verify_evidence_archive(src, expected_sha256=None)


def test_verify_evidence_archive_records_clean_record_count(tmp_path: Path) -> None:
    """The manifest must report the clean OOF total record count from the pack."""
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src, records_per_fold=10)
    manifest = verify_evidence_archive(src, expected_sha256=None)
    assert manifest.n_clean_records_total == 10 * 15


# ---------------------------------------------------------------------------
# Clean / condition-seed OOF stitching
# ---------------------------------------------------------------------------


def test_load_clean_oof_pools_all_15_folds(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    records_per_fold = 6
    _build_full_oof_fixture(src, records_per_fold=records_per_fold)
    clean = load_clean_oof(src)
    assert isinstance(clean, pd.DataFrame)
    assert len(clean) == records_per_fold * 15
    # Every row must be tagged with the full repeat+fold provenance.
    assert set(clean["repeat"].astype(int).unique()) == {0, 1, 2}
    assert set(clean["local_fold"].astype(int).unique()) == {0, 1, 2, 3, 4}


def test_load_clean_oof_rejects_missing_columns(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src)
    # Drop one required column from a single clean CSV.
    target = src / "folds" / "repeat_0" / "fold_0" / "clean" / "record_predictions.csv"
    frame = pd.read_csv(target)
    frame = frame.drop(columns=["confidence"])
    frame.to_csv(target, index=False)
    with pytest.raises(StitchingError):
        load_clean_oof(src)


def test_load_condition_seed_oof_returns_only_that_condition_seed(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src, records_per_fold=4)
    stitched = load_condition_seed_oof(src, "noise_p95_0.10", seed=703)
    assert len(stitched) == 4 * 15
    # All 15 folds must be represented.
    assert stitched.groupby(["repeat", "local_fold"]).ngroups == 15


def test_load_condition_seed_oof_rejects_unknown_condition(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src)
    with pytest.raises(StitchingError, match="condition"):
        load_condition_seed_oof(src, "unknown_condition", seed=701)


def test_load_condition_seed_oof_rejects_unknown_seed(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src)
    with pytest.raises(StitchingError, match="seed"):
        load_condition_seed_oof(src, "noise_p95_0.10", seed=999)


# ---------------------------------------------------------------------------
# Stitched classification metrics
# ---------------------------------------------------------------------------


def test_compute_stitched_metrics_uses_frozen_label_order(tmp_path: Path) -> None:
    """macro_f1 must average over the 5 frozen classes, not over labels present only."""
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src)
    clean = load_clean_oof(src)
    metrics = compute_stitched_classification_metrics(clean)
    assert metrics["n_samples"] == len(clean)
    # All 5 frozen labels must appear in per_class even if the synthetic
    # fixture did not exercise every label.
    assert {entry["label"] for entry in metrics["per_class"]} == set(FROZEN_LABELS)


def test_compute_stitched_metrics_matches_hand_computed_example() -> None:
    """A small hand-computable example proves the metric pipeline is correct.

    Build a 30-row stitched frame with a known confusion matrix and verify
    the metrics match what we get from ``compute_classification_metrics``
    directly (the canonical metric definition).
    """
    rows: list[dict[str, Any]] = []
    labels = list(FROZEN_LABELS)
    rng = np.random.default_rng(42)
    for repeat in (0, 1, 2):
        for local_fold in (0, 1, 2, 3, 4):
            for idx in range(2):
                true_label = labels[(idx + repeat + local_fold) % len(labels)]
                pred_label = labels[(idx + repeat + local_fold + (1 if rng.random() < 0.05 else 0)) % len(labels)]
                rows.append(
                    _make_record_row(
                        record_id=f"R-r{repeat}f{local_fold}-{idx}",
                        subject_id=f"S{idx}",
                        y_true=true_label,
                        y_pred=pred_label,
                        repeat=repeat,
                        local_fold=local_fold,
                    )
                )
    frame = pd.DataFrame(rows)
    metrics = compute_stitched_classification_metrics(frame)
    # Macro-F1 should equal the value computed independently with the canonical
    # metric definition.
    true = np.asarray([LABEL_TO_INDEX[str(v)] for v in frame["y_true"]], dtype=np.int64)
    pred = np.asarray([LABEL_TO_INDEX[str(v)] for v in frame["y_pred"]], dtype=np.int64)
    from topper_perception.neural.metrics import compute_classification_metrics

    reference = compute_classification_metrics(true, pred, FROZEN_LABELS)
    assert metrics["accuracy"] == pytest.approx(reference.accuracy)
    assert metrics["balanced_accuracy"] == pytest.approx(reference.balanced_accuracy)
    assert metrics["macro_f1"] == pytest.approx(reference.macro_f1)


def test_compute_stitched_metrics_diverges_from_per_fold_average() -> None:
    """A synthetic imbalanced example proves pool-first ≠ mean-of-folds.

    Construct 15 folds whose class distributions are highly imbalanced. Each
    fold contains only ONE class (the other 4 contribute 0 to its macro_f1),
    but the stitched frame contains ALL 5 classes with every record correctly
    classified. Pool-first macro_f1 is then near 1.0 while the per-fold mean
    is ~0.2 — they MUST diverge.
    """
    labels = list(FROZEN_LABELS)
    rows: list[dict[str, Any]] = []
    for repeat in (0, 1, 2):
        for local_fold in (0, 1, 2, 3, 4):
            # Each fold is dedicated to ONE class; the choice cycles through
            # the 5 frozen labels so every class appears in 3 folds (= 15/5).
            sole_label = labels[(repeat * 5 + local_fold) % len(labels)]
            rows.append(
                _make_record_row(
                    record_id=f"R-r{repeat}f{local_fold}",
                    subject_id=f"S{repeat}-{local_fold}",
                    y_true=sole_label,
                    y_pred=sole_label,
                    repeat=repeat,
                    local_fold=local_fold,
                )
            )
    frame = pd.DataFrame(rows)
    stitched_metrics = compute_stitched_classification_metrics(frame)
    per_fold_metrics = []
    for (repeat, fold), group in frame.groupby(["repeat", "local_fold"], sort=True):
        true = np.asarray([LABEL_TO_INDEX[str(v)] for v in group["y_true"]], dtype=np.int64)
        pred = np.asarray([LABEL_TO_INDEX[str(v)] for v in group["y_pred"]], dtype=np.int64)
        from topper_perception.neural.metrics import compute_classification_metrics

        per_fold_metrics.append(
            compute_classification_metrics(true, pred, FROZEN_LABELS).macro_f1
        )
    mean_of_folds = float(np.mean(per_fold_metrics))
    # Pool-first MUST be materially higher than the per-fold mean because the
    # pool contains all 5 classes correctly classified while each fold only
    # contains one class.
    assert stitched_metrics["macro_f1"] > mean_of_folds + 0.5
    assert stitched_metrics["accuracy"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# P6 single-checkpoint reject metrics on the stitched frame
# ---------------------------------------------------------------------------


def test_compute_p6_single_uses_frozen_threshold_0_94() -> None:
    """The frozen threshold is 0.94 — it must not be passed in as a free parameter."""
    rows = [
        _make_record_row("R0", "S0", "empty", "empty", repeat=0, local_fold=0, confidence=0.96),
        _make_record_row("R1", "S0", "supine", "supine", repeat=0, local_fold=0, confidence=0.80),
    ]
    frame = pd.DataFrame(rows)
    # High-confidence record (0.96) is above threshold → accepted.
    # Low-confidence record (0.80) is below threshold → rejected (UNKNOWN/REJECT).
    metrics = compute_p6_single(frame)
    assert metrics["threshold"] == pytest.approx(FROZEN_P6_SINGLE_THRESHOLD)
    assert metrics["coverage"] == pytest.approx(0.5)
    assert metrics["wrong_action_n"] == 0
    assert metrics["accepted_accuracy"] == pytest.approx(1.0)


def test_compute_p6_single_signature_rejects_threshold_kwarg() -> None:
    """Round 2: callers cannot pass ``threshold=`` anymore; the parameter is gone.

    We use ``inspect.signature`` because the override parameter was removed;
    passing it via ``**kwargs`` would have raised ``TypeError`` even before,
    but we explicitly assert the signature has no ``threshold`` argument.
    """
    import inspect
    sig = inspect.signature(compute_p6_single)
    assert "threshold" not in sig.parameters, (
        "compute_p6_single must not expose a threshold parameter; it is the "
        "frozen contract value 0.94 and is loaded from the archive rule block."
    )


def test_compute_p6_single_reports_war_as_fraction_of_total() -> None:
    """WAR must be wrong_n / n, never wrong_n / accepted_n (Reviewer point: WAR)."""
    rows = [
        _make_record_row("R0", "S0", "empty", "empty", repeat=0, local_fold=0, confidence=0.99),
        _make_record_row("R1", "S0", "empty", "supine", repeat=0, local_fold=0, confidence=0.99),
    ]
    frame = pd.DataFrame(rows)
    metrics = compute_p6_single(frame)
    assert metrics["n"] == 2
    assert metrics["accepted_n"] == 2
    assert metrics["wrong_action_n"] == 1
    assert metrics["wrong_action_rate"] == pytest.approx(0.5)
    assert metrics["accepted_error_rate"] == pytest.approx(0.5)


def test_compute_p6_single_returns_none_when_no_record_accepted() -> None:
    rows = [
        _make_record_row("R0", "S0", "empty", "supine", repeat=0, local_fold=0, confidence=0.50),
    ]
    metrics = compute_p6_single(pd.DataFrame(rows))
    assert metrics["accepted_n"] == 0
    assert metrics["coverage"] == pytest.approx(0.0)
    assert metrics["accepted_accuracy"] is None
    assert metrics["wrong_action_rate"] == pytest.approx(0.0)


def test_compute_p6_single_handles_empty_stitched_frame() -> None:
    metrics = compute_p6_single(pd.DataFrame(columns=["y_true", "y_pred", "confidence"] + [f"proba__{label}" for label in FROZEN_LABELS]))
    assert metrics["n"] == 0
    assert metrics["coverage"] is None
    assert metrics["wrong_action_rate"] is None


# ---------------------------------------------------------------------------
# P6.1 ensemble metrics on the stitched frame (requires 3 repeats)
# ---------------------------------------------------------------------------


def test_compute_p6_1_ensemble_requires_three_repeats_per_record() -> None:
    """A stitched frame with only one repeat per record_id cannot form an ensemble."""
    rows = [
        _make_record_row("R0", "S0", "empty", "empty", repeat=0, local_fold=0),
        _make_record_row("R1", "S0", "supine", "supine", repeat=0, local_fold=0),
    ]
    frame = pd.DataFrame(rows)
    metrics = compute_p6_1_ensemble(frame)
    assert metrics["rule_kind"] == "p6_1_ensemble"
    assert metrics["temperature"] == pytest.approx(FROZEN_P6_1_TEMPERATURE)
    assert metrics["threshold"] == pytest.approx(FROZEN_P6_1_THRESHOLD)
    assert metrics["require_unanimous"] is True
    # Single-repeat stitched frames cannot produce ensembles: structured empty
    # result, NOT a covert fallback to the P6 single rule.
    assert metrics["coverage"] == pytest.approx(0.0)
    assert metrics["wrong_action_rate"] == pytest.approx(0.0)


def test_compute_p6_1_ensemble_unanimity_branch_drops_non_unanimous_records() -> None:
    """Two of three repeats predicting 'empty', one predicting 'supine' must NOT be accepted."""
    rows = []
    for repeat in (0, 1, 2):
        y_pred = "empty" if repeat != 1 else "supine"
        rows.append(
            _make_record_row(
                "R0", "S0", "empty", y_pred,
                repeat=repeat, local_fold=0, confidence=0.99,
            )
        )
    metrics = compute_p6_1_ensemble(pd.DataFrame(rows))
    assert metrics["unanimous_count"] == 0
    assert metrics["accepted_n"] == 0
    assert metrics["coverage"] == pytest.approx(0.0)


def test_compute_p6_1_ensemble_accepts_unanimous_high_confidence() -> None:
    """Three repeats unanimous on 'empty' with high ensemble confidence → accepted."""
    rows = [
        _make_record_row("R0", "S0", "empty", "empty", repeat=repeat, local_fold=0, confidence=0.99)
        for repeat in (0, 1, 2)
    ]
    metrics = compute_p6_1_ensemble(pd.DataFrame(rows))
    assert metrics["unanimous_count"] == 1
    assert metrics["accepted_n"] == 1
    assert metrics["coverage"] == pytest.approx(1.0)
    assert metrics["accepted_accuracy"] == pytest.approx(1.0)


def test_compute_p6_1_ensemble_war_uses_total_n_not_accepted_n() -> None:
    """Even if a unanimous record was wrong, WAR = wrong_n / total_n."""
    rows = [
        _make_record_row("R0", "S0", "empty", "supine", repeat=repeat, local_fold=0, confidence=0.99)
        for repeat in (0, 1, 2)
    ]
    metrics = compute_p6_1_ensemble(pd.DataFrame(rows))
    assert metrics["unanimous_count"] == 1
    assert metrics["accepted_n"] == 1  # accepted but wrong
    assert metrics["wrong_action_n"] == 1
    assert metrics["wrong_action_rate"] == pytest.approx(1.0)
    assert metrics["accepted_accuracy"] == pytest.approx(0.0)


def test_compute_p6_1_ensemble_signature_rejects_override_kwargs() -> None:
    """Round 2: callers cannot pass temperature/threshold/require_unanimous."""
    import inspect
    sig = inspect.signature(compute_p6_1_ensemble)
    forbidden = {"temperature", "threshold", "require_unanimous"}
    leak = forbidden & set(sig.parameters)
    assert not leak, (
        f"compute_p6_1_ensemble must not expose {leak} as parameters; the rule "
        f"is loaded from the archive's pinned rule block."
    )


# ---------------------------------------------------------------------------
# Seed and condition aggregation (5 seeds as mean/std/worst — never pooled)
# ---------------------------------------------------------------------------


def test_compute_seed_summary_reports_drop_relative_to_clean() -> None:
    """The seed summary must include the drop in macro-F1 / balanced accuracy relative to clean."""
    clean = pd.DataFrame([
        _make_record_row(f"R-clean-{idx}", "S0", "empty", "empty", repeat=0, local_fold=0)
        for idx in range(10)
    ])
    perturbed = pd.DataFrame([
        _make_record_row(f"R-per-{idx}", "S0", "empty", "empty" if idx < 5 else "supine", repeat=0, local_fold=0)
        for idx in range(10)
    ])
    summary = compute_seed_summary(
        condition="noise_p95_0.05",
        seed=703,
        clean_stitched=clean,
        perturbed_stitched=perturbed,
    )
    assert summary["condition"] == "noise_p95_0.05"
    assert summary["seed"] == 703
    assert summary["n_records"] == 10
    delta = summary["delta_vs_clean"]
    assert delta["record_macro_f1"] < 0
    assert delta["record_balanced_accuracy"] < 0


def test_compute_condition_summary_aggregates_seeds_as_mean_std_worst() -> None:
    """5 seeds must be reported as mean/std/worst — never pooled into one sample."""
    seed_summaries = [
        {"seed": 701, "delta_vs_clean": {"record_macro_f1": -0.05}},
        {"seed": 702, "delta_vs_clean": {"record_macro_f1": -0.10}},
        {"seed": 703, "delta_vs_clean": {"record_macro_f1": -0.15}},
        {"seed": 704, "delta_vs_clean": {"record_macro_f1": -0.20}},
        {"seed": 705, "delta_vs_clean": {"record_macro_f1": -0.25}},
    ]
    summary = compute_condition_summary("noise_p95_0.05", seed_summaries)
    assert summary["condition"] == "noise_p95_0.05"
    stats = summary["delta_macro_f1"]
    assert stats["mean"] == pytest.approx(-0.15)
    assert stats["std"] == pytest.approx(0.0707, abs=1e-3)
    assert stats["worst"] == pytest.approx(-0.25)
    assert summary["n_seeds"] == 5
    # Critically: the aggregation MUST NOT pool seeds into a single number
    # with n_seeds=25. This contract is enforced by reporting n_seeds=5.
    assert summary["n_seeds"] == len(seed_summaries)


def test_compute_condition_summary_rejects_partial_seed_set() -> None:
    """If fewer than 5 seeds are supplied the aggregation must fail closed."""
    seed_summaries = [
        {"seed": 701, "delta_vs_clean": {"record_macro_f1": -0.05}},
        {"seed": 702, "delta_vs_clean": {"record_macro_f1": -0.10}},
    ]
    with pytest.raises(ValueError, match="seed"):
        compute_condition_summary("noise_p95_0.05", seed_summaries)


# ---------------------------------------------------------------------------
# Per-class and per-subject breakdowns
# ---------------------------------------------------------------------------


def test_compute_per_class_breakdown_covers_all_frozen_labels(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    # records_per_fold=10 so the index cycles through all 5 frozen labels.
    _build_full_oof_fixture(src, records_per_fold=10)
    stitched = load_condition_seed_oof(src, "noise_p95_0.05", seed=701)
    breakdown = compute_per_class_breakdown(stitched)
    assert isinstance(breakdown, pd.DataFrame)
    assert set(breakdown["y_true"]) == set(FROZEN_LABELS)


def test_compute_per_subject_breakdown_exposes_war_and_coverage(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src, records_per_fold=4, subjects_per_fold=2)
    stitched = load_condition_seed_oof(src, "noise_p95_0.10", seed=702)
    breakdown = compute_per_subject_breakdown(stitched)
    required = {
        "subject_id", "n", "wrong_action_n", "wrong_action_rate",
        "accuracy", "accepted_n", "coverage", "accepted_accuracy",
        "accepted_error_rate", "p6_threshold",
    }
    assert required.issubset(breakdown.columns)
    assert len(breakdown) == 2  # two subjects in the fixture


# ---------------------------------------------------------------------------
# Worst-subject selection by FOUR criteria
# ---------------------------------------------------------------------------


def test_compute_worst_subjects_returns_four_distinct_criteria() -> None:
    per_subject = pd.DataFrame([
        {"subject_id": "S0", "wrong_action_rate": 0.10, "coverage": 0.80, "accepted_accuracy": 0.95, "accuracy": 0.90},
        {"subject_id": "S1", "wrong_action_rate": 0.30, "coverage": 0.50, "accepted_accuracy": 0.70, "accuracy": 0.85},
        {"subject_id": "S2", "wrong_action_rate": 0.05, "coverage": 0.95, "accepted_accuracy": 0.99, "accuracy": 0.98},
    ])
    worst = compute_worst_subjects(per_subject)
    assert set(worst) == {"by_wrong_action_rate", "by_coverage", "by_accepted_accuracy", "by_raw_accuracy"}
    assert worst["by_wrong_action_rate"]["subject_id"] == "S1"
    assert worst["by_coverage"]["subject_id"] == "S1"
    assert worst["by_accepted_accuracy"]["subject_id"] == "S1"
    assert worst["by_raw_accuracy"]["subject_id"] == "S1"


def test_compute_worst_subjects_tie_break_by_subject_id() -> None:
    per_subject = pd.DataFrame([
        {"subject_id": "S2", "wrong_action_rate": 0.10, "coverage": 0.50, "accepted_accuracy": 0.90, "accuracy": 0.90},
        {"subject_id": "S1", "wrong_action_rate": 0.10, "coverage": 0.50, "accepted_accuracy": 0.90, "accuracy": 0.90},
        {"subject_id": "S0", "wrong_action_rate": 0.05, "coverage": 0.95, "accepted_accuracy": 0.99, "accuracy": 0.99},
    ])
    worst = compute_worst_subjects(per_subject)
    # Ties are broken by subject_id ASC.
    assert worst["by_wrong_action_rate"]["subject_id"] == "S1"
    assert worst["by_coverage"]["subject_id"] == "S1"
    assert worst["by_accepted_accuracy"]["subject_id"] == "S1"
    assert worst["by_raw_accuracy"]["subject_id"] == "S1"


def test_compute_worst_subjects_handles_empty_dataframe() -> None:
    worst = compute_worst_subjects(pd.DataFrame())
    assert worst == {
            "by_wrong_action_rate": None,
            "by_coverage": None,
            "by_accepted_accuracy": None,
            "by_raw_accuracy": None,
        }


def test_compute_worst_subjects_treats_none_as_zero_for_sort() -> None:
    per_subject = pd.DataFrame([
        {"subject_id": "S0", "wrong_action_rate": 0.10, "coverage": 0.80, "accepted_accuracy": None, "accuracy": 0.90},
        {"subject_id": "S1", "wrong_action_rate": 0.05, "coverage": 0.95, "accepted_accuracy": 0.99, "accuracy": 0.98},
    ])
    worst = compute_worst_subjects(per_subject)
    # S0 has accepted_accuracy=None → treated as 0.0 → worst-by-accepted_accuracy.
    assert worst["by_accepted_accuracy"]["subject_id"] == "S0"


# ---------------------------------------------------------------------------
# High-confidence error filtering
# ---------------------------------------------------------------------------


def test_compute_high_confidence_errors_filters_by_threshold() -> None:
    error_rows = [
        _make_record_row("R0", "S0", "empty", "supine", repeat=0, local_fold=0, confidence=0.95),
        _make_record_row("R1", "S0", "empty", "supine", repeat=0, local_fold=0, confidence=0.85),
        _make_record_row("R2", "S0", "empty", "supine", repeat=0, local_fold=0, confidence=0.99),
    ]
    errors = pd.DataFrame(error_rows)
    high_conf = compute_high_confidence_errors(errors, threshold=FROZEN_HIGH_CONFIDENCE_THRESHOLD)
    assert set(high_conf["record_id"]) == {"R0", "R2"}


def test_compute_high_confidence_errors_sorts_by_confidence_desc() -> None:
    error_rows = [
        _make_record_row("R0", "S0", "empty", "supine", repeat=0, local_fold=0, confidence=0.91),
        _make_record_row("R1", "S0", "empty", "supine", repeat=0, local_fold=0, confidence=0.99),
        _make_record_row("R2", "S0", "empty", "supine", repeat=0, local_fold=0, confidence=0.95),
    ]
    high_conf = compute_high_confidence_errors(pd.DataFrame(error_rows))
    assert list(high_conf["record_id"]) == ["R1", "R2", "R0"]


# ---------------------------------------------------------------------------
# Artifact writing
# ---------------------------------------------------------------------------


def test_write_analysis_artifacts_produces_eight_files(tmp_path: Path) -> None:
    """End-to-end: run the full pipeline on a tiny fixture and confirm artifacts."""
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src, records_per_fold=4, subjects_per_fold=2)
    out = tmp_path / "out"
    result = analyze_p7_full(src, out)
    expected = {
        "summary.json",
        "condition_metrics.csv",
        "per_class_metrics.csv",
        "per_subject_metrics.csv",
        "worst_subjects.csv",
        "error_cases.csv",
        "high_confidence_errors.csv",
        "evidence_manifest.json",
    }
    written = {path.name for path in out.iterdir()}
    assert expected.issubset(written)
    # summary.json must include the clean baseline + every frozen condition.
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert "clean" in summary
    assert "conditions" in summary
    assert set(summary["conditions"].keys()) == set(FROZEN_CONDITION_NAMES)


def test_write_analysis_artifacts_serializes_nonfinite_as_null(tmp_path: Path) -> None:
    """Per Reviewer point: JSON output must NOT contain raw NaN / Infinity tokens."""
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src, records_per_fold=2, subjects_per_fold=1)
    out = tmp_path / "out"
    analyze_p7_full(src, out)
    # Round-trip through strict JSON to confirm no non-finite constants snuck in.
    raw = (out / "summary.json").read_text(encoding="utf-8")
    for token in ("NaN", "Infinity", "-Infinity"):
        assert token not in raw, f"non-finite JSON constant leaked: {token}"
    strict_json_load(out / "summary.json")  # raises if any NaN / Infinity remains
    strict_json_load(out / "evidence_manifest.json")


def test_analyze_p7_full_records_clean_baseline_first(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src, records_per_fold=3)
    out = tmp_path / "out"
    analyze_p7_full(src, out)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    clean = summary["clean"]
    assert clean["n_records_total"] == 3 * 15
    assert "stitched_metrics" in clean
    assert "p6_single_rule" in clean
    assert "p6_1_ensemble_rule" in clean


# ---------------------------------------------------------------------------
# Integration test against the real evidence pack — opt-in only
# ---------------------------------------------------------------------------


REAL_ARCHIVE = Path(r"C:\Users\23939\AppData\Local\Temp\smarttopper-autodl\EXP-P7-FULL-20260820-R02.tar.gz")
REAL_ARCHIVE_SHA = "cbaffa74878b149e546a42826ae373442c62683af890362684f80963e7fddda1"


@pytest.mark.skipif(not REAL_ARCHIVE.is_file(), reason="real evidence pack not present")
def test_analyze_p7_full_against_real_pack_cross_checks_anchor_metrics() -> None:
    """End-to-end against the real archive; verifies the 8 anchor values the Reviewer pinned.

    Anchor values are documented in ``docs/stage_reports/...``. They are
    allowed to differ by a tiny float epsilon but MUST NOT be silently coerced
    into a match — any material divergence is reported as a hard test failure.
    """
    out = REAL_ARCHIVE.parent / "EXP-P7-FULL-ANALYSIS-20260821-R01"
    if out.exists():
        for child in sorted(out.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
            else:
                child.unlink()
    result = analyze_p7_full(
        REAL_ARCHIVE, out, expected_archive_sha256=REAL_ARCHIVE_SHA,
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    clean_macro_f1 = summary["clean"]["stitched_metrics"]["macro_f1"]
    # Anchor: clean macro-F1 ≈ 0.986644
    assert abs(clean_macro_f1 - 0.986644) < 1e-3, (
        f"clean macro_f1 drift: got {clean_macro_f1}, expected ≈ 0.986644"
    )
    conditions = summary["conditions"]
    # Anchor: noise_p95_0.10 macro-F1 ≈ 0.668383
    noise_macro = conditions["noise_p95_0.10"]["record_metrics_stitched_means"]["macro_f1"]["mean"]
    assert abs(noise_macro - 0.668383) < 5e-3, (
        f"noise_p95_0.10 macro_f1 drift: got {noise_macro}, expected ≈ 0.668383"
    )
    # Anchor: density_stride_4_4 macro-F1 ≈ 0.682021
    dense_macro = conditions["density_stride_4_4"]["record_metrics_stitched_means"]["macro_f1"]["mean"]
    assert abs(dense_macro - 0.682021) < 5e-3, (
        f"density_stride_4_4 macro_f1 drift: got {dense_macro}, expected ≈ 0.682021"
    )
    # Anchor: noise_p95_0.05 macro-F1 ≈ 0.938365
    mild_noise_macro = conditions["noise_p95_0.05"]["record_metrics_stitched_means"]["macro_f1"]["mean"]
    assert abs(mild_noise_macro - 0.938365) < 5e-3, (
        f"noise_p95_0.05 macro_f1 drift: got {mild_noise_macro}, expected ≈ 0.938365"
    )
    # Anchor: 10% noise P6 coverage ≈ 0.689972
    p6_coverage = conditions["noise_p95_0.10"]["p6_single_rule_means"]["coverage"]["mean"]
    assert abs(p6_coverage - 0.689972) < 5e-3, (
        f"10% noise P6 coverage drift: got {p6_coverage}, expected ≈ 0.689972"
    )
    # Anchor: 10% noise P6 accepted accuracy ≈ 0.660838
    p6_acc = conditions["noise_p95_0.10"]["p6_single_rule_means"]["accepted_accuracy"]["mean"]
    assert abs(p6_acc - 0.660838) < 5e-3, (
        f"10% noise P6 accepted_accuracy drift: got {p6_acc}, expected ≈ 0.660838"
    )
    # Anchor: 10% noise P6 WAR ≈ 0.234013
    p6_war = conditions["noise_p95_0.10"]["p6_single_rule_means"]["wrong_action_rate"]["mean"]
    assert abs(p6_war - 0.234013) < 5e-3, (
        f"10% noise P6 WAR drift: got {p6_war}, expected ≈ 0.234013"
    )
    # Worst-subject WAR (across all conditions/seeds) should be near 0.60.
    # The anchor refers to the maximum subject-level wrong_action_rate
    # observed in per_subject_metrics.csv (not in the summary-only
    # worst_subjects.csv, which lists only subject identifiers).
    per_subject = pd.read_csv(out / "per_subject_metrics.csv")
    max_war = float(per_subject["wrong_action_rate"].max())
    assert 0.45 < max_war < 0.75, (
        f"worst-subject WAR drifted outside sanity band: got {max_war}"
    )


# ---------------------------------------------------------------------------
# Round 2: fail-closed rule block + tamper / CLI regression tests
# ---------------------------------------------------------------------------


def test_load_p6_single_rule_from_archive_reads_pinned_block(tmp_path: Path) -> None:
    """The loader must read the rule block from condition_comparison.json verbatim."""
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src)
    rule = load_p6_single_rule_from_archive(src)
    assert rule.rule_kind == "p6_single"
    assert rule.threshold == pytest.approx(FROZEN_P6_SINGLE_THRESHOLD)
    assert rule.source_expected_sha256 == rule.source_actual_sha256


def test_load_p6_1_ensemble_rule_from_archive_reads_pinned_block(tmp_path: Path) -> None:
    """The loader must read the rule block from condition_comparison.json verbatim."""
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src)
    rule = load_p6_1_ensemble_rule_from_archive(src)
    assert rule.rule_kind == "p6_1_ensemble"
    assert rule.temperature == pytest.approx(FROZEN_P6_1_TEMPERATURE)
    assert rule.threshold == pytest.approx(FROZEN_P6_1_THRESHOLD)
    assert rule.require_unanimous is True


def test_verify_evidence_archive_rejects_tampered_p6_threshold(tmp_path: Path) -> None:
    """If condition_comparison.json's p6_single_rule.threshold is tampered, fail closed."""
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src)
    cc_path = src / "condition_comparison.json"
    payload = json.loads(cc_path.read_text(encoding="utf-8"), parse_constant=lambda _: None)
    payload["p6_single_rule"]["threshold"] = 0.80  # tamper — not 0.94
    cc_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ArchiveIntegrityError, match="threshold"):
        verify_evidence_archive(src, expected_sha256=None)


def test_verify_evidence_archive_rejects_tampered_p6_actual_sha(tmp_path: Path) -> None:
    """If the archive-embedded SHA pair is drifted, fail closed (no silent override)."""
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src)
    cc_path = src / "condition_comparison.json"
    payload = json.loads(cc_path.read_text(encoding="utf-8"), parse_constant=lambda _: None)
    payload["p6_single_rule"]["source_actual_sha256"] = "0" * 64  # drift
    cc_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ArchiveIntegrityError, match="SHA"):
        verify_evidence_archive(src, expected_sha256=None)


def test_verify_evidence_archive_rejects_tampered_p6_1_temperature(tmp_path: Path) -> None:
    """If condition_comparison.json's p6_1_ensemble_rule.temperature is tampered, fail closed."""
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src)
    cc_path = src / "condition_comparison.json"
    payload = json.loads(cc_path.read_text(encoding="utf-8"), parse_constant=lambda _: None)
    payload["p6_1_ensemble_rule"]["temperature"] = 1.5  # tamper — not 0.75
    cc_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ArchiveIntegrityError, match="temperature"):
        verify_evidence_archive(src, expected_sha256=None)


def test_verify_evidence_archive_rejects_tampered_p6_1_require_unanimous(tmp_path: Path) -> None:
    """If condition_comparison.json's p6_1_ensemble_rule.require_unanimous flips, fail closed."""
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src)
    cc_path = src / "condition_comparison.json"
    payload = json.loads(cc_path.read_text(encoding="utf-8"), parse_constant=lambda _: None)
    payload["p6_1_ensemble_rule"]["require_unanimous"] = False  # tamper — must be True
    cc_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ArchiveIntegrityError, match="unanimous"):
        verify_evidence_archive(src, expected_sha256=None)


def test_verify_evidence_archive_rejects_missing_rule_block(tmp_path: Path) -> None:
    """If the rule block is missing entirely, fail closed (Reviewer Round 2)."""
    src = tmp_path / "src"
    src.mkdir()
    _build_full_oof_fixture(src)
    cc_path = src / "condition_comparison.json"
    payload = json.loads(cc_path.read_text(encoding="utf-8"), parse_constant=lambda _: None)
    del payload["p6_single_rule"]
    cc_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ArchiveIntegrityError, match="p6_single_rule"):
        verify_evidence_archive(src, expected_sha256=None)


def test_assert_rule_values_match_frozen_rejects_drifted_value() -> None:
    """Even with matching SHAs, drifted numeric values must fail closed."""
    from topper_perception.neural.p7_full_analysis import PinnedRule
    fake_p6 = PinnedRule(
        rule_kind="p6_single",
        source_path="x",
        source_expected_sha256="0" * 64,
        source_actual_sha256="0" * 64,
        threshold=0.80,  # drifted
        temperature=None,
        require_unanimous=None,
        threshold_pointer=None,
        fallback_threshold_pointer=None,
        temperature_pointer=None,
        rule_pointer=None,
        unanimity_field_pointer=None,
    )
    fake_p6_1 = PinnedRule(
        rule_kind="p6_1_ensemble",
        source_path="x",
        source_expected_sha256="0" * 64,
        source_actual_sha256="0" * 64,
        threshold=FROZEN_P6_1_THRESHOLD,
        temperature=FROZEN_P6_1_TEMPERATURE,
        require_unanimous=FROZEN_P6_1_REQUIRE_UNANIMOUS,
        threshold_pointer=None,
        fallback_threshold_pointer=None,
        temperature_pointer=None,
        rule_pointer=None,
        unanimity_field_pointer=None,
    )
    with pytest.raises(ArchiveIntegrityError, match="threshold"):
        assert_rule_values_match_frozen(fake_p6, fake_p6_1)


def test_cli_does_not_expose_frozen_value_override_flags() -> None:
    """Round 2: the CLI MUST NOT expose flags that override the frozen rule values.

    The frozen rule parameters must be loaded from the archive's pinned rule
    block inside condition_comparison.json, never from CLI knobs.
    """
    import importlib.util
    cli_path = Path("scripts") / "analyze_popu_p7_full.py"
    source = cli_path.read_text(encoding="utf-8")
    forbidden_flags = (
        '"--p6-single-threshold"',
        '"--p6-1-temperature"',
        '"--p6-1-threshold"',
        '"--p6-1-require-unanimous"',
        '"--no-p6-1-require-unanimous"',
    )
    for flag in forbidden_flags:
        assert flag not in source, (
            f"CLI source declares forbidden frozen-value override flag {flag!r}; "
            f"per Round 2, rule values must come from the archive rule block."
        )

    spec = importlib.util.spec_from_file_location("analyze_popu_p7_full", cli_path)
    assert spec is not None and spec.loader is not None, "Could not load CLI module spec."
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    import inspect
    sig = inspect.signature(module.parse_args)
    forbidden_kwargs = {
        "p6_single_threshold",
        "p6_1_temperature",
        "p6_1_threshold",
        "p6_1_require_unanimous",
    }
    leak = forbidden_kwargs & set(sig.parameters)
    assert not leak, (
        f"parse_args signature exposes forbidden kwargs {leak}; CLI must not "
        f"override the frozen rule contract."
    )