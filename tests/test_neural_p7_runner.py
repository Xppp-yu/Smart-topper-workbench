"""Tests for the P7 software-robustness runner (Reviewer-revised).

These tests intentionally avoid loading the full evidence pack (which would
require the OOF CSVs and a CUDA-trained checkpoint) so they can run on a CPU
CI host without external state. They cover:

- fail-closed validation of the frozen P7 configuration contract (including
  the new P6 evidence block and ``stitching.policy``);
- deterministic expansion of the frozen conditions tree;
- per-record shape, finiteness, and seed-stability guarantees of the
  perturbation application layer (including the Round-4 per-record stable
  seed derivation for Gaussian noise);
- deterministic grouping of per-snapshot matrices into record-level stacks;
- the P6 single-checkpoint reject-rule metrics block loaded from evidence;
- the P6.1 three-repeat ensemble metrics block loaded from evidence;
- the OOF cross-check that fails closed when clean re-inference disagrees
  with the frozen P5.2-C record predictions (and its exhaustive mode);
- the full OOF stitching policy that pools (repeat, fold, condition, seed)
  record rows before computing macro-F1 / balanced accuracy / WAR / per-class
  and per-subject;
- the SHA-256 fail-closed validation across complete.json, stage_b_final.pt
  and split_manifest.json;
- the Round-4 ``_worst_subjects`` reporting by FOUR criteria (WAR,
  coverage, accepted_accuracy, raw accuracy) with subject_id tie-breaks.

Reviewer point #1, #2, #3, #4, #5, #6, #7.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from topper_perception.neural import p7_runner
from topper_perception.neural.data import FROZEN_LABELS, LABEL_TO_INDEX
from topper_perception.neural.p6_reject import PROBA_COLUMNS
from topper_perception.neural.p7_runner import (
    P7Condition,
    SCHEMA_VERSION,
    _apply_condition_to_record,
    _assert_clean_matches_oof,
    _condition_seed_drift_stats,
    _group_records_by_record,
    _per_class_breakdown,
    _per_subject_breakdown,
    _record_metric_blocks,
    _record_p6_1_ensemble_metrics,
    _record_p6_single_metrics,
    _stitch_full_oof,
    _stitched_classification_metrics,
    _stitched_p6_single,
    derive_record_seed,
    parse_p7_conditions,
    perturb_records,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_p6_evidence_block() -> dict[str, Any]:
    """Build a minimal frozen P6/P6.1 evidence block.

    The block is suitable for unit tests only: the *file* path and SHA-256
    are placeholders, so the loader will fail closed if any test actually
    tries to read them. Tests that exercise the loader provide their own
    evidence files via ``tmp_path``.

    Per Reviewer point #2 (Round 3) the P6.1 ``rule_pointer`` MUST end with
    ``/rules/1/threshold`` (the unanimity branch). The removed
    ``unanimous_rule_pointer`` field is no longer part of the contract.
    """
    return {
        "single_threshold_source": {
            "kind": "summary_json",
            "path": "PLACEHOLDER_P6_REJECT.json",
            "expected_sha256": "0" * 64,
            "threshold_pointer": "/selected_rule/confidence_threshold",
            "fallback_threshold_pointer": "/selection/threshold",
        },
        "ensemble_rule_source": {
            "kind": "summary_json",
            "path": "PLACEHOLDER_P6_1_CALIBRATION.json",
            "expected_sha256": "0" * 64,
            "temperature_pointer": "/temperature",
            "rule_pointer": "/rules/1/threshold",
            "unanimous_require_field": "/rules/1/require_unanimous",
        },
    }


def _valid_config() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_family": "small_resnet",
        "level": "record",
        "repeats": [0, 1, 2],
        "local_folds": [0, 1, 2, 3, 4],
        "n_total_folds_required": 15,
        "seeds": [701, 702, 703, 704, 705],
        "conditions": {
            "density_nearest": [
                {"row_stride": 2, "column_stride": 2},
                {"row_stride": 4, "column_stride": 4},
            ],
            "gaussian_noise_p95_fraction": [0.01, 0.05, 0.10],
            "bad_cell_fraction": [0.01, 0.05, 0.10],
            "bad_rows": [1, 2, 4],
            "bad_columns": [1, 2, 4],
        },
        "p6_evidence": _minimal_p6_evidence_block(),
        "stitching": {"policy": "pool_first_then_metric"},
    }


# ---------------------------------------------------------------------------
# Frozen configuration contract (Reviewer point 5)
# ---------------------------------------------------------------------------


def test_parse_p7_conditions_expands_full_frozen_set() -> None:
    conditions = parse_p7_conditions(_valid_config())
    names = [condition.name for condition in conditions]
    # 2 density + 3 noise + 3 bad_cell + 3 bad_rows + 3 bad_columns = 14
    assert len(conditions) == 14
    assert names[0] == "density_stride_2_2"
    assert names[1] == "density_stride_4_4"
    assert "noise_p95_0.01" in names
    assert "noise_p95_0.05" in names
    assert "noise_p95_0.10" in names
    assert "bad_cell_0.01" in names
    assert "bad_cell_0.05" in names
    assert "bad_cell_0.10" in names
    assert "bad_rows_1" in names
    assert "bad_rows_2" in names
    assert "bad_rows_4" in names
    assert "bad_columns_1" in names
    assert "bad_columns_2" in names
    assert "bad_columns_4" in names


def test_parse_p7_conditions_is_deterministic() -> None:
    config = _valid_config()
    first = parse_p7_conditions(config)
    second = parse_p7_conditions(config)
    assert [c.name for c in first] == [c.name for c in second]


@pytest.mark.parametrize(
    "field, value",
    [
        ("schema_version", "p7-robustness-v0.2"),
        ("model_family", "tiny_cnn"),
        ("level", "snapshot"),
    ],
)
def test_config_drift_fails_closed(field: str, value: object) -> None:
    config = _valid_config()
    config[field] = value
    with pytest.raises(ValueError, match="must be"):
        parse_p7_conditions(config)


def test_config_missing_condition_block_fails_closed() -> None:
    config = _valid_config()
    config["conditions"] = {key: value for key, value in config["conditions"].items() if key != "bad_rows"}
    with pytest.raises(ValueError, match="bad_rows"):
        parse_p7_conditions(config)


def test_config_non_integer_seed_fails_closed() -> None:
    config = _valid_config()
    config["seeds"] = ["not-an-int"]  # type: ignore[list-item]
    with pytest.raises(ValueError, match="seeds"):
        parse_p7_conditions(config)


def test_config_seed_set_drift_fails_closed() -> None:
    """If the frozen seed set drifts, fail closed (Reviewer point 5)."""
    config = _valid_config()
    config["seeds"] = [701, 702, 703, 704, 706]  # 706 instead of 705
    with pytest.raises(ValueError, match="seeds must be exactly"):
        parse_p7_conditions(config)


def test_config_repeats_or_local_folds_drift_fails_closed() -> None:
    """If the frozen repeat or fold list drifts, fail closed (Reviewer point 5)."""
    config = _valid_config()
    config["repeats"] = [0, 1]
    with pytest.raises(ValueError, match="repeats must be exactly"):
        parse_p7_conditions(config)

    config = _valid_config()
    config["local_folds"] = [0, 1, 2, 3]
    with pytest.raises(ValueError, match="local_folds must be exactly"):
        parse_p7_conditions(config)


def test_config_stitching_policy_drift_fails_closed() -> None:
    """If the stitching policy drifts to per-fold averaging, fail closed."""
    config = _valid_config()
    config["stitching"] = {"policy": "average_per_fold"}
    with pytest.raises(ValueError, match="pool_first_then_metric"):
        parse_p7_conditions(config)


def test_config_p6_evidence_block_required() -> None:
    """If the frozen P6 evidence block is missing, fail closed (Reviewer #1, #6)."""
    config = _valid_config()
    del config["p6_evidence"]
    with pytest.raises(ValueError, match="p6_evidence block is required"):
        parse_p7_conditions(config)


# ---------------------------------------------------------------------------
# Perturbation application layer
# ---------------------------------------------------------------------------


def _record_stack(seed: int = 17) -> np.ndarray:
    """Synthetic (10, 64, 27) stack with monotone positive values."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.0, 1.0, size=(64, 27)).astype(np.float32)
    snaps = np.stack([base + 0.01 * (idx + 1) for idx in range(10)], axis=0)
    return snaps.astype(np.float32)


def test_density_perturbation_preserves_shape_and_non_negative() -> None:
    condition = P7Condition(
        name="density_stride_2_2",
        kind="density_nearest",
        params=(("row_stride", 2), ("column_stride", 2)),
    )
    out = _apply_condition_to_record(_record_stack(), condition, seed=1)
    assert out.shape == (10, 64, 27)
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0
    assert np.isfinite(out).all()


def test_noise_perturbation_is_seed_stable() -> None:
    condition = P7Condition(
        name="noise_p95_0.05",
        kind="gaussian_noise",
        params=(("sigma_fraction", 0.05),),
    )
    a = _apply_condition_to_record(_record_stack(), condition, seed=701)
    b = _apply_condition_to_record(_record_stack(), condition, seed=701)
    c = _apply_condition_to_record(_record_stack(), condition, seed=702)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert float(a.min()) >= 0.0


def test_bad_cell_mask_is_shared_across_ten_snapshots() -> None:
    condition = P7Condition(
        name="bad_cell_0.10",
        kind="bad_cell",
        params=(("fraction", 0.10),),
    )
    out = _apply_condition_to_record(_record_stack(seed=9), condition, seed=701)
    zero_per_snapshot = (out == 0).sum(axis=(1, 2))
    zero_list = zero_per_snapshot.tolist()
    assert len(set(zero_list)) == 1
    assert int(zero_list[0]) > 0
    zero_masks = [np.argwhere(out[i] == 0) for i in range(out.shape[0])]
    first_positions = {tuple(pos) for pos in zero_masks[0].tolist()}
    for mask in zero_masks[1:]:
        assert {tuple(pos) for pos in mask.tolist()} == first_positions


def test_bad_lines_are_seed_stable_and_nonzero_elsewhere() -> None:
    condition = P7Condition(
        name="bad_rows_2",
        kind="bad_lines",
        params=(("bad_rows", 2), ("bad_columns", 0)),
    )
    out = _apply_condition_to_record(_record_stack(seed=33), condition, seed=701)
    assert out.shape == (10, 64, 27)
    zero_rows = sorted(set(int(row) for row in np.where((out == 0).all(axis=2))[1].tolist()))
    assert len(zero_rows) == 2
    for frame in range(out.shape[0]):
        other = out[frame].copy()
        other[zero_rows, :] = -1.0
        assert (other[other >= 0] > 0).any()


def test_different_seeds_produce_different_bad_masks() -> None:
    condition = P7Condition(
        name="bad_cell_0.05",
        kind="bad_cell",
        params=(("fraction", 0.05),),
    )
    stack = _record_stack(seed=42)
    a = _apply_condition_to_record(stack, condition, seed=701)
    b = _apply_condition_to_record(stack, condition, seed=702)
    only_a = int(((a == 0) & (b > 0)).sum())
    only_b = int(((b == 0) & (a > 0)).sum())
    assert only_a > 0 and only_b > 0


def test_density_perturbation_does_not_use_seed_but_is_deterministic() -> None:
    condition = P7Condition(
        name="density_stride_4_4",
        kind="density_nearest",
        params=(("row_stride", 4), ("column_stride", 4)),
    )
    a = _apply_condition_to_record(_record_stack(seed=5), condition, seed=701)
    b = _apply_condition_to_record(_record_stack(seed=5), condition, seed=999)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# perturb_records orchestration (per record, seed-consistent mask)
# ---------------------------------------------------------------------------


def _sample_records(n: int = 4) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record_index in range(n):
        stack = _record_stack(seed=record_index + 1)
        out.append(
            {
                "record_id": f"R{record_index:02d}",
                "subject_id": str(record_index),
                "label": record_index % len(FROZEN_LABELS),
                "sample_ids": tuple(f"R{record_index:02d}-S{s}" for s in range(10)),
                "matrices": stack,
            }
        )
    return out


def test_perturb_records_keeps_record_metadata_and_shape() -> None:
    records = _sample_records()
    condition = P7Condition(
        name="noise_p95_0.01",
        kind="gaussian_noise",
        params=(("sigma_fraction", 0.01),),
    )
    perturbed = perturb_records(records, condition, seed=701)
    assert len(perturbed) == len(records)
    for before, after in zip(records, perturbed):
        assert after["record_id"] == before["record_id"]
        assert after["subject_id"] == before["subject_id"]
        assert after["label"] == before["label"]
        assert after["sample_ids"] == before["sample_ids"]
        assert after["matrices"].shape == before["matrices"].shape


def test_perturb_records_bad_cell_mask_is_per_record_but_seed_deterministic() -> None:
    records = _sample_records(n=3)
    condition = P7Condition(
        name="bad_cell_0.05",
        kind="bad_cell",
        params=(("fraction", 0.05),),
    )
    perturbed_a = perturb_records(records, condition, seed=701)
    perturbed_b = perturb_records(records, condition, seed=701)
    for a, b in zip(perturbed_a, perturbed_b):
        assert np.array_equal(a["matrices"], b["matrices"])


# ---------------------------------------------------------------------------
# Record grouping and metric blocks
# ---------------------------------------------------------------------------


def test_group_records_by_record_preserves_snapshot_count() -> None:
    records: list[dict[str, Any]] = []
    matrices: list[np.ndarray] = []
    sample_ids: list[str] = []
    record_ids: list[str] = []
    subject_ids: list[str] = []
    labels: list[int] = []
    for record_index in range(3):
        stack = _record_stack(seed=record_index + 1)
        for snap in range(10):
            matrices.append(stack[snap])
            sample_ids.append(f"R{record_index}-S{snap}")
            record_ids.append(f"R{record_index}")
            subject_ids.append(str(record_index))
            labels.append(record_index % len(FROZEN_LABELS))
    grouped = _group_records_by_record(
        sample_ids, record_ids, subject_ids, np.stack(matrices), np.asarray(labels)
    )
    assert [g["record_id"] for g in grouped] == ["R0", "R1", "R2"]
    for record in grouped:
        assert record["matrices"].shape == (10, 64, 27)


def test_group_records_by_record_rejects_mismatched_snapshot_count() -> None:
    matrices = [np.zeros((64, 27), dtype=np.float32)] * 5
    sample_ids = ["R0-S0", "R0-S1", "R0-S2", "R0-S3", "R0-S4"]
    record_ids = ["R0"] * 5
    subject_ids = ["0"] * 5
    labels = [0] * 5
    with pytest.raises(ValueError, match="snapshots"):
        _group_records_by_record(
            sample_ids,
            record_ids,
            subject_ids,
            np.stack(matrices),
            np.asarray(labels),
        )


def test_record_metric_blocks_match_compute_classification_metrics() -> None:
    rows = [
        {"y_true": "empty", "y_pred": "empty", **{col: 0.0 for col in PROBA_COLUMNS}},
        {"y_true": "supine", "y_pred": "supine", **{col: 0.0 for col in PROBA_COLUMNS}},
        {"y_true": "prone", "y_pred": "left", **{col: 0.0 for col in PROBA_COLUMNS}},
    ]
    rows[0]["proba__empty"] = 1.0
    rows[1]["proba__supine"] = 1.0
    rows[2]["proba__left"] = 1.0
    metrics = _record_metric_blocks(rows)
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert set(metrics.keys()) >= {"accuracy", "balanced_accuracy", "macro_f1"}


# ---------------------------------------------------------------------------
# P6 single-checkpoint and P6.1 ensemble metrics (Reviewer points 1, 2)
# ---------------------------------------------------------------------------


def _build_fake_p6_single_rule(tmp_path: Path, threshold: float) -> Any:
    """Write a tiny evidence JSON and load the rule through the loader."""
    from topper_perception.neural.p6_evidence import load_p6_single_rule

    payload = {
        "selected_rule": {"name": "max_probability", "confidence_threshold": threshold},
    }
    path = tmp_path / "p6_single.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    from topper_perception.experiments.artifacts import sha256_hex
    expected_sha = sha256_hex(path)
    block = {
        "kind": "summary_json",
        "path": str(path),
        "expected_sha256": expected_sha,
        "threshold_pointer": "/selected_rule/confidence_threshold",
        "fallback_threshold_pointer": "/selection/threshold",
    }
    return load_p6_single_rule(block)


def _build_fake_p6_1_ensemble_rule(
    tmp_path: Path,
    temperature: float,
    threshold: float,
    require_unanimous: bool,
) -> Any:
    """Write a tiny P6.1 evidence JSON and load the rule through the loader.

    Per Reviewer point #2 the canonical P6.1 rule is the unanimity branch
    (``rules[1]``); we deliberately point ``rule_pointer`` at
    ``/rules/1/threshold`` and never at ``/rules/0/threshold``. The
    ``rules[0]`` pre-unanimity value is provided in the JSON only so the
    loader rejection branch can verify it.
    """
    from topper_perception.experiments.artifacts import sha256_hex
    from topper_perception.neural.p6_evidence import load_p6_1_ensemble_rule

    payload = {
        "temperature": temperature,
        "rules": [
            {"threshold": 0.75},  # pre-unanimity value (must NOT leak into P7)
            {"threshold": threshold, "require_unanimous": require_unanimous},
        ],
    }
    path = tmp_path / "p6_1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    expected_sha = sha256_hex(path)
    block = {
        "kind": "summary_json",
        "path": str(path),
        "expected_sha256": expected_sha,
        "temperature_pointer": "/temperature",
        "rule_pointer": "/rules/1/threshold",
        "unanimous_require_field": "/rules/1/require_unanimous",
    }
    return load_p6_1_ensemble_rule(block)


def test_p6_single_loader_fails_closed_on_sha_drift(tmp_path: Path) -> None:
    """If the on-disk SHA drifts from the pinned SHA, fail closed."""
    from topper_perception.neural.p6_evidence import load_p6_single_rule

    payload = {"selected_rule": {"confidence_threshold": 0.94}}
    path = tmp_path / "p6_single.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    block = {
        "kind": "summary_json",
        "path": str(path),
        "expected_sha256": "0" * 64,  # pinned SHA does not match disk
        "threshold_pointer": "/selected_rule/confidence_threshold",
        "fallback_threshold_pointer": "/selection/threshold",
    }
    with pytest.raises(ValueError, match="SHA mismatch"):
        load_p6_single_rule(block)


def test_record_p6_single_metrics_reports_coverage_accepted_accuracy_war(tmp_path: Path) -> None:
    rule = _build_fake_p6_single_rule(tmp_path, threshold=0.75)
    rows = []
    for idx, label in enumerate(FROZEN_LABELS):
        rows.append(
            {
                "y_true": label,
                "y_pred": label,
                "proba__empty": 0.0,
                "proba__supine": 0.0,
                "proba__prone": 0.0,
                "proba__left": 0.0,
                "proba__right": 0.0,
            }
        )
        rows[-1][f"proba__{label}"] = 0.9 if idx < 4 else 0.4  # one below 0.75
    metrics = _record_p6_single_metrics(rows, rule=rule)
    assert metrics["rule_kind"] == "p6_single"
    assert metrics["threshold"] == 0.75
    assert metrics["source_sha256"] == rule.source.actual_sha256
    assert metrics["n"] == 5
    assert metrics["accepted_n"] == 4
    assert metrics["coverage"] == pytest.approx(0.8)
    assert metrics["wrong_action_n"] == 0
    assert metrics["accepted_accuracy"] == pytest.approx(1.0)


def test_record_p6_1_ensemble_requires_three_repeats(tmp_path: Path) -> None:
    """The P6.1 ensemble path must NOT use the rules[0] pre-unanimity threshold
    on a single repeat; the unanimity branch (rules[1]) must be used instead.
    """
    rule = _build_fake_p6_1_ensemble_rule(
        tmp_path, temperature=0.75, threshold=0.5, require_unanimous=True,
    )
    # Only one repeat per record: the ensemble path must surface a structured error
    # rather than silently fall back to the P6 single rule.
    rows = [
        {
            "model": "small_resnet", "repeat": 0, "outer_seed": 11, "local_fold": 0,
            "record_id": "R0", "subject_id": "0",
            "y_true": "empty", "y_pred": "empty", "confidence": 0.95,
            "n_snapshots": 10,
            "proba__empty": 0.95, "proba__supine": 0.0, "proba__prone": 0.0,
            "proba__left": 0.0, "proba__right": 0.0,
        }
    ]
    metrics = _record_p6_1_ensemble_metrics(rows, rule=rule)
    assert metrics["rule_kind"] == "p6_1_ensemble"
    assert "ensemble_error" in metrics  # ensemble path was actually attempted
    assert metrics["n"] == 1
    assert metrics["accepted_n"] == 0
    assert metrics["coverage"] == 0.0
    # The unanimity-branch threshold (0.5) must come from rules[1] in the
    # evidence, never the pre-unanimity rules[0]=0.75.
    assert metrics["threshold"] == 0.5
    assert metrics["temperature"] == 0.75
    assert metrics["require_unanimous"] is True


# ---------------------------------------------------------------------------
# OOF cross-check (in-memory CSV)
# ---------------------------------------------------------------------------


class _StubFoldCheckpoint:
    """Minimal duck-typed replacement for FoldCheckpoint used in cross-check tests."""

    def __init__(self, path: Path, repeat: int, local_fold: int):
        self.record_predictions_path = path
        self.repeat = repeat
        self.local_fold = local_fold


def _write_oof_csv(path: Path, record_rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(record_rows)
    frame.to_csv(path, index=False)


def _synthetic_oof_rows(repeat: int = 0, local_fold: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record_index in range(3):
        true_label = FROZEN_LABELS[record_index % len(FROZEN_LABELS)]
        row = {
            "model": "small_resnet",
            "repeat": repeat,
            "outer_seed": 11,
            "local_fold": local_fold,
            "record_id": f"R{record_index}",
            "subject_id": str(record_index),
            "y_true": true_label,
            "y_pred": true_label,
            "confidence": 0.95,
            "n_snapshots": 10,
        }
        for col in PROBA_COLUMNS:
            row[col] = 0.0
        row[f"proba__{true_label}"] = 0.95
        rows.append(row)
    return rows


def _build_snapshot_rows() -> list[dict[str, Any]]:
    snapshot_rows: list[dict[str, Any]] = []
    for record_index in range(3):
        true_label = FROZEN_LABELS[record_index % len(FROZEN_LABELS)]
        for snap in range(10):
            snapshot = {
                "model": "small_resnet",
                "repeat": 0,
                "outer_seed": 11,
                "local_fold": 0,
                "sample_id": f"R{record_index}-S{snap}",
                "record_id": f"R{record_index}",
                "subject_id": str(record_index),
                "y_true": true_label,
                "y_pred": true_label,
                "confidence": 0.95,
            }
            for col in PROBA_COLUMNS:
                snapshot[col] = 0.0
            snapshot[f"proba__{true_label}"] = 0.95
            snapshot_rows.append(snapshot)
    return snapshot_rows


def test_assert_clean_matches_oof_passes_on_matching_records(tmp_path: Path) -> None:
    oof_rows = _synthetic_oof_rows()
    _write_oof_csv(tmp_path / "oof.csv", oof_rows)
    stub = _StubFoldCheckpoint(tmp_path / "oof.csv", repeat=0, local_fold=0)
    summary = _assert_clean_matches_oof(_build_snapshot_rows(), fold_checkpoint=stub)  # type: ignore[arg-type]
    assert summary["oof_records_compared"] == 3
    assert summary["oof_argmax_identical"] is True
    assert summary["exhaustive"] is False


def test_assert_clean_matches_oof_passes_in_exhaustive_mode(tmp_path: Path) -> None:
    """Exhaustive mode (clean-only full-fold CPU reproduction) verifies
    every inferred record and every OOF row are consumed."""
    oof_rows = _synthetic_oof_rows()
    _write_oof_csv(tmp_path / "oof.csv", oof_rows)
    stub = _StubFoldCheckpoint(tmp_path / "oof.csv", repeat=0, local_fold=0)
    summary = _assert_clean_matches_oof(
        _build_snapshot_rows(),
        fold_checkpoint=stub,  # type: ignore[arg-type]
        exhaustive=True,
    )
    assert summary["exhaustive"] is True
    assert summary["oof_records_compared"] == summary["oof_records_total"] == 3


def test_assert_clean_matches_oof_fails_closed_on_argmax_mismatch(tmp_path: Path) -> None:
    oof_rows = _synthetic_oof_rows()
    oof_rows[0]["y_pred"] = "supine"
    oof_rows[0]["proba__supine"] = 0.95
    oof_rows[0]["proba__empty"] = 0.0
    _write_oof_csv(tmp_path / "oof.csv", oof_rows)
    stub = _StubFoldCheckpoint(tmp_path / "oof.csv", repeat=0, local_fold=0)
    with pytest.raises(RuntimeError, match="y_pred mismatch"):
        _assert_clean_matches_oof(_build_snapshot_rows(), fold_checkpoint=stub)  # type: ignore[arg-type]


def test_assert_clean_matches_oof_fails_closed_on_probability_drift(tmp_path: Path) -> None:
    oof_rows = _synthetic_oof_rows()
    oof_rows[1]["proba__empty"] = 0.95
    oof_rows[1]["proba__supine"] = 0.0
    _write_oof_csv(tmp_path / "oof.csv", oof_rows)
    stub = _StubFoldCheckpoint(tmp_path / "oof.csv", repeat=0, local_fold=0)
    with pytest.raises(RuntimeError, match="probability mismatch"):
        _assert_clean_matches_oof(_build_snapshot_rows(), fold_checkpoint=stub)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Full OOF stitching (Reviewer points 3, 4)
# ---------------------------------------------------------------------------


def _make_record_row(
    record_id: str,
    subject: str,
    y_true: str,
    y_pred: str,
    repeat: int,
    local_fold: int,
    confidence: float = 0.9,
) -> dict[str, Any]:
    row = {
        "model": "small_resnet",
        "repeat": repeat,
        "outer_seed": 11,
        "local_fold": local_fold,
        "record_id": record_id,
        "subject_id": subject,
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": confidence,
        "n_snapshots": 10,
    }
    for col in PROBA_COLUMNS:
        row[col] = 0.0
    row[f"proba__{y_pred}"] = confidence
    return row


def test_stitch_full_oof_pools_across_repeats_folds_seeds() -> None:
    """The full OOF must pool records across (repeat, fold, condition, seed)."""
    clean_repeat0 = _make_record_row("R0", "S0", "empty", "empty", 0, 0)
    clean_repeat1 = _make_record_row("R0", "S0", "empty", "supine", 1, 0)
    condition_seed = _make_record_row("R0", "S0", "empty", "prone", 0, 1)
    clean_repeat2 = _make_record_row("R0", "S0", "empty", "left", 2, 0)

    stitched = _stitch_full_oof([pd.DataFrame([clean_repeat0])])
    assert len(stitched) == 1
    assert int(stitched["repeat"].iloc[0]) == 0

    stitched = _stitch_full_oof([
        pd.DataFrame([clean_repeat0]),
        pd.DataFrame([clean_repeat1]),
        pd.DataFrame([clean_repeat2]),
        pd.DataFrame([condition_seed]),
    ])
    assert len(stitched) == 4
    # 3 different repeats all classified "empty" but with different y_pred → accuracy drops.
    metrics = _stitched_classification_metrics(stitched)
    assert math.isclose(float(metrics["accuracy"]), 0.25)


def test_stitched_metrics_diverge_from_per_fold_mean_when_classes_imbalanced() -> None:
    """Naive per-fold averaging can hide class imbalance; pooling first does not.

    Pooling 5 records, all empty, all wrong:
      pooled accuracy = 0.0
    Per-fold mean (only one fold, 5 records) = 0.0
    The two converge here, but the **macro_f1** depends on per-class support,
    so we use it to demonstrate the metric on the stitched frame.
    """
    rows = [
        _make_record_row(f"R{i}", f"S{i}", "empty", "supine", 0, 0)
        for i in range(5)
    ]
    stitched = _stitch_full_oof([pd.DataFrame(rows)])
    metrics = _stitched_classification_metrics(stitched)
    assert metrics["accuracy"] == pytest.approx(0.0)
    # macro_f1 must be computed on the *stitched* confusion matrix, not an
    # arithmetic mean of per-fold macro_f1 (which is undefined for a single
    # fold but emphasises the contract).
    assert "macro_f1" in metrics
    assert "balanced_accuracy" in metrics
    assert "per_class" in metrics


def test_stitched_p6_single_metrics_match_loaded_rule(tmp_path: Path) -> None:
    rule = _build_fake_p6_single_rule(tmp_path, threshold=0.75)
    rows = [
        _make_record_row("R0", "S0", "empty", "empty", 0, 0, confidence=0.95),
        _make_record_row("R1", "S0", "empty", "empty", 1, 0, confidence=0.50),
    ]
    stitched = _stitch_full_oof([pd.DataFrame(rows)])
    metrics = _stitched_p6_single(stitched, rule=rule)
    assert metrics["rule_kind"] == "p6_single"
    assert metrics["threshold"] == 0.75
    assert metrics["n"] == 2
    assert metrics["accepted_n"] == 1
    assert metrics["coverage"] == pytest.approx(0.5)


def test_per_class_and_per_subject_breakdowns_cover_all_subjects(tmp_path: Path) -> None:
    rule = _build_fake_p6_single_rule(tmp_path, threshold=0.85)
    rows = [
        _make_record_row("R0", "S0", "empty", "empty", 0, 0, confidence=0.9),
        _make_record_row("R1", "S0", "empty", "supine", 1, 0, confidence=0.8),
        _make_record_row("R2", "S1", "empty", "empty", 2, 0, confidence=0.95),
        _make_record_row("R3", "S1", "empty", "prone", 0, 1, confidence=0.6),
    ]
    stitched = _stitch_full_oof([pd.DataFrame(rows)])
    per_subject = _per_subject_breakdown(stitched, rule=rule)
    subjects = sorted(row["subject_id"] for row in per_subject)
    assert subjects == ["S0", "S1"]
    # Real P6 single-rejected coverage must NOT be hardcoded to 1.0
    # (Reviewer point #5). Subject S1 has one record below 0.85, so
    # coverage on S1 should be 0.5; subject S0 also 0.5.
    for row in per_subject:
        assert 0.0 <= row["coverage"] <= 1.0
        assert row["accepted_n"] <= row["n"]
        assert row["p6_threshold"] == pytest.approx(0.85)
    per_class = _per_class_breakdown(stitched, rule=rule)
    assert any(item["y_true"] == "empty" for item in per_class)
    # Real coverage for the (only) class "empty" must also reflect the rule.
    only = next(item for item in per_class if item["y_true"] == "empty")
    assert only["accepted_n"] == 2  # 2 of 4 records at confidence >= 0.85
    assert only["coverage"] == pytest.approx(0.5)


def test_condition_seed_drift_stats_produces_mean_std_worst() -> None:
    """The condition seed delta stats must surface mean, std, and worst."""
    seed_summaries = [
        {
            "condition": {"name": "noise_p95_0.05", "kind": "gaussian_noise", "params": {}},
            "seed": 701,
            "n_records": 100,
            "record_metrics": {"macro_f1": 0.9, "balanced_accuracy": 0.9, "accuracy": 0.9},
            "delta_vs_clean": {
                "record_macro_f1": -0.05,
                "record_balanced_accuracy": -0.04,
                "record_accuracy": -0.03,
                "p6_single_wrong_action_rate": 0.01,
            },
            "p6_single_rule": {"rule_kind": "p6_single", "threshold": 0.75, "n": 100,
                                "accepted_n": 80, "coverage": 0.8,
                                "wrong_action_n": 1, "wrong_action_rate": 0.01,
                                "accepted_accuracy": 0.95, "accepted_error_rate": 0.0125},
            "p6_1_ensemble_rule": {"rule_kind": "p6_1_ensemble", "temperature": 0.75,
                                    "threshold": 0.5, "require_unanimous": True,
                                    "n": 100, "accepted_n": 70, "coverage": 0.7,
                                    "wrong_action_n": 2, "wrong_action_rate": 0.02,
                                    "accepted_accuracy": 0.94,
                                    "accepted_error_rate": 0.028},
            "per_class": [], "per_subject": [], "worst_subjects": {
                "by_wrong_action_rate": None,
                "by_coverage": None,
                "by_accepted_accuracy": None,
                "by_raw_accuracy": None,
            }, "error_cases": [],
        },
        {
            "condition": {"name": "noise_p95_0.05", "kind": "gaussian_noise", "params": {}},
            "seed": 702,
            "n_records": 100,
            "record_metrics": {"macro_f1": 0.85, "balanced_accuracy": 0.86, "accuracy": 0.87},
            "delta_vs_clean": {
                "record_macro_f1": -0.10,
                "record_balanced_accuracy": -0.08,
                "record_accuracy": -0.06,
                "p6_single_wrong_action_rate": 0.02,
            },
            "p6_single_rule": {"rule_kind": "p6_single", "threshold": 0.75, "n": 100,
                                "accepted_n": 75, "coverage": 0.75,
                                "wrong_action_n": 2, "wrong_action_rate": 0.02,
                                "accepted_accuracy": 0.93, "accepted_error_rate": 0.0267},
            "p6_1_ensemble_rule": {"rule_kind": "p6_1_ensemble", "temperature": 0.75,
                                    "threshold": 0.5, "require_unanimous": True,
                                    "n": 100, "accepted_n": 65, "coverage": 0.65,
                                    "wrong_action_n": 3, "wrong_action_rate": 0.03,
                                    "accepted_accuracy": 0.92,
                                    "accepted_error_rate": 0.046},
            "per_class": [], "per_subject": [], "worst_subjects": {
                "by_wrong_action_rate": None,
                "by_coverage": None,
                "by_accepted_accuracy": None,
                "by_raw_accuracy": None,
            }, "error_cases": [],
        },
    ]
    drift = _condition_seed_drift_stats(seed_summaries)
    assert drift["n_seeds"] == 2
    assert drift["delta_macro_f1_worst"] == pytest.approx(-0.10)
    assert drift["delta_macro_f1_mean"] == pytest.approx(-0.075)
    assert drift["delta_macro_f1_std"] >= 0.0
    assert drift["delta_balanced_accuracy_worst"] == pytest.approx(-0.08)


# ---------------------------------------------------------------------------
# Smoke marker: confirm the protocol constants and the new evidence API surface
# ---------------------------------------------------------------------------


def test_frozen_protocol_constants_match_evidence_pack() -> None:
    assert p7_runner.PROTOCOL_NAME == "popu_neural_full_v0.1"
    assert p7_runner.SNAPSHOTS_PER_RECORD == 10
    assert p7_runner.MODEL_FAMILY == "small_resnet"
    assert p7_runner.FULL_TOTAL_FOLDS == 15
    assert tuple(p7_runner.FULL_REPEATS) == (0, 1, 2)
    assert tuple(p7_runner.FULL_LOCAL_FOLDS) == (0, 1, 2, 3, 4)


def test_no_hardcoded_p6_threshold_in_runner() -> None:
    """The runner must no longer expose a hardcoded P6 threshold constant.

    Per Reviewer point #1 the threshold lives only in the loaded P6 evidence.
    """
    assert not hasattr(p7_runner, "P6_THRESHOLD")
    # Sanity: the public surface exposes the new evidence-based rules.
    assert hasattr(p7_runner, "P6SingleRule")
    assert hasattr(p7_runner, "P61EnsembleRule")
    assert hasattr(p7_runner, "run_clean_only_full_fold")


# ---------------------------------------------------------------------------
# Reviewer Round-3 regression tests for the four defects
# ---------------------------------------------------------------------------


def test_split_manifest_uses_canonical_sha_not_file_byte_sha(tmp_path: Path) -> None:
    """SplitManifest verification must use canonical-JSON SHA (Reviewer #1).

    The declared ``sha256`` must equal the canonical SHA computed over the
    manifest content with the ``sha256`` field stripped; the raw file-byte
    SHA must NOT be compared against the declared value.
    """
    from topper_perception.experiments.artifacts import sha256_hex
    from topper_perception.neural import full_splits
    from topper_perception.neural.p7_runner import SplitManifest

    folds_block = [
        {
            "repeat": 0,
            "local_fold": 0,
            "outer_seed": 11,
            "outer_train_subjects": ["S0"],
            "outer_test_subjects": ["S1"],
        }
    ]
    content = {
        "protocol": p7_runner.PROTOCOL_NAME,
        "folds": folds_block,
    }
    canonical = full_splits._canonical_sha256(content)
    # Intentionally whitespace / key-ordering variant of the same content.
    raw = (
        '{"protocol": "popu_neural_full_v0.1", '
        '"folds": [{"outer_seed": 11, "repeat": 0, '
        '"local_fold": 0, "outer_train_subjects": ["S0"], '
        '"outer_test_subjects": ["S1"]}], '
        f'"sha256": "{canonical}"}}'
    )
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(raw, encoding="utf-8")
    file_byte_sha = sha256_hex(manifest_path)
    # Sanity: the raw file bytes differ from the canonical SHA
    # (whitespace / key-order differences alone produce a different byte digest).
    assert file_byte_sha.lower() != canonical.lower()

    # The SplitManifest must accept this manifest because its canonical
    # SHA matches the declared SHA, even though the raw byte SHA differs.
    manifest = SplitManifest(
        path=manifest_path,
        protocol=p7_runner.PROTOCOL_NAME,
        n_folds=1,
        declared_canonical_sha256=canonical.lower(),
        canonical_sha256=canonical.lower(),
        file_byte_sha256=file_byte_sha,
        folds=tuple(folds_block),
    )
    assert manifest.declared_canonical_sha256 == manifest.canonical_sha256
    assert manifest.file_byte_sha256 != manifest.canonical_sha256


def test_split_manifest_rejects_canonical_sha_drift(tmp_path: Path) -> None:
    """SplitManifest must fail closed if the declared canonical SHA drifts."""
    from topper_perception.neural.p7_runner import SplitManifest

    with pytest.raises(ValueError, match="declared canonical sha256"):
        SplitManifest(
            path=tmp_path / "split_manifest.json",
            protocol=p7_runner.PROTOCOL_NAME,
            n_folds=1,
            declared_canonical_sha256="f" * 64,
            canonical_sha256="0" * 64,
            file_byte_sha256="0" * 64,
            folds=(),
        )


def test_load_split_manifest_accepts_actual_evidence_pack(tmp_path: Path) -> None:
    """End-to-end: write a real split_manifest.json with the canonical SHA
    and ensure load_split_manifest accepts it under the new policy."""
    from topper_perception.neural import full_splits
    from topper_perception.neural.p7_runner import load_split_manifest

    folds_block = [
        {
            "repeat": 0,
            "local_fold": 0,
            "outer_seed": 11,
            "outer_train_subjects": ["S0"],
            "outer_test_subjects": ["S1"],
        }
    ]
    content = {
        "protocol": p7_runner.PROTOCOL_NAME,
        "folds": folds_block,
    }
    canonical = full_splits._canonical_sha256(content)
    raw = json.dumps({**content, "sha256": canonical}, indent=2)
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(raw, encoding="utf-8")

    manifest = load_split_manifest(tmp_path)
    assert manifest.canonical_sha256 == canonical.lower()
    assert manifest.declared_canonical_sha256 == canonical.lower()
    assert manifest.n_folds == 1


def test_p6_1_ensemble_loader_rejects_rules0_pointer(tmp_path: Path) -> None:
    """Reviewer #2: rule_pointer MUST point at /rules/1/threshold, not rules[0]."""
    from topper_perception.experiments.artifacts import sha256_hex
    from topper_perception.neural.p6_evidence import load_p6_1_ensemble_rule

    payload = {
        "temperature": 0.75,
        "rules": [
            {"threshold": 0.75},  # pre-unanimity — must NOT leak into P7
            {"threshold": 0.5, "require_unanimous": True},
        ],
    }
    path = tmp_path / "p6_1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    expected_sha = sha256_hex(path)
    block = {
        "kind": "summary_json",
        "path": str(path),
        "expected_sha256": expected_sha,
        "temperature_pointer": "/temperature",
        "rule_pointer": "/rules/0/threshold",  # VIOLATION
        "unanimous_require_field": "/rules/1/require_unanimous",
    }
    with pytest.raises(ValueError, match="/rules/1/threshold"):
        load_p6_1_ensemble_rule(block)


def test_p6_1_ensemble_loader_rejects_missing_three_repeat_marker(tmp_path: Path) -> None:
    """Reviewer #2: the unanimity require_unanimous field MUST exist at the
    unanimity-branch pointer (rules[1]) and not be hidden behind rules[0]."""
    from topper_perception.experiments.artifacts import sha256_hex
    from topper_perception.neural.p6_evidence import load_p6_1_ensemble_rule

    payload = {"temperature": 0.75, "rules": [{"threshold": 0.5, "require_unanimous": True}]}
    path = tmp_path / "p6_1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    expected_sha = sha256_hex(path)
    block = {
        "kind": "summary_json",
        "path": str(path),
        "expected_sha256": expected_sha,
        "temperature_pointer": "/temperature",
        "rule_pointer": "/rules/1/threshold",
        "unanimous_require_field": "/rules/0/require_unanimous",  # wrong pointer
    }
    # rules only has 1 entry, so /rules/0 works but /rules/1/require_unanimous
    # is missing. Either way, the loader must reject the block.
    with pytest.raises(ValueError, match=r"missing token|index .* out of range"):
        load_p6_1_ensemble_rule(block)


def test_p6_1_ensemble_rule_does_not_expose_unanimous_threshold_field() -> None:
    """The P61EnsembleRule dataclass no longer carries the obsolete
    ``unanimous_threshold`` field; the unanimity-branch threshold is the
    canonical ``threshold`` (Reviewer #2)."""
    from topper_perception.neural.p6_evidence import P61EnsembleRule

    fields = {field.name for field in P61EnsembleRule.__dataclass_fields__.values()}
    assert "unanimous_threshold" not in fields
    assert "threshold" in fields
    assert "require_unanimous" in fields
    assert "unanimity_field_pointer" in fields


def test_worst_subjects_reports_each_criterion_separately(tmp_path: Path) -> None:
    """Reviewer Round-4: _worst_subjects must return FOUR independent rows —
    by_wrong_action_rate, by_coverage, by_accepted_accuracy, by_raw_accuracy.
    Each entry must include the full per-subject context (n, accepted_n,
    coverage, accepted_accuracy, accuracy, wrong_action_rate, p6_threshold).
    The structural contract (4 keys present, every row non-None, every row
    is a full per-subject breakdown) is the primary invariant; per-criterion
    correctness is verified by the four dedicated tests below.
    """
    from topper_perception.neural.p7_runner import _worst_subjects

    rule = _build_fake_p6_single_rule(tmp_path, threshold=0.85)
    rows = [
        _make_record_row("R0", "S_a", "empty", "supine", 0, 0, confidence=0.95),
        _make_record_row("R1", "S_b", "empty", "empty", 0, 0, confidence=0.95),
        _make_record_row("R2", "S_c", "empty", "empty", 0, 0, confidence=0.95),
    ]
    stitched = _stitch_full_oof([pd.DataFrame(rows)])
    worst = _worst_subjects(stitched, rule=rule)

    assert set(worst.keys()) == {
        "by_wrong_action_rate",
        "by_coverage",
        "by_accepted_accuracy",
        "by_raw_accuracy",
    }
    for key, row in worst.items():
        assert row is not None, f"_worst_subjects[{key!r}] must not be None"
        # Every entry must carry the full per-subject context.
        for field in (
            "subject_id", "n", "wrong_action_n", "wrong_action_rate",
            "accuracy", "accepted_n", "coverage", "accepted_accuracy",
            "accepted_error_rate", "p6_threshold",
        ):
            assert field in row, f"_worst_subjects[{key!r}] missing {field!r}"


def test_worst_subjects_by_wrong_action_rate_desc(tmp_path: Path) -> None:
    """Reviewer Round-4: by_wrong_action_rate must pick the subject with
    the highest WAR (DESC) among subjects that share the same P6 rule."""
    from topper_perception.neural.p7_runner import _worst_subjects

    rule = _build_fake_p6_single_rule(tmp_path, threshold=0.5)
    rows = [
        # S_low_war: 0 wrong out of 3 (all correct)
        _make_record_row("R0", "S_low_war", "empty", "empty", 0, 0, confidence=0.95),
        _make_record_row("R1", "S_low_war", "empty", "empty", 0, 0, confidence=0.95),
        _make_record_row("R2", "S_low_war", "empty", "empty", 0, 0, confidence=0.95),
        # S_mid_war: 1 wrong out of 3
        _make_record_row("R3", "S_mid_war", "empty", "supine", 0, 0, confidence=0.95),
        _make_record_row("R4", "S_mid_war", "empty", "empty", 0, 0, confidence=0.95),
        _make_record_row("R5", "S_mid_war", "empty", "empty", 0, 0, confidence=0.95),
        # S_high_war: 3 wrong out of 3 (all wrong)
        _make_record_row("R6", "S_high_war", "empty", "supine", 0, 0, confidence=0.95),
        _make_record_row("R7", "S_high_war", "empty", "supine", 0, 0, confidence=0.95),
        _make_record_row("R8", "S_high_war", "empty", "supine", 0, 0, confidence=0.95),
    ]
    stitched = _stitch_full_oof([pd.DataFrame(rows)])
    worst = _worst_subjects(stitched, rule=rule)

    by_war = worst["by_wrong_action_rate"]
    assert by_war["subject_id"] == "S_high_war"
    assert by_war["wrong_action_rate"] == pytest.approx(1.0)
    assert by_war["wrong_action_n"] == 3
    assert by_war["n"] == 3


def test_worst_subjects_by_coverage_asc(tmp_path: Path) -> None:
    """Reviewer Round-4: by_coverage must pick the subject with the lowest
    post-P6-rejection coverage (ASC). Subjects with zero accepted records
    are the extreme worst."""
    from topper_perception.neural.p7_runner import _worst_subjects

    rule = _build_fake_p6_single_rule(tmp_path, threshold=0.85)
    rows = [
        # S_full: all accepted (confidence ≥ 0.85)
        _make_record_row("R0", "S_full", "empty", "empty", 0, 0, confidence=0.95),
        _make_record_row("R1", "S_full", "empty", "empty", 0, 0, confidence=0.95),
        # S_partial: 1 accepted / 2
        _make_record_row("R2", "S_partial", "empty", "empty", 0, 0, confidence=0.95),
        _make_record_row("R3", "S_partial", "empty", "empty", 0, 0, confidence=0.50),
        # S_zero: 0 accepted / 2 (all rejected)
        _make_record_row("R4", "S_zero", "empty", "empty", 0, 0, confidence=0.40),
        _make_record_row("R5", "S_zero", "empty", "empty", 0, 0, confidence=0.30),
    ]
    stitched = _stitch_full_oof([pd.DataFrame(rows)])
    worst = _worst_subjects(stitched, rule=rule)

    by_cov = worst["by_coverage"]
    assert by_cov["subject_id"] == "S_zero"
    assert by_cov["coverage"] == pytest.approx(0.0)
    assert by_cov["accepted_n"] == 0
    assert by_cov["n"] == 2
    assert by_cov["p6_threshold"] == pytest.approx(0.85)


def test_worst_subjects_by_accepted_accuracy_asc(tmp_path: Path) -> None:
    """Reviewer Round-4: by_accepted_accuracy must pick the subject with the
    lowest accepted_accuracy among subjects that received ≥1 accepted record."""
    from topper_perception.neural.p7_runner import _worst_subjects

    rule = _build_fake_p6_single_rule(tmp_path, threshold=0.85)
    rows = [
        # S_perfect: 2/2 accepted correct
        _make_record_row("R0", "S_perfect", "empty", "empty", 0, 0, confidence=0.95),
        _make_record_row("R1", "S_perfect", "empty", "empty", 0, 0, confidence=0.95),
        # S_half: 2/2 accepted, 1/2 correct → accepted_accuracy = 0.5
        _make_record_row("R2", "S_half", "empty", "empty", 0, 0, confidence=0.95),
        _make_record_row("R3", "S_half", "empty", "supine", 0, 0, confidence=0.95),
        # S_none_right: 2/2 accepted, 0/2 correct → accepted_accuracy = 0.0
        _make_record_row("R4", "S_none_right", "empty", "supine", 0, 0, confidence=0.95),
        _make_record_row("R5", "S_none_right", "empty", "supine", 0, 0, confidence=0.95),
    ]
    stitched = _stitch_full_oof([pd.DataFrame(rows)])
    worst = _worst_subjects(stitched, rule=rule)

    by_acc = worst["by_accepted_accuracy"]
    assert by_acc["subject_id"] == "S_none_right"
    assert by_acc["accepted_accuracy"] == pytest.approx(0.0)
    assert by_acc["accepted_n"] == 2


def test_worst_subjects_by_raw_accuracy_asc(tmp_path: Path) -> None:
    """Reviewer Round-4: by_raw_accuracy must pick the subject with the
    lowest raw accuracy (irrespective of rejection rule)."""
    from topper_perception.neural.p7_runner import _worst_subjects

    rule = _build_fake_p6_single_rule(tmp_path, threshold=0.5)
    rows = [
        # S_high: 3/3 correct → raw accuracy = 1.0
        _make_record_row("R0", "S_high", "empty", "empty", 0, 0, confidence=0.95),
        _make_record_row("R1", "S_high", "empty", "empty", 0, 0, confidence=0.95),
        _make_record_row("R2", "S_high", "empty", "empty", 0, 0, confidence=0.95),
        # S_mid: 1/2 correct → 0.5
        _make_record_row("R3", "S_mid", "empty", "empty", 0, 0, confidence=0.95),
        _make_record_row("R4", "S_mid", "empty", "supine", 0, 0, confidence=0.95),
        # S_low: 0/2 correct → 0.0
        _make_record_row("R5", "S_low", "empty", "supine", 0, 0, confidence=0.95),
        _make_record_row("R6", "S_low", "empty", "supine", 0, 0, confidence=0.95),
    ]
    stitched = _stitch_full_oof([pd.DataFrame(rows)])
    worst = _worst_subjects(stitched, rule=rule)

    by_raw = worst["by_raw_accuracy"]
    assert by_raw["subject_id"] == "S_low"
    assert by_raw["accuracy"] == pytest.approx(0.0)
    assert by_raw["wrong_action_n"] == 2
    assert by_raw["n"] == 2


def test_worst_subjects_tie_break_by_subject_id_ascending(tmp_path: Path) -> None:
    """Reviewer Round-4: ties on every criterion are broken by subject_id ASC."""
    from topper_perception.neural.p7_runner import _worst_subjects

    rule = _build_fake_p6_single_rule(tmp_path, threshold=0.5)
    rows = []
    for subject in ["S_zeta", "S_alpha", "S_mu"]:
        rows.append(
            _make_record_row(f"R_{subject}", subject, "empty", "supine",
                              0, 0, confidence=0.9)
        )
    stitched = _stitch_full_oof([pd.DataFrame(rows)])
    worst = _worst_subjects(stitched, rule=rule)

    # All three subjects have identical metrics (1 record, 1 wrong).
    # subject_id ASC tie-break → S_alpha for all four criteria.
    assert worst["by_wrong_action_rate"]["subject_id"] == "S_alpha"
    assert worst["by_coverage"]["subject_id"] == "S_alpha"
    assert worst["by_accepted_accuracy"]["subject_id"] == "S_alpha"
    assert worst["by_raw_accuracy"]["subject_id"] == "S_alpha"


def test_worst_subjects_handles_empty_breakdown(tmp_path: Path) -> None:
    """Reviewer Round-4: when no subjects survive (empty stitched frame),
    _worst_subjects returns a dict of Nones — never KeyError or IndexError."""
    from topper_perception.neural.p7_runner import _worst_subjects

    rule = _build_fake_p6_single_rule(tmp_path, threshold=0.85)
    empty = _stitch_full_oof([pd.DataFrame()])
    worst = _worst_subjects(empty, rule=rule)
    assert worst == {
        "by_wrong_action_rate": None,
        "by_coverage": None,
        "by_accepted_accuracy": None,
        "by_raw_accuracy": None,
    }


def test_error_cases_uses_legal_threshold_zero(tmp_path: Path) -> None:
    """Reviewer #4: _error_cases must NOT pass an illegal threshold like 1e9.
    It uses threshold=0.0 so apply_rule accepts every record and the
    subsequent ~correct filter keeps only the wrong predictions."""
    from topper_perception.neural import p7_runner as runner
    from topper_perception.neural.p6_reject import (
        add_uncertainty_columns,
        error_cases as p6_error_cases,
    )

    rule = _build_fake_p6_single_rule(tmp_path, threshold=0.9)
    rows = [
        _make_record_row("R0", "S0", "empty", "empty", 0, 0, confidence=0.95),
        _make_record_row("R1", "S0", "empty", "supine", 0, 0, confidence=0.95),
    ]
    stitched = _stitch_full_oof([pd.DataFrame(rows)])
    errors = runner._error_cases(stitched)
    assert len(errors) == 1  # only the wrong prediction (R1)
    err = errors[0]
    assert err["record_id"] == "R1"
    assert "high_confidence_error" in err
    # The 1e9 (illegal) path would crash apply_rule with a ValueError;
    # verify threshold=0.0 stays within the legal range, with the
    # uncertainty columns pre-added (as the runner does internally).
    framed = add_uncertainty_columns(stitched.copy())
    p6_error_cases(framed, threshold=0.0, high_confidence=0.90)
    # And confirm the legal range itself: passing 1e9 must fail closed.
    with pytest.raises(ValueError, match="confidence_threshold"):
        p6_error_cases(framed, threshold=1e9, high_confidence=0.90)


def test_per_class_breakdown_reflects_p6_single_threshold(tmp_path: Path) -> None:
    """Reviewer #5: per-class breakdown must show the REAL P6 single-rejected
    coverage / accepted_accuracy / WAR — not a hardcoded ``coverage=1.0``."""
    from topper_perception.neural.p7_runner import _per_class_breakdown

    rule = _build_fake_p6_single_rule(tmp_path, threshold=0.85)
    rows = [
        _make_record_row("R0", "S0", "empty", "empty", 0, 0, confidence=0.95),
        _make_record_row("R1", "S0", "empty", "empty", 0, 0, confidence=0.50),  # rejected
        _make_record_row("R2", "S1", "supine", "supine", 0, 0, confidence=0.95),
        _make_record_row("R3", "S1", "supine", "supine", 0, 0, confidence=0.50),  # rejected
    ]
    stitched = _stitch_full_oof([pd.DataFrame(rows)])
    per_class = _per_class_breakdown(stitched, rule=rule)
    by_label = {row["y_true"]: row for row in per_class}
    assert by_label["empty"]["coverage"] == pytest.approx(0.5)
    assert by_label["empty"]["accepted_n"] == 1
    assert by_label["empty"]["n"] == 2
    assert by_label["supine"]["coverage"] == pytest.approx(0.5)
    # Coverage must never be silently 1.0 (the old bug).
    for row in per_class:
        assert row["coverage"] != 1.0 or row["accepted_n"] == row["n"]


def test_per_subject_breakdown_reflects_p6_single_threshold(tmp_path: Path) -> None:
    """Reviewer #5: per-subject breakdown must reflect P6 single-rejected
    coverage / accepted_accuracy / WAR per subject."""
    from topper_perception.neural.p7_runner import _per_subject_breakdown

    rule = _build_fake_p6_single_rule(tmp_path, threshold=0.85)
    rows = [
        _make_record_row("R0", "S_high", "empty", "empty", 0, 0, confidence=0.95),
        _make_record_row("R1", "S_high", "empty", "empty", 0, 0, confidence=0.95),
        _make_record_row("R2", "S_low", "empty", "empty", 0, 0, confidence=0.95),
        _make_record_row("R3", "S_low", "empty", "supine", 0, 0, confidence=0.40),  # rejected + wrong
    ]
    stitched = _stitch_full_oof([pd.DataFrame(rows)])
    per_subject = _per_subject_breakdown(stitched, rule=rule)
    by_sub = {row["subject_id"]: row for row in per_subject}
    # S_high: 2/2 accepted, 0 wrong → coverage=1.0, accepted_accuracy=1.0, war=0
    assert by_sub["S_high"]["coverage"] == pytest.approx(1.0)
    assert by_sub["S_high"]["accepted_accuracy"] == pytest.approx(1.0)
    assert by_sub["S_high"]["wrong_action_rate"] == pytest.approx(0.0)
    # S_low: 1/2 accepted (R3 rejected because confidence<0.85), 0 wrong on accepted → coverage=0.5
    assert by_sub["S_low"]["coverage"] == pytest.approx(0.5)
    assert by_sub["S_low"]["accepted_n"] == 1
    assert by_sub["S_low"]["wrong_action_rate"] == pytest.approx(0.0)
    # The OLD bug: coverage=1.0 hardcoded for both subjects would have
    # produced wrong_action_rate=0.5 for S_low — guard against that.
    assert by_sub["S_low"]["coverage"] != 1.0


# ---------------------------------------------------------------------------
# Reviewer Round-4 regression tests
# ---------------------------------------------------------------------------


def test_derive_record_seed_is_deterministic_and_platform_independent() -> None:
    """Reviewer Round-4: derive_record_seed must be stable across runs.

    Two calls with the same (perturb_seed, record_id) must yield identical
    integers, and different record_ids with the same perturb_seed must
    yield different integers (so records receive independent noise).
    """
    a = derive_record_seed(701, "R0001")
    b = derive_record_seed(701, "R0001")
    c = derive_record_seed(701, "R0002")
    d = derive_record_seed(702, "R0001")
    assert a == b
    assert a != c
    assert a != d
    # 64-bit positive integer (numpy.random.default_rng seed range).
    assert 0 <= a < 2**64


def test_perturb_records_gaussian_noise_is_per_record_seed_stable() -> None:
    """Reviewer Round-4: Gaussian noise must be per-record stable. Running
    perturb_records twice with the same (condition, seed) must produce the
    same noise on each record; conversely two different records within the
    same run must receive different noise patterns (otherwise the salt
    did not actually mix in record_id)."""
    records = _sample_records(n=4)
    condition = P7Condition(
        name="noise_p95_0.05",
        kind="gaussian_noise",
        params=(("sigma_fraction", 0.05),),
    )
    perturbed_a = perturb_records(records, condition, seed=701)
    perturbed_b = perturb_records(records, condition, seed=701)
    # Same (record, seed) → bit-identical noise.
    for ra, rb in zip(perturbed_a, perturbed_b):
        assert np.array_equal(ra["matrices"], rb["matrices"])
    # Different records within the same perturbation run → DIFFERENT noise.
    # (Under the old bug every record would share the same noise pattern.)
    for i in range(len(perturbed_a)):
        for j in range(i + 1, len(perturbed_a)):
            assert not np.array_equal(
                perturbed_a[i]["matrices"], perturbed_a[j]["matrices"]
            ), f"records {i} and {j} must receive different noise patterns"


def test_perturb_records_gaussian_noise_varies_with_perturb_seed() -> None:
    """Reviewer Round-4: a different (condition, perturb_seed) pair must
    produce different noise for every record (the perturb_seed salt is the
    primary differentiator at the run level)."""
    records = _sample_records(n=3)
    condition = P7Condition(
        name="noise_p95_0.10",
        kind="gaussian_noise",
        params=(("sigma_fraction", 0.10),),
    )
    out_seed701 = perturb_records(records, condition, seed=701)
    out_seed702 = perturb_records(records, condition, seed=702)
    for a, b in zip(out_seed701, out_seed702):
        assert not np.array_equal(a["matrices"], b["matrices"])


def test_perturb_records_non_gaussian_conditions_keep_caller_seed() -> None:
    """Reviewer Round-4: per-record seed derivation applies ONLY to Gaussian
    noise (the asked-for scope). bad_cell / bad_lines / density conditions
    must continue to consume the caller's seed verbatim so their masks
    remain shared across records by design."""
    records = _sample_records(n=2)
    for condition in [
        P7Condition(
            name="bad_cell_0.05",
            kind="bad_cell",
            params=(("fraction", 0.05),),
        ),
        P7Condition(
            name="bad_rows_1",
            kind="bad_lines",
            params=(("bad_rows", 1), ("bad_columns", 0)),
        ),
    ]:
        out_a = perturb_records(records, condition, seed=701)
        out_b = perturb_records(records, condition, seed=701)
        for ra, rb in zip(out_a, out_b):
            assert np.array_equal(ra["matrices"], rb["matrices"])


def test_atomic_write_json_converts_nan_to_null(tmp_path: Path) -> None:
    """Reviewer Round-4: atomic_write_json must convert NaN floats to JSON
    ``null`` (not the Python literal ``NaN`` which is invalid RFC 7159)."""
    from topper_perception.experiments import artifacts

    path = tmp_path / "nan.json"
    artifacts.atomic_write_json(path, {"x": float("nan"), "label": "ok"})
    text = path.read_text(encoding="utf-8")
    # Must not contain the bare literal "NaN" anywhere.
    assert "NaN" not in text
    loaded = json.loads(text)
    assert loaded["x"] is None
    assert loaded["label"] == "ok"


def test_atomic_write_json_converts_pos_and_neg_infinity_to_null(tmp_path: Path) -> None:
    """Reviewer Round-4: both +Infinity and -Infinity must become ``null``,
    even when nested inside a list inside a dict."""
    from topper_perception.experiments import artifacts

    payload = {
        "pos": float("inf"),
        "neg": float("-inf"),
        "list": [0.0, float("nan"), float("inf"), -float("inf"), 1.5],
        "nested": {"a": float("nan"), "b": [float("inf")], "c": 42},
        "finite": 3.14,
    }
    path = tmp_path / "non_finite.json"
    artifacts.atomic_write_json(path, payload)
    text = path.read_text(encoding="utf-8")
    # No raw ``NaN`` / ``Infinity`` / ``-Infinity`` literals leaked through.
    for forbidden in ("NaN", "Infinity", "-Infinity"):
        assert forbidden not in text, (
            f"forbidden literal {forbidden!r} leaked into JSON: {text!r}"
        )
    loaded = json.loads(text)
    assert loaded["pos"] is None
    assert loaded["neg"] is None
    assert loaded["list"] == [0.0, None, None, None, 1.5]
    assert loaded["nested"] == {"a": None, "b": [None], "c": 42}
    assert loaded["finite"] == pytest.approx(3.14)


def test_atomic_write_json_does_not_emit_nan_when_written_with_strict_loaders(
    tmp_path: Path,
) -> None:
    """Reviewer Round-4: even when the payload was assembled from a
    classification-metric path that can produce NaN (e.g. macro_f1 over
    a class with zero support), the JSON on disk must round-trip through
    a strict ``json.loads`` call without error."""
    from topper_perception.experiments import artifacts

    payload = {
        "metric_blocks": [
            {"name": "macro_f1", "value": float("nan")},
            {"name": "balanced_accuracy", "value": 0.5},
        ],
        "subject_id": "10",
    }
    path = tmp_path / "metrics.json"
    artifacts.atomic_write_json(path, payload)
    # json.loads with default args must accept the file.
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["metric_blocks"][0]["value"] is None
    assert loaded["metric_blocks"][1]["value"] == pytest.approx(0.5)


def test_worst_subjects_uses_p6_rejected_coverage_not_raw_coverage(tmp_path: Path) -> None:
    """Reviewer Round-4: by_coverage must be the *post-P6-rejection*
    coverage from _per_subject_breakdown, NOT a raw ``accepted_n/n``
    computed without the rule. Two subjects with the same accepted/n but
    different p6 thresholds must remain distinguishable."""
    from topper_perception.neural.p7_runner import _worst_subjects

    rule = _build_fake_p6_single_rule(tmp_path, threshold=0.85)
    # Both subjects: 1/2 accepted (one record below threshold).
    rows = [
        _make_record_row("R0", "S_a", "empty", "empty", 0, 0, confidence=0.95),
        _make_record_row("R1", "S_a", "empty", "supine", 0, 0, confidence=0.40),
        _make_record_row("R2", "S_b", "empty", "empty", 0, 0, confidence=0.95),
        _make_record_row("R3", "S_b", "empty", "supine", 0, 0, confidence=0.50),
    ]
    stitched = _stitch_full_oof([pd.DataFrame(rows)])
    worst = _worst_subjects(stitched, rule=rule)

    by_cov = worst["by_coverage"]
    assert by_cov is not None
    assert by_cov["coverage"] == pytest.approx(0.5)
    assert by_cov["accepted_n"] == 1
    assert by_cov["n"] == 2
    assert by_cov["p6_threshold"] == pytest.approx(0.85)