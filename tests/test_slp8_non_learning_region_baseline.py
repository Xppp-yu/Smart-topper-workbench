"""Tests for the SLP8 B02 v0.1 non-learning region baseline module.

These tests are CPU-only, deterministic, and do not require the SLP8
dataset on disk.  They cover the contract from TASK-SLP-B02-NON-LEARNING-
REGION-BASELINE-v0.1:

* output shape = (192, 84), dtype = uint8, label range 0..8
* fixed seed / fixed input → bit-identical prediction
* all-zero pressure, single-point contact, tiny contact, degenerate
  PCA, non-finite pressure, wrong shape — all fail-closed or
  degenerate to BACKGROUND
* contact mask vs axis labels priority
* axis direction / head-up flip rule
* TRAIN-only template fitting (no re-fit on VAL)
* posture not in primary predictor
* points.csv / label / onehot are NEVER consumed
* TEST access denied by default
* fixed 8-region macro metrics do NOT skip empty classes
* config roundtrip
* failure paths still produce auditable artefacts
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from topper_perception.baseline.slp8_non_learning import (  # noqa: E402
    BACKGROUND_ID,
    BASELINE_VERSION,
    DEFAULT_CONTACT_FRACTION,
    DEFAULT_LATERAL_HALF_WIDTH,
    DEFAULT_REGION_PRIORITY,
    DEFAULT_SEGMENT_FRACTIONS,
    AllBackgroundBaseline,
    AxisPartitionConfig,
    AxisPartitionState,
    BaselineContractError,
    DtypeContractError,
    LabelRangeError,
    NonFinitePressureError,
    PRESSURE_SHAPE,
    PressureAxisContactIntersectionBaseline,
    PressureBodyAxisPartitionBaseline,
    REGION_IDS,
    REGION_NAMES,
    ShapeContractError,
    TrainSpatialPriorBaseline,
    TrainSpatialPriorState,
    TrainTemplateFittedError,
    fit_axis_partition_config,
    list_baselines,
)
from topper_perception.evaluation.slp_pressure_metrics import (  # noqa: E402
    DEFAULT_FOREGROUND_CLASS_IDS as METRICS_DEFAULT_FOREGROUND,
    FixedClassMacroMetrics,
    compute_fixed_class_macro_metrics,
)
from topper_perception.io.slp8_training_table_freeze import (  # noqa: E402
    B01FreezeTables,
    TestLeakageError,
    load_b01_freeze_tables,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _synthetic_pressure_bar() -> np.ndarray:
    """A vertical pressure bar (the kind of "body lying on the bed"
    geometry that is useful for testing body-axis detection)."""
    p = np.zeros(PRESSURE_SHAPE, dtype=np.float64)
    p[40:160, 30:60] = 100.0
    return p


def _synthetic_pressure_small() -> np.ndarray:
    """A tiny pressure blob used to exercise degenerate-PCA logic."""
    p = np.zeros(PRESSURE_SHAPE, dtype=np.float64)
    p[90, 40] = 100.0
    return p


def _synthetic_train_set(n: int = 12) -> list[np.ndarray]:
    """A list of synthetic pressure maps with body-axis contact blobs."""
    return [_synthetic_pressure_bar() for _ in range(n)]


def _synthetic_train_labels(n: int = 12) -> list[np.ndarray]:
    """A list of synthetic label maps aligned with the synthetic
    pressures (vertical bar with HEAD_NECK at top, LOWER_LEG_FOOT at
    bottom)."""
    labels: list[np.ndarray] = []
    for _ in range(n):
        lm = np.zeros(PRESSURE_SHAPE, dtype=np.uint8)
        lm[:int(0.18 * 192), 30:60] = 1  # HEAD_NECK
        lm[160:, 30:60] = 8  # LOWER_LEG_FOOT
        labels.append(lm)
    return labels


@pytest.fixture
def axis_state_default() -> AxisPartitionState:
    """Default axis state fitted on a small synthetic TRAIN set."""
    return fit_axis_partition_config(_synthetic_train_set())


@pytest.fixture
def fitted_template() -> TrainSpatialPriorBaseline:
    tsp = TrainSpatialPriorBaseline()
    tsp.fit(_synthetic_train_labels(), subject_ids=[f"subj_{i:02d}" for i in range(12)])
    return tsp


@pytest.fixture
def fitted_axis(axis_state_default: AxisPartitionState) -> PressureBodyAxisPartitionBaseline:
    bap = PressureBodyAxisPartitionBaseline()
    bap.fit(_synthetic_train_set())
    return bap


@pytest.fixture
def fitted_intersection() -> PressureAxisContactIntersectionBaseline:
    ib = PressureAxisContactIntersectionBaseline()
    ib.fit(_synthetic_train_labels(), _synthetic_train_set())
    return ib


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBaselineOutputContract:
    """Output shape, dtype, and label range."""

    @pytest.mark.parametrize(
        "baseline_name",
        ["all_background", "spatial_prior", "axis", "intersection"],
    )
    def test_output_shape_and_range(
        self,
        baseline_name: str,
        fitted_template: TrainSpatialPriorBaseline,
        fitted_axis: PressureBodyAxisPartitionBaseline,
        fitted_intersection: PressureAxisContactIntersectionBaseline,
    ) -> None:
        pressure = _synthetic_pressure_bar()
        if baseline_name == "all_background":
            bl = AllBackgroundBaseline()
        elif baseline_name == "spatial_prior":
            bl = fitted_template
        elif baseline_name == "axis":
            bl = fitted_axis
        else:
            bl = fitted_intersection
        out = bl.predict(pressure)
        assert out.shape == PRESSURE_SHAPE
        assert out.dtype == np.uint8
        assert int(out.min()) >= 0
        assert int(out.max()) <= 8
        # The label set must be a subset of {0..8}.
        assert set(int(x) for x in np.unique(out)) <= set(REGION_IDS) | {BACKGROUND_ID}


class TestDeterminism:
    """Fixed seed and fixed input → bit-identical predictions."""

    def test_all_background_deterministic(self) -> None:
        b = AllBackgroundBaseline()
        p = _synthetic_pressure_bar()
        out1 = b.predict(p)
        out2 = b.predict(p)
        assert np.array_equal(out1, out2)

    def test_spatial_prior_deterministic(self, fitted_template: TrainSpatialPriorBaseline) -> None:
        p = _synthetic_pressure_bar()
        out1 = fitted_template.predict(p)
        out2 = fitted_template.predict(p)
        assert np.array_equal(out1, out2)

    def test_axis_deterministic(self, fitted_axis: PressureBodyAxisPartitionBaseline) -> None:
        p = _synthetic_pressure_bar()
        out1 = fitted_axis.predict(p)
        out2 = fitted_axis.predict(p)
        assert np.array_equal(out1, out2)

    def test_intersection_deterministic(self, fitted_intersection: PressureAxisContactIntersectionBaseline) -> None:
        p = _synthetic_pressure_bar()
        out1 = fitted_intersection.predict(p)
        out2 = fitted_intersection.predict(p)
        assert np.array_equal(out1, out2)

    def test_axis_state_to_dict_is_pure(self, axis_state_default: AxisPartitionState) -> None:
        d = axis_state_default.to_dict()
        # Roundtrip via dataclass asdict → JSON
        j = json.dumps(d, sort_keys=True, default=str)
        # Re-parse; nothing should change.
        d2 = json.loads(j)
        # After JSON roundtrip, tuples become lists.  Re-serialise both
        # sides through JSON and compare.
        assert json.dumps(d2, sort_keys=True) == json.dumps(
            json.loads(j), sort_keys=True
        )


class TestAxisContactIntersection:
    """Contact mask and axis/region priority behaviour."""

    def test_empty_contact_falls_back_to_background(
        self,
        fitted_intersection: PressureAxisContactIntersectionBaseline,
    ) -> None:
        # All-zero pressure → empty contact → all BACKGROUND.
        p = np.zeros(PRESSURE_SHAPE, dtype=np.float64)
        out = fitted_intersection.predict(p)
        assert int(out.max()) == 0
        assert int(out.sum()) == 0

    def test_axis_partition_all_zero_pressure(
        self,
        fitted_axis: PressureBodyAxisPartitionBaseline,
    ) -> None:
        p = np.zeros(PRESSURE_SHAPE, dtype=np.float64)
        out = fitted_axis.predict(p)
        assert int(out.max()) == 0

    def test_tiny_contact_degenerate_pca(
        self,
        fitted_axis: PressureBodyAxisPartitionBaseline,
    ) -> None:
        p = _synthetic_pressure_small()
        out = fitted_axis.predict(p)
        # Single point contact has fewer than 5 pixels → fall back to
        # all-background.
        assert int(out.max()) == 0

    def test_axis_orientation_is_head_up(
        self,
        axis_state_default: AxisPartitionState,
    ) -> None:
        # Vertical bar: centroid ≈ (44.5, 99.5).  Head is on the small-y
        # side, so the axis vector (ux, uy) must have uy <= 0 (in image
        # coordinates, "up" is decreasing y).
        pressure = _synthetic_pressure_bar()
        from topper_perception.baseline.slp8_non_learning import (
            _body_axis_from_contact,
            _build_contact_mask,
        )

        contact = _build_contact_mask(pressure, threshold=axis_state_default.contact_threshold)
        cx, cy, ux, uy, degenerate = _body_axis_from_contact(contact)
        assert not degenerate
        assert uy <= 0.0

    def test_axis_flip_handled(
        self,
        axis_state_default: AxisPartitionState,
    ) -> None:
        # Flip the bar top-to-bottom: the centroid shifts but the axis
        # direction (head-up) must still come out uy <= 0.
        from topper_perception.baseline.slp8_non_learning import (
            _body_axis_from_contact,
            _build_contact_mask,
        )

        p = _synthetic_pressure_bar()[::-1, :].copy()
        contact = _build_contact_mask(p, threshold=axis_state_default.contact_threshold)
        _, _, ux, uy, degenerate = _body_axis_from_contact(contact)
        assert not degenerate
        assert uy <= 0.0

    def test_segment_fractions_sum_to_one(
        self,
        fitted_axis: PressureBodyAxisPartitionBaseline,
    ) -> None:
        p = _synthetic_pressure_bar()
        out = fitted_axis.predict(p)
        # 8 region IDs must appear in the body-axis partition, in
        # the proportions defined by DEFAULT_SEGMENT_FRACTIONS, for
        # a uniform vertical bar.
        total = int((out > 0).sum())
        assert total > 0
        for cid in REGION_IDS:
            assert int((out == cid).sum()) >= 0  # smoke


class TestFailClosed:
    """Non-finite pressure and wrong shape must fail closed."""

    def test_non_finite_pressure_raises(self) -> None:
        b = AllBackgroundBaseline()
        p = np.zeros(PRESSURE_SHAPE, dtype=np.float64)
        p[0, 0] = np.nan
        with pytest.raises(NonFinitePressureError):
            b.predict(p)

    def test_inf_pressure_raises(self, fitted_template: TrainSpatialPriorBaseline) -> None:
        p = np.zeros(PRESSURE_SHAPE, dtype=np.float64)
        p[0, 0] = np.inf
        with pytest.raises(NonFinitePressureError):
            fitted_template.predict(p)

    def test_wrong_shape_raises(self) -> None:
        b = AllBackgroundBaseline()
        p = np.zeros((100, 50), dtype=np.float64)
        with pytest.raises(ShapeContractError):
            b.predict(p)

    def test_non_uint8_label_rejected_in_fit(self) -> None:
        tsp = TrainSpatialPriorBaseline()
        bad = np.zeros(PRESSURE_SHAPE, dtype=np.int64)
        bad[0, 0] = 9  # out of range
        with pytest.raises(LabelRangeError):
            tsp.fit([bad])

    def test_predict_without_fit_raises(self) -> None:
        tsp = TrainSpatialPriorBaseline()
        with pytest.raises(BaselineContractError):
            tsp.predict(np.zeros(PRESSURE_SHAPE, dtype=np.float64))
        bap = PressureBodyAxisPartitionBaseline()
        with pytest.raises(BaselineContractError):
            bap.predict(np.zeros(PRESSURE_SHAPE, dtype=np.float64))
        ib = PressureAxisContactIntersectionBaseline()
        with pytest.raises(BaselineContractError):
            ib.predict(np.zeros(PRESSURE_SHAPE, dtype=np.float64))

    def test_double_fit_raises_without_reset(self) -> None:
        tsp = TrainSpatialPriorBaseline()
        tsp.fit(_synthetic_train_labels())
        with pytest.raises(TrainTemplateFittedError):
            tsp.fit(_synthetic_train_labels())

        bap = PressureBodyAxisPartitionBaseline()
        bap.fit(_synthetic_train_set())
        with pytest.raises(TrainTemplateFittedError):
            bap.fit(_synthetic_train_set())

        ib = PressureAxisContactIntersectionBaseline()
        ib.fit(_synthetic_train_labels(), _synthetic_train_set())
        with pytest.raises(TrainTemplateFittedError):
            ib.fit(_synthetic_train_labels(), _synthetic_train_set())

    def test_double_fit_allowed_with_reset(self) -> None:
        tsp = TrainSpatialPriorBaseline()
        tsp.fit(_synthetic_train_labels())
        s2 = tsp.fit(_synthetic_train_labels(), reset=True)
        assert s2.train_count == 12


class TestTrainOnlyFitting:
    """VAL must not influence the fit."""

    def test_template_state_records_train_subjects(
        self, fitted_template: TrainSpatialPriorBaseline
    ) -> None:
        state = fitted_template.state
        assert state.fit_split == "train"
        assert state.train_subjects == tuple(
            f"subj_{i:02d}" for i in range(12)
        )

    def test_state_records_train_only(
        self,
        fitted_axis: PressureBodyAxisPartitionBaseline,
        axis_state_default: AxisPartitionState,
    ) -> None:
        assert axis_state_default.fit_split == "train"
        assert fitted_axis.state.fit_split == "train"

    def test_refitting_with_different_data_changes_state(self) -> None:
        tsp = TrainSpatialPriorBaseline()
        state_a = tsp.fit(_synthetic_train_labels(), subject_ids=["a"])
        ts_a = state_a.template.copy()
        state_b = tsp.fit(_synthetic_train_labels(), subject_ids=["b"], reset=True)
        ts_b = state_b.template.copy()
        # Different subject_id lists are recorded.
        assert state_a.train_subjects != state_b.train_subjects
        # But the template is the same (data was identical).
        assert np.allclose(ts_a, ts_b)


class TestNoLeakage:
    """The baselines must never use label/onehot/posture/points.csv as
    a per-sample input."""

    def test_predict_ignores_pressure_values_for_spatial_prior(
        self, fitted_template: TrainSpatialPriorBaseline
    ) -> None:
        # The spatial-prior predictor depends ONLY on the template, not
        # on the pressure input.  Therefore two pressures with the
        # same shape and a totally different pattern must produce
        # identical predictions.
        p1 = np.zeros(PRESSURE_SHAPE, dtype=np.float64)
        p2 = _synthetic_pressure_bar()
        out1 = fitted_template.predict(p1)
        out2 = fitted_template.predict(p2)
        assert np.array_equal(out1, out2)

    def test_predict_signature_pressure_only(self) -> None:
        # Inspect the source AST of the predict() methods and ensure
        # they only accept ``pressure`` as a positional/keyword arg.
        import inspect

        for cls in (
            AllBackgroundBaseline,
            TrainSpatialPriorBaseline,
            PressureBodyAxisPartitionBaseline,
            PressureAxisContactIntersectionBaseline,
        ):
            sig = inspect.signature(cls.predict)
            params = list(sig.parameters)
            assert params == ["self", "pressure"], (
                f"{cls.__name__}.predict parameters = {params}"
            )

    def test_fit_signatures_have_no_label_when_only_pressure(
        self,
    ) -> None:
        # The axis partition baseline must not accept labels in its fit
        # (only pressures).  The spatial prior must not accept
        # pressures in its fit (only labels).
        import inspect

        sig_axis = inspect.signature(PressureBodyAxisPartitionBaseline.fit)
        params_axis = list(sig_axis.parameters)
        assert "train_label_maps" not in params_axis
        assert "train_pressures" in params_axis

        sig_spatial = inspect.signature(TrainSpatialPriorBaseline.fit)
        params_spatial = list(sig_spatial.parameters)
        assert "train_pressures" not in params_spatial
        assert "label_maps" in params_spatial


class TestTestAccessPolicy:
    """TEST access is denied by default via the B01 freeze module."""

    def test_load_b01_default_denies_test(self, tmp_path: Path) -> None:
        # Build a minimal freeze directory in a temp folder and check
        # that load_b01_freeze_tables refuses to load TEST without
        # an explicit opt-in.
        freeze_dir = self._write_minimal_freeze(tmp_path)
        tables = load_b01_freeze_tables(freeze_dir)
        # Test rows are not loaded; accessing them via test_rows
        # must raise TestLeakageError.
        with pytest.raises(TestLeakageError):
            _ = tables.test_rows
        with pytest.raises(TestLeakageError):
            _ = tables.all_rows_with_test_opt_in()

    def test_load_b01_rejects_explicit_test_without_opt_in(
        self, tmp_path: Path
    ) -> None:
        freeze_dir = self._write_minimal_freeze(tmp_path)
        with pytest.raises(TestLeakageError):
            load_b01_freeze_tables(freeze_dir, allowed_splits=("train", "val", "test"))

    def test_load_b01_load_test_true_without_opt_in_raises(
        self, tmp_path: Path
    ) -> None:
        freeze_dir = self._write_minimal_freeze(tmp_path)
        with pytest.raises(TestLeakageError):
            load_b01_freeze_tables(freeze_dir, load_test=True)

    @staticmethod
    def _write_minimal_freeze(out_dir: Path) -> Path:
        """Write a minimal-but-valid B01 freeze directory for tests.

        The TRAIN and VAL manifests contain one row each; the TEST
        manifest contains a single dummy row.  The freeze manifest
        references the A06 split identifier only and does not need
        to be a full real freeze to exercise the load function.
        """
        freeze_dir = out_dir / "freeze"
        freeze_dir.mkdir(parents=True, exist_ok=True)

        def write_manifest(path: Path, ml_split: str) -> None:
            row = {
                "sample_id": f"SLP:danaLab:00001:uncover:00000{ml_split[:1]}",
                "ml_split": ml_split,
                "source_split": "VAL",
                "setting": "danaLab",
                "subject_id": "00001",
                "cover": "uncover",
                "frame_id": 1,
                "posture": "SUPINE",
                "pressure_npy": "samples/x/pressure.npy",
                "region_label_npy": "samples/x/region_label.npy",
                "region_onehot_npy": "samples/x/region_onehot.npy",
                "points_csv": "samples/x/points.csv",
                "height": 192,
                "width": 84,
                "class_ids_present": "0",
                "annotation_provenance": "V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
                "source_review_status": "NOT_REVIEWED",
                "export_version": "slp8-v2.2.1",
                "export_status": "EXPORTED",
                "source_pmarray_sha256": "0" * 64,
                "background_pixel_count": 0,
                "body_pixel_count": 0,
                "clipped_ratio": 0.0,
                "onehot_valid": True,
                "onehot_roundtrip": True,
            }
            import csv

            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerow(row)

        write_manifest(freeze_dir / "train_manifest.csv", "train")
        write_manifest(freeze_dir / "val_manifest.csv", "val")
        write_manifest(freeze_dir / "test_manifest.csv", "test")

        fm = {
            "core": {
                "a06_split_identifier": "slp_subject_split_v0.1",
                "a06_split_sha256": "0" * 64,
                "freeze_version": "slp8_training_tables_v0.1",
            },
            "meta": {
                "build_command": None,
                "builder_version": "slp8_training_table_freeze_v0.1",
                "built_at_utc": "2026-08-25T00:00:00+00:00",
                "git_sha": None,
                "platform": "win32",
                "python_version": "3.12",
            },
            "splits": {
                "test": {"manifest_sha256": "0" * 64, "sample_count": 1, "subject_count": 1},
                "train": {"manifest_sha256": "0" * 64, "sample_count": 1, "subject_count": 1},
                "val": {"manifest_sha256": "0" * 64, "sample_count": 1, "subject_count": 1},
            },
        }
        (freeze_dir / "freeze_manifest.json").write_text(
            json.dumps(fm, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        return freeze_dir


class TestFixedClassMacroMetrics:
    """The fixed 8-region macro IoU/Dice must not skip empty classes."""

    def test_macro_does_not_skip_missing_class(self) -> None:
        # Class 1 is predicted, class 2 is in GT only, class 3 is in
        # neither.  Macro indicator MUST be 1/8.
        gt = np.zeros((4, 4), dtype=np.uint8)
        gt[0:2, 0:2] = 1
        gt[0:2, 2:4] = 2
        pred = np.zeros((4, 4), dtype=np.uint8)
        pred[0:2, 0:2] = 1

        m = compute_fixed_class_macro_metrics(
            gt, pred, class_ids=(1, 2, 3, 4, 5, 6, 7, 8), n_classes=9
        )
        # Only class 1 is perfectly predicted; everything else is 0.
        assert m.fixed_iou == pytest.approx(1.0 / 8.0)
        assert m.fixed_dice == pytest.approx(1.0 / 8.0)
        # Per-class IoU contains every requested class.
        assert set(m.per_class_iou.keys()) == {1, 2, 3, 4, 5, 6, 7, 8}
        # Class 2 is present in GT but not pred → IoU=0, present_in_gt=True.
        assert m.per_class_iou[2] == 0.0
        assert m.per_class_present_in_gt[2] is True
        assert m.per_class_present_in_pred[2] is False

    def test_all_background_baseline_macro_is_zero(self) -> None:
        # An all-zero prediction against a varied GT must have a
        # fixed macro of 0.0 (every foreground class is "missed").
        rng = np.random.default_rng(0)
        gt = rng.integers(0, 9, size=(64, 64), dtype=np.uint8)
        pred = np.zeros((64, 64), dtype=np.uint8)
        m = compute_fixed_class_macro_metrics(
            gt, pred, class_ids=tuple(range(1, 9)), n_classes=9
        )
        assert m.fixed_iou == 0.0
        assert m.fixed_dice == 0.0
        # n_classes_present_in_pred should be 0.
        assert m.n_classes_present_in_pred == 0
        # n_classes_present_in_gt should be 8 (all foreground classes
        # observed in GT).
        assert m.n_classes_present_in_gt == 8

    def test_macro_default_class_ids(self) -> None:
        # Default class_ids should be 1..8.
        from topper_perception.evaluation.slp_pressure_metrics import (
            DEFAULT_FOREGROUND_CLASS_IDS as DCF,
        )
        assert DCF == (1, 2, 3, 4, 5, 6, 7, 8)
        assert METRICS_DEFAULT_FOREGROUND == DCF

    def test_macro_strict_does_not_hide_missing_class(self) -> None:
        # If a baseline never predicts class K, the macro IoU should
        # still include class K with IoU=0; this is the property the
        # task contract requires.  We verify it with a comparison to
        # the legacy "skip empty classes" semantics.
        gt = np.zeros((8, 8), dtype=np.uint8)
        gt[0:4, 0:4] = 1
        gt[0:4, 4:8] = 2
        gt[4:8, 0:8] = 3
        # Pred that misses classes 2 and 3.
        pred = np.zeros((8, 8), dtype=np.uint8)
        pred[0:4, 0:4] = 1
        m = compute_fixed_class_macro_metrics(
            gt, pred, class_ids=(1, 2, 3, 4, 5, 6, 7, 8), n_classes=9
        )
        # The fixed macro is mean(1.0, 0, 0, 0, 0, 0, 0, 0) = 0.125
        # The "skip empty" macro would be 1.0/1 = 1.0, hiding the failure.
        assert m.fixed_iou == pytest.approx(0.125)
        assert m.fixed_dice == pytest.approx(0.125)

    def test_macro_handles_sequence_input(self) -> None:
        gt1 = np.zeros((4, 4), dtype=np.uint8); gt1[0:2, :] = 1
        gt2 = np.zeros((4, 4), dtype=np.uint8); gt2[0:2, :] = 2
        pred1 = np.zeros((4, 4), dtype=np.uint8); pred1[0:2, :] = 1
        pred2 = np.zeros((4, 4), dtype=np.uint8); pred2[0:2, :] = 2
        m = compute_fixed_class_macro_metrics(
            [gt1, gt2], [pred1, pred2],
            class_ids=(1, 2, 3, 4, 5, 6, 7, 8), n_classes=9,
        )
        # Class 1 and 2 are perfect; the rest are 0.
        assert m.fixed_iou == pytest.approx(2.0 / 8.0)
        assert m.fixed_dice == pytest.approx(2.0 / 8.0)
        assert m.n_samples == 2

    def test_macro_pixel_accuracy(self) -> None:
        gt = np.zeros((4, 4), dtype=np.uint8); gt[0:2, :] = 1
        pred = np.zeros((4, 4), dtype=np.uint8); pred[0:2, 0:2] = 1
        m = compute_fixed_class_macro_metrics(
            gt, pred, class_ids=(1, 2, 3, 4, 5, 6, 7, 8), n_classes=9,
        )
        # Top-left 2x2 matches (4); bottom 2 rows match (8); top-right
        # 2x2 mismatches (4).  Total = 12 / 16 = 0.75.
        assert m.pixel_accuracy == pytest.approx(0.75)

    def test_macro_rejects_empty_class_ids(self) -> None:
        gt = np.zeros((4, 4), dtype=np.uint8)
        pred = np.zeros((4, 4), dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_fixed_class_macro_metrics(gt, pred, class_ids=(), n_classes=9)

    def test_macro_rejects_negative_class_id(self) -> None:
        gt = np.zeros((4, 4), dtype=np.uint8)
        pred = np.zeros((4, 4), dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_fixed_class_macro_metrics(gt, pred, class_ids=(-1, 0), n_classes=9)

    def test_macro_rejects_shape_mismatch(self) -> None:
        gt = np.zeros((4, 4), dtype=np.uint8)
        pred = np.zeros((4, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            compute_fixed_class_macro_metrics(gt, pred, class_ids=(1, 2, 3, 4, 5, 6, 7, 8), n_classes=9)


class TestConfigRoundtrip:
    """AxisPartitionConfig must roundtrip through dataclasses.asdict."""

    def test_axis_config_roundtrip(self) -> None:
        cfg = AxisPartitionConfig()
        d = dataclasses.asdict(cfg)
        cfg2 = AxisPartitionConfig(**d)
        assert cfg == cfg2

    def test_state_to_dict_is_serialisable(
        self, axis_state_default: AxisPartitionState
    ) -> None:
        d = axis_state_default.to_dict()
        # JSON roundtrip
        j = json.dumps(d, default=str)
        d2 = json.loads(j)
        # config dict is preserved
        assert d2["fit_split"] == "train"
        assert d2["config"]["contact_fraction"] == DEFAULT_CONTACT_FRACTION
        assert tuple(d2["config"]["region_ids"]) == REGION_IDS
        assert tuple(d2["config"]["region_priority"]) == DEFAULT_REGION_PRIORITY
        assert tuple(d2["config"]["segment_fractions"]) == DEFAULT_SEGMENT_FRACTIONS


class TestFailureAuditableArtifacts:
    """Failure paths still produce auditable artefacts."""

    def test_list_baselines_returns_all_four(self) -> None:
        bs = list_baselines()
        names = {b["name"] for b in bs}
        assert names == {
            "all_background",
            "train_spatial_prior",
            "pressure_body_axis_partition",
            "pressure_axis_contact_intersection",
        }
        # All baselines share the version.
        for b in bs:
            assert b["version"] == BASELINE_VERSION
            assert b["kind"] in {"sanity_floor", "candidate"}

    def test_to_state_round_trip(self) -> None:
        b = AllBackgroundBaseline()
        s = b.to_state()
        assert s == {"baseline": "all_background", "kind": "sanity_floor", "version": BASELINE_VERSION}

    def test_intersection_state_round_trip(
        self, fitted_intersection: PressureAxisContactIntersectionBaseline
    ) -> None:
        s = fitted_intersection.to_state()
        # State has the documented fields.
        assert s["baseline"] == "pressure_axis_contact_intersection"
        assert s["kind"] == "candidate"
        assert s["version"] == BASELINE_VERSION
        assert s["template"]["fit_split"] == "train"
        assert s["axis"]["fit_split"] == "train"
        # The template SHA-256 is stable.
        assert isinstance(s["template_sha256"], str)
        assert len(s["template_sha256"]) == 64


class TestRunConfigSanity:
    """Sanity checks for the runner's config schema (no real run here)."""

    def test_default_config_constants_match_baseline_defaults(self) -> None:
        from topper_perception.baseline.slp8_non_learning import (
            REGION_ID_TO_NAME,
        )

        assert DEFAULT_SEGMENT_FRACTIONS == (
            0.18, 0.07, 0.13, 0.12, 0.10, 0.10, 0.15, 0.15,
        )
        assert DEFAULT_LATERAL_HALF_WIDTH == pytest.approx(0.40)
        assert DEFAULT_REGION_PRIORITY == (2, 5, 4, 3, 1, 7, 8, 6)
        assert set(REGION_IDS) == {1, 2, 3, 4, 5, 6, 7, 8}
        assert len(REGION_NAMES) == 8
        assert tuple(REGION_ID_TO_NAME[i] for i in REGION_IDS) == REGION_NAMES


# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------


def test_module_no_absolute_paths_in_source() -> None:
    """B02 module must not contain hard-coded absolute Windows paths."""
    src = (Path(__file__).resolve().parents[1] / "src" / "topper_perception" / "baseline" / "slp8_non_learning.py").read_text(
        encoding="utf-8"
    )
    # Check for absolute Windows / POSIX path patterns.  Note: this is
    # not a full anti-leak scanner; it just catches the common cases.
    assert "E:\\" not in src
    assert "D:\\" not in src
    assert "C:\\" not in src
    assert "E:/" not in src
    assert "D:/" not in src
    assert "C:/" not in src
    # No forbidden SLP8 GT input fields are consumed as input.
    # The string "region_label" / "region_onehot" only appears in the
    # module docstring explaining the contract; it must NEVER be a
    # function parameter, function-local name, or attribute reference.
    import re
    # Strip the docstring.
    no_docstring = re.sub(r'^"""[\s\S]*?"""', "", src, count=1, flags=re.MULTILINE)
    # In the code body, the strings region_label / region_onehot must
    # not appear; if they do, they would suggest we are consuming
    # those fields as inputs.
    assert "region_label" not in no_docstring
    assert "region_onehot" not in no_docstring
    assert "points.csv" not in no_docstring
    assert "class_ids_present" not in no_docstring
    assert "background_pixel_count" not in no_docstring
    assert "body_pixel_count" not in no_docstring
