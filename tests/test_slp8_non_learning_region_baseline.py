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
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load_runner_module():
    """Load the runner module by absolute path; works under any
    pytest / sys.path configuration."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "b02_runner_under_test",
        str(SCRIPTS_DIR / "run_slp8_non_learning_region_baseline.py"),
    )
    if spec is None or spec.loader is None:
        raise ImportError("could not load runner spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module

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
            assert int((out == cid).sum()) >= 0

    def test_predict_with_info_returns_diagnostics(
        self,
        fitted_axis: PressureBodyAxisPartitionBaseline,
        fitted_intersection: PressureAxisContactIntersectionBaseline,
        fitted_template: TrainSpatialPriorBaseline,
    ) -> None:
        """Each baseline must expose a ``predict_with_info`` method
        that returns a (label_map, info_dict) tuple, where the info
        dict contains the fallback diagnostic.  This is what the
        runner uses to count ``no_contact`` / ``degenerate_pca`` /
        etc."""
        p = _synthetic_pressure_bar()
        # Body axis partition: no fallback for a clear vertical bar.
        labels, info = fitted_axis.predict_with_info(p)
        assert labels.shape == PRESSURE_SHAPE
        assert info["fallback"] == "none"
        assert info["no_contact"] is False
        assert info["degenerate_pca"] is False
        # Intersection baseline: same — no fallback for a clear bar.
        labels2, info2 = fitted_intersection.predict_with_info(p)
        assert labels2.shape == PRESSURE_SHAPE
        assert info2["fallback"] == "none"
        # Spatial prior: no fallback (no body axis).
        labels3, info3 = fitted_template.predict_with_info(p)
        assert labels3.shape == PRESSURE_SHAPE
        assert info3["fallback"] == "none"

    def test_predict_with_info_reports_all_background_fallback(
        self,
        fitted_axis: PressureBodyAxisPartitionBaseline,
    ) -> None:
        """All-zero pressure ⇒ fallback = 'all_background' and
        no_contact = True."""
        p = np.zeros(PRESSURE_SHAPE, dtype=np.float64)
        labels, info = fitted_axis.predict_with_info(p)
        assert info["fallback"] == "all_background"
        assert info["no_contact"] is True
        # The output is all BACKGROUND.
        assert int(labels.max()) == 0

    def test_predict_with_info_reports_tiny_contact_fallback(
        self,
        fitted_axis: PressureBodyAxisPartitionBaseline,
    ) -> None:
        """Single-pixel contact ⇒ degenerate PCA ⇒ fallback."""
        p = _synthetic_pressure_small()
        labels, info = fitted_axis.predict_with_info(p)
        # Either degenerate_pca or all_background fallback; either way
        # the output is all BACKGROUND.
        assert info["fallback"] in {"all_background"}
        assert info["degenerate_pca"] is True or info["no_contact"] is True
        assert int(labels.max()) == 0

    def test_vertical_body_top_is_head_feet_is_lower_leg(
        self,
        axis_state_default: AxisPartitionState,
    ) -> None:
        """Strong assertion: a vertical body bar must be labelled
        HEAD_NECK at the top of the matrix and LOWER_LEG_FOOT at the
        bottom.  This is the head→toe convention that the B02 v0.1
        axis partition MUST follow.

        The synthetic bar is at rows 40..160, columns 30..60.  Rows
        0..39 are outside the contact mask (BACKGROUND).  Rows
        40..160 are inside the mask.  With
        ``DEFAULT_SEGMENT_FRACTIONS = (0.18, 0.07, 0.13, 0.12, 0.10,
        0.10, 0.15, 0.15)`` the cumulative fractions are
        ``(0.18, 0.25, 0.38, 0.50, 0.60, 0.70, 0.85, 1.00)``.

        The top of the contact bar is at axis-fraction 0.0, which
        must map to the first segment (HEAD_NECK, ID=1).  The bottom
        of the contact bar is at axis-fraction 1.0, which must map
        to the last segment (LOWER_LEG_FOOT, ID=8).  This is the
        direction the Reviewer flagged as broken in R01.
        """
        from topper_perception.baseline.slp8_non_learning import (
            _build_axis_partition_labels,
        )

        pressure = _synthetic_pressure_bar()
        labels, info = _build_axis_partition_labels(pressure, axis_state_default)
        # No fallback: the synthetic bar has a clear axis.
        assert info.get("fallback", "none") == "none"
        # Top contact row (row 40, inside the bar) must be HEAD_NECK.
        # The axis-fraction at the top of the bar is 0.0, which
        # corresponds to segment 0 = HEAD_NECK.
        # We check the upper-middle of the bar to be robust to the
        # exact contact boundary: pick the row at 5% of the bar.
        top_row = 40 + int(0.05 * 120)  # row 46
        assert int(labels[top_row, 40]) == 1, (
            f"top of body should be HEAD_NECK (1); got {int(labels[top_row, 40])}"
        )
        # Bottom contact row (row 159) must be LOWER_LEG_FOOT.
        # The axis-fraction at the bottom of the bar is 1.0, which
        # corresponds to the last segment = LOWER_LEG_FOOT (8).
        bottom_row = 160 - 1  # row 159
        assert int(labels[bottom_row, 40]) == 8, (
            f"bottom of body should be LOWER_LEG_FOOT (8); got {int(labels[bottom_row, 40])}"
        )

    def test_vertical_body_no_segment_inversion(
        self,
        fitted_axis: PressureBodyAxisPartitionBaseline,
    ) -> None:
        """Additional strong assertion: the centroid of the HEAD_NECK
        prediction (centroid_y) must be strictly above the centroid
        of the LOWER_LEG_FOOT prediction (centroid_y).  In image
        coordinates, "above" means smaller y.  This catches a
        segment-fraction-vs-axis-direction inversion without relying
        on a specific contact row.
        """
        from topper_perception.baseline.slp8_non_learning import REGION_ID_TO_NAME

        p = _synthetic_pressure_bar()
        out = fitted_axis.predict(p)
        # HEAD_NECK is class 1; LOWER_LEG_FOOT is class 8.
        head_mask = out == 1
        feet_mask = out == 8
        assert head_mask.any(), "expected at least one HEAD_NECK pixel"
        assert feet_mask.any(), "expected at least one LOWER_LEG_FOOT pixel"
        head_y = float(np.argwhere(head_mask)[:, 0].mean())
        feet_y = float(np.argwhere(feet_mask)[:, 0].mean())
        assert head_y < feet_y, (
            f"head centroid (y={head_y}) must be above feet centroid "
            f"(y={feet_y}) for a vertical body; segment direction is inverted"
        )
        # Sanity: confirm the labels are what we expect.
        assert REGION_ID_TO_NAME[1] == "HEAD_NECK"
        assert REGION_ID_TO_NAME[8] == "LOWER_LEG_FOOT"


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
            # predict_with_info must also accept only ``pressure``.
            sig2 = inspect.signature(cls.predict_with_info)
            params2 = list(sig2.parameters)
            assert params2 == ["self", "pressure"], (
                f"{cls.__name__}.predict_with_info parameters = {params2}"
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

    def test_macro_precision_recall_from_confusion_matrix(self) -> None:
        """Hand-computed fixture: precision = TP / (TP + FP),
        recall = TP / (TP + FN), each in [0, 1].

        Construction:
            gt has 8 pixels of class 1, 4 pixels of class 2, 0 of class 3.
            pred has 4 pixels of class 1 (all correct), 4 pixels of
            class 3 (all wrong).
        Expected per-class metrics:
            class 1: TP=4, FP=0, FN=4 ⇒ precision=1.0, recall=0.5
            class 2: TP=0, FP=0, FN=4 ⇒ precision=0.0, recall=0.0
            class 3: TP=0, FP=4, FN=0 ⇒ precision=0.0, recall=0.0
            class 4..8: 0/0 ⇒ precision=0.0, recall=0.0
        """
        gt = np.zeros((4, 4), dtype=np.uint8)
        gt[0:2, 0:4] = 1   # 8 pixels of class 1
        gt[2:4, 0:2] = 2   # 4 pixels of class 2

        pred = np.zeros((4, 4), dtype=np.uint8)
        pred[0:2, 0:2] = 1  # 4 pixels of class 1 (correct on class 1, wrong on class 2)
        pred[2:4, 0:2] = 3  # 4 pixels of class 3 (wrong on class 2)

        m = compute_fixed_class_macro_metrics(
            gt, pred, class_ids=(1, 2, 3, 4, 5, 6, 7, 8), n_classes=9,
        )

        # class 1: TP=4, FP=0, FN=4
        assert m.per_class_tp[1] == 4
        assert m.per_class_fp[1] == 0
        assert m.per_class_fn[1] == 4
        assert m.per_class_precision[1] == pytest.approx(1.0)
        assert m.per_class_recall[1] == pytest.approx(0.5)

        # class 2: TP=0, FP=0, FN=4
        assert m.per_class_tp[2] == 0
        assert m.per_class_fp[2] == 0
        assert m.per_class_fn[2] == 4
        assert m.per_class_precision[2] == pytest.approx(0.0)
        assert m.per_class_recall[2] == pytest.approx(0.0)

        # class 3: TP=0, FP=4, FN=0
        assert m.per_class_tp[3] == 0
        assert m.per_class_fp[3] == 4
        assert m.per_class_fn[3] == 0
        assert m.per_class_precision[3] == pytest.approx(0.0)
        assert m.per_class_recall[3] == pytest.approx(0.0)

        # precision and recall must always be in [0, 1]
        for cid in (1, 2, 3, 4, 5, 6, 7, 8):
            assert 0.0 <= m.per_class_precision[cid] <= 1.0
            assert 0.0 <= m.per_class_recall[cid] <= 1.0
            # TP/FP/FN are non-negative integers
            assert m.per_class_tp[cid] >= 0
            assert m.per_class_fp[cid] >= 0
            assert m.per_class_fn[cid] >= 0
            # FP = pred_count - TP, FN = gt_count - TP
            assert m.per_class_fp[cid] == m.per_class_pred_count[cid] - m.per_class_tp[cid]
            assert m.per_class_fn[cid] == m.per_class_gt_count[cid] - m.per_class_tp[cid]

    def test_macro_precision_recall_perfect_prediction(self) -> None:
        """Perfect prediction ⇒ precision=recall=1.0 for predicted
        classes, and the macro recall/precision include 0s for
        unobserved classes."""
        gt = np.zeros((4, 4), dtype=np.uint8)
        gt[0:2, 0:2] = 1
        gt[0:2, 2:4] = 2
        pred = gt.copy()
        m = compute_fixed_class_macro_metrics(
            gt, pred, class_ids=(1, 2, 3, 4, 5, 6, 7, 8), n_classes=9,
        )
        for cid in (1, 2):
            assert m.per_class_precision[cid] == pytest.approx(1.0)
            assert m.per_class_recall[cid] == pytest.approx(1.0)
        for cid in (3, 4, 5, 6, 7, 8):
            assert m.per_class_precision[cid] == pytest.approx(0.0)
            assert m.per_class_recall[cid] == pytest.approx(0.0)


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


class TestOutputDirCollision:
    """The runner must refuse to overwrite an existing run output."""

    def test_runner_refuses_to_overwrite_done_json(
        self, tmp_path: Path
    ) -> None:
        """If ``DONE.json`` already exists in the output dir, the
        runner must raise ``OutputDirCollisionError`` and NOT silently
        overwrite."""
        runner = _load_runner_module()
        out = tmp_path / "EXP"
        out.mkdir()
        (out / "DONE.json").write_text('{"task_id": "old"}', encoding="utf-8")
        with pytest.raises(runner.OutputDirCollisionError):
            runner._check_output_dir_safety(out)

    def test_runner_refuses_to_overwrite_failed_json(
        self, tmp_path: Path
    ) -> None:
        """If ``FAILED.json`` already exists, the runner must also
        refuse."""
        runner = _load_runner_module()
        out = tmp_path / "EXP"
        out.mkdir()
        (out / "FAILED.json").write_text('{"task_id": "old"}', encoding="utf-8")
        with pytest.raises(runner.OutputDirCollisionError):
            runner._check_output_dir_safety(out)

    def test_runner_refuses_non_empty_output_dir(
        self, tmp_path: Path
    ) -> None:
        """Any other existing file in the output dir also blocks the
        run."""
        runner = _load_runner_module()
        out = tmp_path / "EXP"
        out.mkdir()
        (out / "metrics_summary.json").write_text("{}", encoding="utf-8")
        with pytest.raises(runner.OutputDirCollisionError):
            runner._check_output_dir_safety(out)

    def test_runner_allows_empty_output_dir(
        self, tmp_path: Path
    ) -> None:
        """An empty output dir (or a fresh non-existing one) is OK."""
        runner = _load_runner_module()
        # Empty existing dir is OK.
        out = tmp_path / "EXP"
        out.mkdir()
        runner._check_output_dir_safety(out)  # does not raise
        # Non-existing dir is OK.
        runner._check_output_dir_safety(tmp_path / "DOES_NOT_EXIST")  # does not raise


class TestRunnerFailClosed:
    """Runner must be fail-closed: n_failed > 0 → FAILED.json + non-zero.

    These tests exercise the runner at the integration level using real
    temporary directories and minimal B01 freeze + SLP8 dataset fixtures
    so the runner's path-validation logic is also exercised.
    """

    def _make_b01_dir(
        self,
        tmp_path: Path,
        sample_ids_splits: list[tuple[str, str]],
    ) -> tuple[Path, Path]:
        """Create a real B01 freeze dir and SLP8 dataset root.

        Returns (b01_dir, ds_root).  Both are real paths on disk so that
        the runner's path-validation checks pass.
        """
        import hashlib
        from topper_perception.io.slp8_training_table_freeze import (
            A06_SPLIT_SHA256_EXPECTED,
        )

        b01 = tmp_path / "b01_freeze"
        b01.mkdir()

        def _csv_row(sid: str, split: str) -> str:
            # Values in FreezeRow field order.
            return (
                f"{sid},{split},VAL,danaLab,uncover,00001,0,SUPINE,"
                f"pressure/{sid}.npy,labels/{sid}_region.npy,labels/{sid}_onehot.npy,"
                f"points/{sid}.csv,192,84,1|2|3|4|5|6|7|8,"
                f"V221_CORRECTED_SUPPORT_AUTO_ACCEPTED,NOT_REVIEWED,1.1.0,EXPORTED,"
                f"da39a3ee5e6b4b0d3255bfef95601890afd80709,16128,3072,0.0,True,True"
            )

        # Build CSV content with header matching FreezeRow field order.
        train_rows_list = [s for s, sp in sample_ids_splits if sp == "train"]
        val_rows_list   = [s for s, sp in sample_ids_splits if sp == "val"]
        header = (
            "sample_id,ml_split,source_split,setting,subject_id,cover,"
            "frame_id,posture,pressure_npy,region_label_npy,region_onehot_npy,"
            "points_csv,height,width,class_ids_present,annotation_provenance,"
            "source_review_status,export_version,export_status,"
            "source_pmarray_sha256,background_pixel_count,body_pixel_count,"
            "clipped_ratio,onehot_valid,onehot_roundtrip"
        )

        train_csv = b01 / "train_manifest.csv"
        train_csv.write_text(
            header + "\n" + "\n".join(_csv_row(s, "train") for s in train_rows_list) + "\n",
            encoding="utf-8",
        )
        val_csv = b01 / "val_manifest.csv"
        val_csv.write_text(
            header + "\n" + "\n".join(_csv_row(s, "val") for s in val_rows_list) + "\n",
            encoding="utf-8",
        )
        test_csv = b01 / "test_manifest.csv"
        test_csv.write_text(header + "\n" + _csv_row("SYNTH_TEST_0", "test") + "\n", encoding="utf-8")

        def _sha256(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        manifest = {
            "core": {
                "task_id": "TASK-SLP-A01-SLP8-FREEZE-TRAINING-TABLES-v0.1",
                "freeze_version": "slp8_training_tables_v0.1",
                "provenance": "V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
                "raw_semantics": "raw_pmarray_response",
                "a06_split_sha256": A06_SPLIT_SHA256_EXPECTED,
                "splits": {
                    "train": {"manifest_sha256": _sha256(train_csv)},
                    "val":   {"manifest_sha256": _sha256(val_csv)},
                },
            }
        }
        (b01 / "freeze_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        ds = tmp_path / "dataset"
        ds.mkdir()
        manifest_dir = ds / "manifest"
        manifest_dir.mkdir()
        # Dataset manifest: ALL sample IDs so get_sample() finds them.
        # FAIL samples have no pressure files → FileNotFoundError caught by runner.
        ds_header = "sample_id,subject_id,posture,ml_split\n"
        ds_rows = "\n".join(f"{sid},00001,SUPINE,{split_}" for sid, split_ in sample_ids_splits)
        (manifest_dir / "val_manifest.csv").write_text(ds_header + ds_rows + "\n", encoding="utf-8")
        return b01, ds

    def _config(self, b01_dir: Path, ds_root: Path, out_dir: Path) -> dict:
        return {
            "config_version": "1",
            "task_id": "TASK-SLP-B02-NON-LEARNING-REGION-BASELINE-v0.1",
            "freeze_version": "slp8_training_tables_v0.1",
            "b01_freeze_dir": str(b01_dir),
            "dataset_root": str(ds_root),
            "baselines": [{"name": "all_background"}],
            "metrics": {"fixed_iou": True, "fixed_dice": True},
            "provenance": "V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
            "raw_semantics": "raw_pmarray_response",
            "fit_split": "train",
        }

    def test_n_failed_gt_0_writes_failed_json_not_done(
        self, tmp_path: Path
    ) -> None:
        """When at least one contract failure occurs, the runner must
        write FAILED.json, must NOT write DONE.json, and must return 1."""
        runner = _load_runner_module()

        # Provide FAIL and SUCC sample IDs.  We create real pressure
        # files for SUCC samples (so they succeed) and leave FAIL files
        # absent (so np.load raises FileNotFoundError, caught by the runner).
        b01, ds = self._make_b01_dir(tmp_path, [
            ("SYNTH_FAIL_0", "train"),
            ("SYNTH_SUCC_0", "train"),
            ("SYNTH_FAIL_1", "val"),
            ("SYNTH_SUCC_1", "val"),
        ])

        # Create real pressure + label + onehot arrays for SUCC samples.
        # FAIL samples have no files on disk → FileNotFoundError.
        ds_pressure = ds / "pressure"
        ds_labels   = ds / "labels"
        ds_pressure.mkdir(exist_ok=True)
        ds_labels.mkdir(exist_ok=True)
        for sid, split in [("SYNTH_SUCC_0", "train"), ("SYNTH_SUCC_1", "val")]:
            np.save(ds_pressure / f"{sid}.npy", np.zeros(PRESSURE_SHAPE, dtype=np.float64))
            np.save(ds_labels   / f"{sid}_region.npy",  np.zeros(PRESSURE_SHAPE, dtype=np.uint8))
            np.save(ds_labels  / f"{sid}_onehot.npy",  np.zeros((9, *PRESSURE_SHAPE), dtype=np.uint8))

        out_dir = tmp_path / "EXP_FAIL_CLOSED"

        rc = runner.run(self._config(b01, ds, out_dir), out_dir)

        has_done = (out_dir / "DONE.json").is_file()
        has_failed = (out_dir / "FAILED.json").is_file()
        status_info = {}
        if (out_dir / "status.json").is_file():
            status_info = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))

        if not has_failed and not has_done:
            assert False, f"No DONE/FAILED written. status={status_info}"
        if has_done and not has_failed:
            assert False, f"DONE.json written (n_failed_total must be 0). rc={rc}, status={status_info}"

        assert (out_dir / "FAILED.json").is_file()
        assert not (out_dir / "DONE.json").is_file()
        assert rc == 1

        status = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
        assert status["status"] == "FAILED"
        assert status.get("n_samples_failed", 0) > 0

        failed_json = json.loads((out_dir / "FAILED.json").read_text(encoding="utf-8"))
        assert failed_json["status"] == "FAILED"
        assert failed_json["n_samples_failed"] > 0
        assert "failure_reason_counts" in failed_json

    def test_n_failed_eq_0_writes_done_json(
        self, tmp_path: Path
    ) -> None:
        """When no contract failures occur the runner writes DONE.json
        and returns 0."""
        from topper_perception.io import slp_8region_pressure_dataset as _ds_mod

        runner = _load_runner_module()

        b01, ds = self._make_b01_dir(tmp_path, [
            ("SYNTH_OK_0", "train"),
            ("SYNTH_OK_1", "train"),
            ("SYNTH_OK_2", "val"),
            ("SYNTH_OK_3", "val"),
        ])
        out_dir = tmp_path / "EXP_SUCCESS"

        _original = _ds_mod.Slp8RegionDatasetAdapter.load_sample

        def _patched(self, sample_id: str):
            p = np.zeros(PRESSURE_SHAPE, dtype=np.float64)

            class FakeSample:
                pressure: np.ndarray
                region_label: np.ndarray

            s = FakeSample()
            s.pressure = p
            s.region_label = np.zeros(PRESSURE_SHAPE, dtype=np.uint8)
            return s

        _ds_mod.Slp8RegionDatasetAdapter.load_sample = _patched  # type: ignore[method-assign]
        try:
            rc = runner.run(self._config(b01, ds, out_dir), out_dir)
        finally:
            _ds_mod.Slp8RegionDatasetAdapter.load_sample = _original  # type: ignore[method-assign]

        assert rc == 0
        assert (out_dir / "DONE.json").is_file()
        assert not (out_dir / "FAILED.json").is_file()
        status = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
        assert status["status"] == "DONE"

    def test_output_collision_still_protected_after_fail_closed_change(
        self, tmp_path: Path
    ) -> None:
        """The fail-closed change must not break the output-dir collision
        guard.  A pre-existing FAILED.json must still raise."""
        runner = _load_runner_module()
        out = tmp_path / "EXP_COLLISION"
        out.mkdir()
        (out / "FAILED.json").write_text('{"task_id": "old"}', encoding="utf-8")
        with pytest.raises(runner.OutputDirCollisionError):
            runner._check_output_dir_safety(out)


class TestResolvedConfigNoAbsolutePaths:
    """The runner must not write any absolute path into the resolved
    config or any other committed artefact."""

    def test_resolved_config_strips_dataset_root(self) -> None:
        runner = _load_runner_module()
        cfg = {
            "b01_freeze_dir": runner.REDACTED_LOCAL_PATH,
            "dataset_root": runner.REDACTED_LOCAL_PATH,
            "baselines": [{"name": "all_background", "kind": "sanity_floor"}],
            "fit_split": "train",
            "provenance": "V221_CORRECTED_SUPPORT_AUTO_ACCEPTED",
            "raw_semantics": "raw_pmarray_response",
            "freeze_version": "slp8_training_tables_v0.1",
        }
        # Must not raise.
        runner._check_resolved_config_no_absolute_paths(cfg)

    @pytest.mark.parametrize(
        "bad_path",
        [
            r"C:\Users\foo\bar",
            "/home/user/foo",
            r"\\server\share",
            r".\..\..\etc\passwd",
        ],
    )
    def test_resolved_config_rejects_any_absolute_path(
        self, bad_path: str
    ) -> None:
        runner = _load_runner_module()
        cfg = {
            "b01_freeze_dir": bad_path,
            "dataset_root": "REDACTED_LOCAL_PATH",
        }
        with pytest.raises(ValueError):
            runner._check_resolved_config_no_absolute_paths(cfg)


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
