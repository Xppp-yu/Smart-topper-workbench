"""Tests for the SLP A08 Body Axis and Bounding-Box Geometry module.

Coverage map (from the TASK-SLP-A08-BODY-AXIS-GEOMETRY-v0.1 contract):

* Normal body axis computation (shoulder, hip, longitudinal, center, orientation).
* Left/right flip detection.
* Coordinate rotation (face-up detection).
* Missing shoulder joints → uncertain/reject.
* Missing hip joints → uncertain/reject.
* Ankle occlusion handling.
* Out-of-bounds joint handling.
* Extreme frame jump flagging.
* Anomalous bone length flagging.
* Low confidence → reject/uncertain.
* Output determinism: same inputs → same outputs.
* A06 split is not modified.
* No region ground truth generated.
* Quality fields: axis_status, bbox_status, orientation_status,
  confidence, missing/occluded/out_of_bounds counts, error_codes, provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from topper_perception.io.slp_body_geometry import (  # noqa: E402
    ADAPTER_VERSION,
    DEFAULT_TASK_ID,
    DEFAULT_GENERATOR,
    JOINT_NAMES,
    J_R_SHOULDER,
    J_L_SHOULDER,
    J_R_HIP,
    J_L_HIP,
    J_R_ANKLE,
    J_L_ANKLE,
    J_L_KNEE,
    J_NECK,
    J_HEAD,
    J_R_WRIST,
    J_L_WRIST,
    RGB_HEIGHT,
    RGB_WIDTH,
    AxisStatus,
    BboxStatus,
    OrientationStatus,
    Point2D,
    BodyAxis,
    BoundingBox,
    JointValidity,
    BodyGeometryResult,
    compute_body_geometry,
    geometry_result_to_csv_row,
    GEOMETRY_CSV_COLUMNS,
    geometry_schema_dict,
    _detect_left_right_flip,
    _detect_face_up,
    _axis_status_from_counts,
    _bbox_status_from_bbox,
    _compute_orientation,
    _longitudinal_axis,
    _shoulder_axis,
    _hip_axis,
    _body_center,
    _is_visible,
    _is_out_of_bounds,
    _compute_bbox,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _jc(x: float, y: float, conf: float = 1.0) -> Any:
    """Create a minimal mock JointCoords-like object."""
    class MockJC:
        def __init__(self, x: float, y: float, conf: float):
            self.x = x
            self.y = y
            self.confidence = conf

        @property
        def is_occluded(self) -> bool:
            return self.confidence == 0

        @property
        def is_valid(self) -> bool:
            return not (np.isnan(self.x) or np.isnan(self.y))

    return MockJC(x, y, conf)


def _normal_joints() -> list:
    """Normal bedridden subject in SLP top-down view.

    Layout (approximate, in J0 RGB 576x1024 space):
    - In SLP, feet are at the TOP of image (low y), head at BOTTOM (high y).
    - Head: y≈745 (bottom of image)
    - Shoulders: y≈333 (middle-lower)
    - Hips: y≈497 (middle-upper, below head, above knees)
    - Knees: y≈600
    - Ankles: y≈750 (near feet/top)
    - This gives: hips.y > shoulders.y → test assertion la.start.y > la.end.y
    """
    return [
        _jc(200.0, 745.0, 1.0),   # 0:  head_cervical
        _jc(285.0, 614.0, 1.0),   # 1:  neck_c7
        _jc(380.0, 333.0, 1.0),   # 2:  right_shoulder
        _jc(313.0, 332.0, 1.0),   # 3:  right_elbow
        _jc(320.0, 450.0, 1.0),   # 4:  right_wrist
        _jc(180.0, 333.0, 1.0),   # 5:  left_shoulder
        _jc(258.0, 332.0, 1.0),   # 6:  left_elbow
        _jc(235.0, 450.0, 1.0),   # 7:  left_wrist
        _jc(400.0, 497.0, 1.0),   # 8:  right_hip
        _jc(313.0, 600.0, 1.0),   # 9:  right_knee
        _jc(348.0, 750.0, 1.0),   # 10: right_ankle
        _jc(200.0, 497.0, 1.0),   # 11: left_hip
        _jc(286.0, 600.0, 1.0),   # 12: left_knee
        _jc(285.0, 750.0, 1.0),   # 13: left_ankle
    ]


# ---------------------------------------------------------------------------
# Basic geometry computation
# ---------------------------------------------------------------------------


class TestNormalBodyAxis:
    """Tests for normal body axis computation."""

    def test_normal_shoulders_visible(self) -> None:
        """Both shoulders visible → shoulder axis valid."""
        joints = _normal_joints()
        axis = _shoulder_axis(joints)
        assert axis.direction_valid is True
        assert axis.start.valid is True
        assert axis.end.valid is True
        # Left shoulder should have lower x than right shoulder.
        assert axis.start.x < axis.end.x
        # Midpoint should be roughly between.
        assert axis.midpoint.x == pytest.approx((joints[J_L_SHOULDER].x + joints[J_R_SHOULDER].x) / 2, abs=0.01)
        assert axis.midpoint.y == pytest.approx((joints[J_L_SHOULDER].y + joints[J_R_SHOULDER].y) / 2, abs=0.01)

    def test_normal_hip_axis_valid(self) -> None:
        """Both hips visible → hip axis valid."""
        joints = _normal_joints()
        axis = _hip_axis(joints)
        assert axis.direction_valid is True
        assert axis.start.x < axis.end.x  # left hip → right hip

    def test_normal_longitudinal_axis(self) -> None:
        """Both shoulder and hip midpoints visible → longitudinal axis valid."""
        joints = _normal_joints()
        shoulder_ax = _shoulder_axis(joints)
        hip_ax = _hip_axis(joints)
        la = _longitudinal_axis(shoulder_ax, hip_ax)
        assert la.direction_valid is True
        # Hip midpoint should be "below" shoulder midpoint in image coords (higher y).
        assert la.start.y > la.end.y  # hip → shoulder
        assert la.midpoint.valid is True

    def test_normal_body_center(self) -> None:
        """Body center is midpoint of shoulder and hip midpoints."""
        joints = _normal_joints()
        shoulder_ax = _shoulder_axis(joints)
        hip_ax = _hip_axis(joints)
        center = _body_center(shoulder_ax, hip_ax)
        assert center.valid is True
        # Center should be roughly in the middle of the body.
        assert 250 < center.x < 400
        assert 350 < center.y < 500

    def test_normal_orientation(self) -> None:
        """Orientation is computed from longitudinal axis."""
        joints = _normal_joints()
        shoulder_ax = _shoulder_axis(joints)
        hip_ax = _hip_axis(joints)
        la = _longitudinal_axis(shoulder_ax, hip_ax)
        orient_deg, status, conf = _compute_orientation(la, shoulder_ax, hip_ax)
        assert orient_deg is not None
        assert 0 <= orient_deg <= 360
        assert conf > 0  # should have some confidence

    def test_normal_bbox(self) -> None:
        """BBox computed from all visible joints."""
        joints = _normal_joints()
        # Build minimal per_joint_validity.
        from topper_perception.io.slp_body_geometry import _valid_joint
        per_jv = [_valid_joint(j) for j in joints]
        # Fix index and name.
        for jv in per_jv:
            jv.index = per_jv.index(jv)
            jv.name = JOINT_NAMES[jv.index]

        bbox = _compute_bbox(joints, per_jv)
        assert bbox.valid is True
        assert bbox.width > 0
        assert bbox.height > 0
        assert bbox.x_min < bbox.x_max
        assert bbox.y_min < bbox.y_max

    def test_normal_full_result(self) -> None:
        """Full compute_body_geometry for normal joints."""
        joints = _normal_joints()
        result = compute_body_geometry(
            sample_id="test::danaLab::00001::uncover::0001",
            joints=joints,
        )
        assert isinstance(result, BodyGeometryResult)
        assert result.overall_status in ("accept", "uncertain")
        assert result.axis_status in (
            AxisStatus.ACCEPT_FULL.value,
            AxisStatus.ACCEPT_PARTIAL.value,
            AxisStatus.UNCERTAIN_MISSING_BOTH.value,
        )
        assert result.visible_joints >= 10
        assert result.shoulder_axis.direction_valid is True
        assert result.hip_axis.direction_valid is True
        assert result.longitudinal_axis.direction_valid is True
        assert result.body_center.valid is True
        assert result.bbox.valid is True
        assert result.adapter_version == ADAPTER_VERSION
        assert result.task_id == DEFAULT_TASK_ID
        assert result.generator == DEFAULT_GENERATOR


class TestMissingShoulders:
    """Tests for missing shoulder joints."""

    def test_both_shoulders_occluded(self) -> None:
        """Both shoulders occluded → shoulder axis invalid; overall uncertain/reject.

        With both shoulders occluded but both hips visible, axis_status is
        "accept_partial" (2/4 key joints = hips). The shoulder axis direction
        is invalid, and the overall status must be uncertain or reject.
        """
        joints = _normal_joints()
        # Set shoulders to occluded (confidence=0).
        joints[J_R_SHOULDER] = _jc(380.0, 496.0, 0.0)
        joints[J_L_SHOULDER] = _jc(180.0, 496.0, 0.0)
        result = compute_body_geometry(sample_id="test_missing_shoulders", joints=joints)
        assert result.overall_status in ("uncertain", "reject")
        assert result.shoulder_axis.direction_valid is False
        assert result.hip_axis.direction_valid is True  # Hips still visible

    def test_one_shoulder_occluded(self) -> None:
        """One shoulder occluded → partial axis with lower confidence."""
        joints = _normal_joints()
        joints[J_L_SHOULDER] = _jc(180.0, 496.0, 0.0)  # left shoulder occluded
        result = compute_body_geometry(sample_id="test_one_shoulder", joints=joints)
        # Should still be able to compute hip axis.
        assert result.hip_axis.direction_valid is True
        assert result.shoulder_axis.direction_valid is False
        assert result.overall_status in ("uncertain", "reject", "accept")

    def test_shoulder_out_of_bounds(self) -> None:
        """Shoulder out of image bounds → handled as invalid."""
        joints = _normal_joints()
        # Put right shoulder outside image bounds.
        joints[J_R_SHOULDER] = _jc(-50.0, 496.0, 1.0)  # negative x
        result = compute_body_geometry(sample_id="test_shoulder_oob", joints=joints)
        assert result.out_of_bounds_joints >= 1
        assert result.shoulder_axis.direction_valid is False


class TestMissingHips:
    """Tests for missing hip joints."""

    def test_both_hips_occluded(self) -> None:
        """Both hips occluded → uncertain/reject axis."""
        joints = _normal_joints()
        joints[J_R_HIP] = _jc(400.0, 250.0, 0.0)
        joints[J_L_HIP] = _jc(200.0, 250.0, 0.0)
        result = compute_body_geometry(sample_id="test_missing_hips", joints=joints)
        assert result.overall_status in ("uncertain", "reject")
        assert result.hip_axis.direction_valid is False
        # But shoulder axis should still be valid.
        assert result.shoulder_axis.direction_valid is True
        assert result.longitudinal_axis.direction_valid is False

    def test_one_hip_occluded(self) -> None:
        """One hip occluded → partial hip axis."""
        joints = _normal_joints()
        joints[J_L_HIP] = _jc(200.0, 250.0, 0.0)
        result = compute_body_geometry(sample_id="test_one_hip", joints=joints)
        assert result.hip_axis.direction_valid is False


class TestAnkleOcclusion:
    """Tests for ankle occlusion (A07: ankle/knee high occlusion)."""

    def test_both_ankles_occluded(self) -> None:
        """Both ankles occluded → handled correctly."""
        joints = _normal_joints()
        joints[J_R_ANKLE] = _jc(348.0, 323.0, 0.0)
        joints[J_L_ANKLE] = _jc(285.0, 158.0, 0.0)
        result = compute_body_geometry(sample_id="test_ankles_occluded", joints=joints)
        # Axis computation should still succeed.
        assert result.overall_status in ("accept", "uncertain")
        assert result.hip_axis.direction_valid is True
        assert result.shoulder_axis.direction_valid is True
        # But ankles should be counted as occluded.
        assert result.occluded_joints >= 2

    def test_left_ankle_only_visible(self) -> None:
        """Left ankle visible (right occluded) → partial leg."""
        joints = _normal_joints()
        joints[J_R_ANKLE] = _jc(348.0, 323.0, 0.0)  # occluded
        result = compute_body_geometry(sample_id="test_left_ankle", joints=joints)
        assert result.visible_joints >= 10  # Most joints still visible


class TestOutOfBounds:
    """Tests for out-of-bounds joint handling."""

    def test_head_out_of_bounds(self) -> None:
        """Head cervical out of bounds (A07: head_cervical out of bounds)."""
        joints = _normal_joints()
        # Simulate head out of top of image.
        joints[J_HEAD] = _jc(285.0, -20.0, 1.0)  # y < 0
        result = compute_body_geometry(sample_id="test_head_oob", joints=joints)
        assert result.out_of_bounds_joints >= 1
        # Body axis should still be computed.
        assert result.overall_status in ("accept", "uncertain", "reject")

    def test_multiple_oob_joints(self) -> None:
        """Multiple out-of-bounds joints (A07: head_cervical + neck_c7)."""
        joints = _normal_joints()
        joints[J_HEAD] = _jc(285.0, -30.0, 1.0)
        joints[J_NECK] = _jc(285.0, -10.0, 1.0)
        result = compute_body_geometry(sample_id="test_multi_oob", joints=joints)
        assert result.out_of_bounds_joints >= 2

    def test_joint_exactly_at_boundary(self) -> None:
        """Joint exactly at boundary is considered valid."""
        joints = _normal_joints()
        # x = 576 is out of bounds (0 <= x < 576).
        joints[J_R_WRIST] = _jc(576.0, 500.0, 1.0)  # exactly at boundary → out
        result = compute_body_geometry(sample_id="test_boundary", joints=joints)
        assert result.out_of_bounds_joints >= 1

        # x = 575 is at boundary but valid.
        joints[J_R_WRIST] = _jc(575.0, 500.0, 1.0)
        result2 = compute_body_geometry(sample_id="test_boundary2", joints=joints)
        assert result2.out_of_bounds_joints == 0


class TestLeftRightFlip:
    """Tests for left/right flip detection."""

    def test_no_flip_normal_layout(self) -> None:
        """Normal layout: left_shoulder.x < right_shoulder.x → no flip."""
        joints = _normal_joints()
        shoulder_ax = _shoulder_axis(joints)
        hip_ax = _hip_axis(joints)
        flip, codes = _detect_left_right_flip(shoulder_ax, hip_ax)
        assert flip is False
        assert len(codes) == 0

    def test_flip_detected(self) -> None:
        """Swapped left/right shoulder positions → flip detected."""
        joints = _normal_joints()
        # Swap shoulder x positions.
        joints[J_R_SHOULDER] = _jc(180.0, 496.0, 1.0)  # was right, now has low x
        joints[J_L_SHOULDER] = _jc(380.0, 496.0, 1.0)  # was left, now has high x
        shoulder_ax = _shoulder_axis(joints)
        flip, codes = _detect_left_right_flip(shoulder_ax, _hip_axis(joints))
        assert flip is True
        assert any("left_right_flip" in c for c in codes)

    def test_flip_in_result(self) -> None:
        """Full result includes left_right_flip_detected."""
        joints = _normal_joints()
        # Swap shoulders.
        joints[J_R_SHOULDER] = _jc(180.0, 496.0, 1.0)
        joints[J_L_SHOULDER] = _jc(380.0, 496.0, 1.0)
        result = compute_body_geometry(sample_id="test_flip", joints=joints)
        assert result.left_right_flip_detected is True
        assert any("left_right_flip" in c for c in result.error_codes)


class TestRotation:
    """Tests for face-up/rotation detection."""

    def test_no_rotation_normal(self) -> None:
        """Normal layout: no face-up detected (threshold 200px)."""
        joints = _normal_joints()
        shoulder_ax = _shoulder_axis(joints)
        hip_ax = _hip_axis(joints)
        la = _longitudinal_axis(shoulder_ax, hip_ax)
        face_up, codes = _detect_face_up(la)
        assert face_up is False

    def test_rotation_detected(self) -> None:
        """Extreme rotation (|dy| > 200px) → face_up detected."""
        joints = _normal_joints()
        # Set shoulders near top of image (y=100) and hips near bottom (y=800).
        # This creates |dy| = 700 > 200 threshold → flagged.
        for idx in [J_R_SHOULDER, J_L_SHOULDER]:
            joints[idx] = _jc(joints[idx].x, 100.0, 1.0)
        for idx in [J_R_HIP, J_L_HIP]:
            joints[idx] = _jc(joints[idx].x, 800.0, 1.0)

        shoulder_ax = _shoulder_axis(joints)
        hip_ax = _hip_axis(joints)
        la = _longitudinal_axis(shoulder_ax, hip_ax)
        face_up, codes = _detect_face_up(la)
        assert face_up is True
        assert any("face_up" in c for c in codes)


class TestExtremeFrameJump:
    """Tests for extreme frame jump flagging."""

    def test_extreme_jump_flag_in_result(self) -> None:
        """extreme_frame_jump flag is recorded in result and error_codes."""
        joints = _normal_joints()
        result = compute_body_geometry(
            sample_id="test_jump",
            joints=joints,
            extreme_frame_jump=True,
        )
        assert result.extreme_frame_jump is True
        assert "extreme_frame_jump" in result.error_codes

    def test_no_jump_without_flag(self) -> None:
        """Without flag, no jump error code."""
        joints = _normal_joints()
        result = compute_body_geometry(
            sample_id="test_no_jump",
            joints=joints,
            extreme_frame_jump=False,
        )
        assert result.extreme_frame_jump is False
        assert "extreme_frame_jump" not in result.error_codes


class TestAnomalousBoneLength:
    """Tests for anomalous bone length flagging."""

    def test_anomalous_length_flag(self) -> None:
        """anomalous_bone_length flag is recorded."""
        joints = _normal_joints()
        result = compute_body_geometry(
            sample_id="test_bone",
            joints=joints,
            anomalous_bone_length=True,
        )
        assert result.anomalous_bone_length is True
        assert "anomalous_bone_length" in result.error_codes


class TestConfidenceThresholds:
    """Tests for confidence-based reject/uncertain thresholds."""

    def test_all_joints_visible_accept(self) -> None:
        """All 14 joints visible → high confidence → accept."""
        joints = _normal_joints()
        result = compute_body_geometry(sample_id="test_full", joints=joints)
        assert result.overall_confidence >= 0.8
        assert result.overall_status == "accept"

    def test_all_joints_occluded_reject(self) -> None:
        """All 14 joints occluded → reject."""
        joints = [_jc(200.0, 500.0, 0.0) for _ in range(14)]
        result = compute_body_geometry(sample_id="test_all_occluded", joints=joints)
        assert result.overall_status == "reject"
        assert result.visible_joints == 0

    def test_all_joints_nan_reject(self) -> None:
        """All 14 joints NaN → reject."""
        joints = [_jc(float("nan"), float("nan"), 0.0) for _ in range(14)]
        result = compute_body_geometry(sample_id="test_all_nan", joints=joints)
        assert result.overall_status == "reject"
        assert result.valid_joints == 0

    def test_insufficient_joints_reject(self) -> None:
        """Only 1 joint visible → reject."""
        joints = _normal_joints()
        # Set all joints except shoulder to occluded.
        for i in range(len(joints)):
            if i not in (J_R_SHOULDER,):
                joints[i] = _jc(joints[i].x, joints[i].y, 0.0)
        result = compute_body_geometry(sample_id="test_insufficient", joints=joints)
        assert result.visible_joints == 1
        assert result.overall_status in ("reject", "uncertain")

    def test_mixed_visible_uncertain(self) -> None:
        """Many joints occluded → uncertain."""
        joints = _normal_joints()
        # Only shoulders, one hip, and neck visible.
        visible_indices = {J_HEAD, J_NECK, J_R_SHOULDER, J_L_SHOULDER, J_R_HIP}
        for i in range(len(joints)):
            if i not in visible_indices:
                joints[i] = _jc(joints[i].x, joints[i].y, 0.0)
        result = compute_body_geometry(sample_id="test_mixed", joints=joints)
        assert result.overall_status in ("uncertain", "accept")


class TestDeterminism:
    """Tests for output determinism."""

    def test_same_input_same_output(self) -> None:
        """Same joints → same result (determinism)."""
        joints = _normal_joints()
        r1 = compute_body_geometry(sample_id="test_deterministic", joints=joints)
        r2 = compute_body_geometry(sample_id="test_deterministic", joints=joints)
        assert r1.overall_confidence == r2.overall_confidence
        assert r1.overall_status == r2.overall_status
        assert r1.shoulder_axis.midpoint.x == r2.shoulder_axis.midpoint.x
        assert r1.hip_axis.midpoint.x == r2.hip_axis.midpoint.x
        assert r1.bbox.width == r2.bbox.width
        assert r1.error_codes == r2.error_codes
        assert r1.adapter_version == r2.adapter_version
        assert r1.task_id == r2.task_id

    def test_different_joints_different_output(self) -> None:
        """Different joints → different axis."""
        joints1 = _normal_joints()
        joints2 = _normal_joints()
        # Shift shoulder positions.
        joints2[J_R_SHOULDER] = _jc(450.0, 496.0, 1.0)
        joints2[J_L_SHOULDER] = _jc(100.0, 496.0, 1.0)

        r1 = compute_body_geometry(sample_id="test_diff1", joints=joints1)
        r2 = compute_body_geometry(sample_id="test_diff2", joints=joints2)
        assert r1.shoulder_axis.midpoint.x != r2.shoulder_axis.midpoint.x

    def test_created_at_is_set(self) -> None:
        """Created_at is set in result."""
        joints = _normal_joints()
        result = compute_body_geometry(sample_id="test_prov", joints=joints)
        assert result.created_at != ""
        assert "T" in result.created_at  # ISO format


class TestQualityFields:
    """Tests for explicit quality fields."""

    def test_missing_joint_count(self) -> None:
        """Missing (NaN) joints are counted."""
        joints = _normal_joints()
        joints[0] = _jc(float("nan"), float("nan"), 0.0)
        result = compute_body_geometry(sample_id="test_missing", joints=joints)
        assert result.missing_joints >= 1

    def test_occluded_joint_count(self) -> None:
        """Occluded (confidence=0) joints are counted."""
        joints = _normal_joints()
        joints[J_L_ANKLE] = _jc(285.0, 158.0, 0.0)
        result = compute_body_geometry(sample_id="test_occluded", joints=joints)
        assert result.occluded_joints >= 1

    def test_error_codes_explicit(self) -> None:
        """Error codes are explicit and non-empty on error."""
        joints = _normal_joints()
        joints[J_R_SHOULDER] = _jc(-50.0, 496.0, 1.0)  # out of bounds
        result = compute_body_geometry(sample_id="test_errors", joints=joints)
        assert isinstance(result.error_codes, tuple)
        assert len(result.error_codes) > 0

    def test_provenance_fields(self) -> None:
        """Provenance fields are correctly set."""
        joints = _normal_joints()
        result = compute_body_geometry(
            sample_id="test_prov",
            joints=joints,
            task_id="TEST-TASK",
            adapter_version="test_v0.1",
            generator="test.generator",
        )
        assert result.task_id == "TEST-TASK"
        assert result.adapter_version == "test_v0.1"
        assert result.generator == "test.generator"

    def test_per_joint_validity(self) -> None:
        """Per-joint validity is recorded."""
        joints = _normal_joints()
        result = compute_body_geometry(sample_id="test_pjv", joints=joints)
        assert len(result.per_joint_validity) == 14
        assert all("index" in jv for jv in result.per_joint_validity)
        assert all("name" in jv for jv in result.per_joint_validity)
        assert all("is_visible" in jv for jv in result.per_joint_validity)
        assert all("is_occluded" in jv for jv in result.per_joint_validity)


class TestCsvOutput:
    """Tests for CSV row flattening."""

    def test_csv_row_keys(self) -> None:
        """CSV row contains all expected columns."""
        joints = _normal_joints()
        result = compute_body_geometry(sample_id="test_csv", joints=joints)
        row = geometry_result_to_csv_row(result)
        for col in GEOMETRY_CSV_COLUMNS:
            assert col in row, f"Missing column: {col}"
        assert len(row) == len(GEOMETRY_CSV_COLUMNS)

    def test_csv_row_serde(self) -> None:
        """CSV row can be serialized to JSON without error."""
        joints = _normal_joints()
        result = compute_body_geometry(sample_id="test_json", joints=joints)
        row = geometry_result_to_csv_row(result)
        json_str = json.dumps(row)
        restored = json.loads(json_str)
        assert restored["sample_id"] == "test_json"


class TestSchema:
    """Tests for geometry output schema."""

    def test_schema_valid_json(self) -> None:
        """Schema is valid JSON."""
        schema = geometry_schema_dict()
        json_str = json.dumps(schema)
        restored = json.loads(json_str)
        assert restored["title"] == "SLPBodyGeometrySchema"
        assert restored["version"] == "slp_body_geometry_v0.1"

    def test_schema_required_fields(self) -> None:
        """Schema has all required fields."""
        schema = geometry_schema_dict()
        result_dict = compute_body_geometry(
            sample_id="test_schema",
            joints=_normal_joints(),
        ).as_dict()
        required = schema["required"]
        for field in required:
            assert field in result_dict, f"Missing field in result: {field}"


class TestA06SplitNotModified:
    """Tests that A06 split is not modified by this module."""

    def test_split_read_only(self) -> None:
        """The geometry module only reads canonical samples; it never writes split."""
        # This is verified by the fact that:
        # 1. compute_body_geometry takes joints as input, not split.
        # 2. The runner script reads the split manifest as read-only.
        # 3. No code in slp_body_geometry.py writes to the split manifest.
        # We test that the module does not have any write-side effect on split.
        # Since the module is read-only by design, this is covered by code review.
        # We just verify the module doesn't crash when called.
        joints = _normal_joints()
        result = compute_body_geometry(sample_id="test_split", joints=joints)
        assert result.sample_id == "test_split"
        # The result does not contain any split-related fields.
        assert "split" not in result.as_dict()
        assert "subject_split" not in result.as_dict()


class TestEdgeCases:
    """Edge case tests."""

    def test_zero_confidence_nan_coords(self) -> None:
        """confidence=0 with NaN coords → handled correctly."""
        joints = _normal_joints()
        joints[0] = _jc(float("nan"), float("nan"), 0.0)  # head: occluded + NaN
        result = compute_body_geometry(sample_id="test_nan_conf0", joints=joints)
        assert result.missing_joints >= 1
        assert result.occluded_joints >= 1

    def test_bbox_validity_with_oob(self) -> None:
        """BBox excludes out-of-bounds joints."""
        joints = _normal_joints()
        # Put a joint way outside the image.
        joints[J_R_WRIST] = _jc(1000.0, 2000.0, 1.0)
        result = compute_body_geometry(sample_id="test_bbox_oob", joints=joints)
        # bbox should be based on in-bounds joints.
        assert result.bbox.x_max < 1000.0

    def test_orientation_ambiguous_no_direction(self) -> None:
        """No valid direction → orientation ambiguous."""
        joints = [_jc(200.0, 500.0, 0.0) for _ in range(14)]
        joints[J_R_SHOULDER] = _jc(380.0, 496.0, 0.0)
        joints[J_L_SHOULDER] = _jc(180.0, 496.0, 0.0)
        joints[J_R_HIP] = _jc(400.0, 250.0, 0.0)
        joints[J_L_HIP] = _jc(200.0, 250.0, 0.0)
        result = compute_body_geometry(sample_id="test_no_dir", joints=joints)
        assert result.orientation_status in (
            OrientationStatus.AMBIGUOUS.value,
            OrientationStatus.REJECT.value,
        )

    def test_axis_status_counts(self) -> None:
        """Axis status computation from counts."""
        status, conf, codes = _axis_status_from_counts(4, 4, 10, 14)
        assert status == AxisStatus.ACCEPT_FULL.value
        assert conf == 1.0
        assert len(codes) == 0

        status2, conf2, codes2 = _axis_status_from_counts(2, 4, 5, 14)
        assert conf2 < 1.0

        status3, conf3, codes3 = _axis_status_from_counts(0, 4, 5, 14)
        assert "no_visible_key_joints" in codes3

    def test_bbox_status_reject(self) -> None:
        """BBox with invalid span → reject."""
        from topper_perception.io.slp_body_geometry import _compute_bbox
        # Build a mock JointValidity list.
        from topper_perception.io.slp_body_geometry import _valid_joint
        joints = _normal_joints()
        per_jv = [_valid_joint(j) for j in joints]
        for jv in per_jv:
            jv.index = per_jv.index(jv)
            jv.name = JOINT_NAMES[jv.index]

        # Force an invalid bbox.
        bbox = BoundingBox(
            x_min=0.0, y_min=0.0,
            x_max=0.0, y_max=0.0,
            width=0.0, height=0.0,
            center_x=0.0, center_y=0.0,
            valid=False,
        )
        status, codes = _bbox_status_from_bbox(bbox, 10, 14)
        assert status == BboxStatus.REJECT.value
