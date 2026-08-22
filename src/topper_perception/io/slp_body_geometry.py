"""SLP Body Axis and Bounding-Box Geometry — A08.

This module computes deterministic geometric primitives from the 14-joint SLP
ground truth (J0, RGB space).  It is NOT a ground-truth builder and does NOT
train segmentation models.

Design contract (mirroring the TASK-SLP-A08-BODY-AXIS-GEOMETRY-v0.1):

* Deterministic: the same inputs always produce the same outputs.
* No silent imputation: when critical joints are missing the result is
  marked ``reject`` or ``uncertain`` with an explicit error code — never a
  quietly interpolated coordinate.
* Input-agnostic rules: geometry rules are fixed; they do not adapt based on
  model outputs or data distributions.
* Only J0 (original RGB-space) joints are processed here.  J1 (homography-
  derived PM-space) remains the responsibility of downstream tasks.
* Left/right confusion is handled by the ``left_right_flip`` flag and the
  orientation confidence field.
* Coordinate rotation (face-up / face-down) is detected and reported.

Outputs:
  * Body axes (shoulder, hip, longitudinal, center)
  * Body orientation and confidence
  * Axis-aligned bounding box with per-joint validity breakdown
  * Quality fields: status codes, joint counts, error codes, provenance

No region ground truth is generated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .slp_joint_eda import JointCoords

# ---------------------------------------------------------------------------
# Version constants
# ---------------------------------------------------------------------------

ADAPTER_VERSION = "slp_body_geometry_v0.1"
DEFAULT_TASK_ID = "TASK-SLP-A08-BODY-AXIS-GEOMETRY-v0.1"
DEFAULT_GENERATOR = "topper_perception.io.slp_body_geometry"

# Image dimensions for out-of-bounds checking (J0 RGB space, danaLab).
RGB_WIDTH = 576
RGB_HEIGHT = 1024

# ---------------------------------------------------------------------------
# Joint indices (mirrors A07 slp_joint_eda.py)
# ---------------------------------------------------------------------------

JOINT_NAMES = (
    "head_cervical",
    "neck_c7",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
)

# Index constants for clarity.
J_HEAD = 0
J_NECK = 1
J_R_SHOULDER = 2
J_R_ELBOW = 3
J_R_WRIST = 4
J_L_SHOULDER = 5
J_L_ELBOW = 6
J_L_WRIST = 7
J_R_HIP = 8
J_R_KNEE = 9
J_R_ANKLE = 10
J_L_HIP = 11
J_L_KNEE = 12
J_L_ANKLE = 13

# Key pairs for axis computation.
_SHOULDER_PAIR = (J_R_SHOULDER, J_L_SHOULDER)
_HIP_PAIR = (J_R_HIP, J_L_HIP)

# Joints needed for each axis (index → whether it must be visible).
# Minimum viable: at least 1 shoulder + 1 hip → center-only axis.
# For orientation: need both shoulders or both hips for direction.
_SHOULDER_AXIS_JOINTS = (J_R_SHOULDER, J_L_SHOULDER)
_HIP_AXIS_JOINTS = (J_R_HIP, J_L_HIP)
_LONGITUDINAL_JOINTS = (J_R_SHOULDER, J_L_SHOULDER, J_R_HIP, J_L_HIP)

# Minimum visible joints required for ACCEPT (not all 14 needed).
_MIN_VISIBLE_FOR_AXIS = 2   # at least 2 visible to compute any axis
_MIN_VISIBLE_FOR_ORIENTATION = 2  # at least 2 visible to estimate orientation

# Confidence thresholds.
_CONFIDENCE_ACCEPT = 0.80   # ≥80 % visible key joints → ACCEPT
_CONFIDENCE_UNCERTAIN = 0.50  # ≥50 % visible key joints → UNCERTAIN
# Below 50 % visible → REJECT

# Padding fraction applied to bbox (fraction of body span).
_BBOX_PADDING_FRAC = 0.05

# Maximum plausible body span in J0 RGB pixels (used to sanity-check bbox).
_MAX_BODY_SPAN_PX = 1200.0


# ---------------------------------------------------------------------------
# Status enums
# ---------------------------------------------------------------------------

class AxisStatus(str, Enum):
    ACCEPT = "accept"
    UNCERTAIN = "uncertain"
    REJECT = "reject"
    # Detailed sub-states for ACCEPT/UNCERTAIN
    ACCEPT_FULL = "accept_full"         # all key joints visible
    ACCEPT_PARTIAL = "accept_partial"    # ≥80 % visible key joints
    UNCERTAIN_MISSING_SHOULDERS = "uncertain_missing_shoulders"
    UNCERTAIN_MISSING_HIPS = "uncertain_missing_hips"
    UNCERTAIN_MISSING_BOTH = "uncertain_missing_both"
    REJECT_INSUFFICIENT_VISIBLE = "reject_insufficient_visible"
    REJECT_NO_VALID_JOINTS = "reject_no_valid_joints"
    REJECT_ALL_OCCLUDED = "reject_all_occluded"


class BboxStatus(str, Enum):
    ACCEPT = "accept"
    UNCERTAIN = "uncertain"
    REJECT = "reject"


class OrientationStatus(str, Enum):
    ACCEPT = "accept"
    UNCERTAIN = "uncertain"
    REJECT = "reject"
    # Sub-states
    NORMAL = "normal"          # head above feet (upright bedridden)
    FACE_UP = "face_up"        # rotated ~180° in y
    AMBIGUOUS = "ambiguous"    # cannot determine from available joints


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Point2D:
    """A 2-D point with explicit validity."""
    x: float
    y: float
    valid: bool

    def __post_init__(self) -> None:
        # Coerce NaN to invalid.
        if self.valid and (np.isnan(self.x) or np.isnan(self.y)):
            object.__setattr__(self, "valid", False)

    def as_tuple(self) -> tuple[float, float] | None:
        return (self.x, self.y) if self.valid else None

    def as_dict(self) -> dict[str, float | bool | None]:
        return {"x": self.x, "y": self.y, "valid": self.valid}


@dataclass(frozen=True, slots=True)
class BodyAxis:
    """A directed axis segment between two points."""
    start: Point2D
    end: Point2D
    midpoint: Point2D
    direction_valid: bool   # False if start/end are not both valid

    @property
    def is_valid(self) -> bool:
        return self.direction_valid

    def as_dict(self) -> dict[str, dict]:
        return {
            "start": self.start.as_dict(),
            "end": self.end.as_dict(),
            "midpoint": self.midpoint.as_dict(),
            "direction_valid": self.direction_valid,
        }


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned bounding box derived from valid visible joints."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    width: float
    height: float
    center_x: float
    center_y: float
    valid: bool

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "width": self.width,
            "height": self.height,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "valid": self.valid,
        }


@dataclass
class JointValidity:
    """Per-joint visibility breakdown."""
    index: int
    name: str
    x: float
    y: float
    confidence: float  # 0=occluded, 1=visible
    is_valid: bool    # not NaN
    is_visible: bool  # is_valid AND confidence == 1
    is_occluded: bool
    is_out_of_bounds: bool
    status: str       # "visible", "occluded", "invalid", "out_of_bounds"


@dataclass
class BodyGeometryResult:
    """Complete geometric result for one frame."""

    # Identity
    sample_id: str
    coordinate_frame: str  # "J0_RGB"

    # Body axes
    shoulder_axis: BodyAxis
    hip_axis: BodyAxis
    longitudinal_axis: BodyAxis
    body_center: Point2D

    # Orientation
    orientation_degrees: float | None   # angle of longitudinal axis from horizontal (degrees)
    orientation_status: str
    orientation_confidence: float  # 0.0 – 1.0

    # Bounding box
    bbox: BoundingBox
    bbox_status: str

    # Quality fields
    axis_status: str
    overall_confidence: float  # 0.0 – 1.0
    overall_status: str       # "accept" | "uncertain" | "reject"

    # Joint counts
    total_joints: int
    valid_joints: int          # not NaN
    visible_joints: int        # valid AND confidence == 1
    occluded_joints: int       # confidence == 0 (even if coords valid)
    out_of_bounds_joints: int
    missing_joints: int        # NaN coords

    # Per-joint validity
    per_joint_validity: list[dict]

    # Error codes
    error_codes: tuple[str, ...]

    # Flags
    left_right_flip_detected: bool
    face_up_detected: bool      # subject rotated ~180°
    extreme_frame_jump: bool
    anomalous_bone_length: bool

    # Provenance
    task_id: str
    adapter_version: str
    generator: str
    created_at: str

    def as_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "coordinate_frame": self.coordinate_frame,
            "shoulder_axis": self.shoulder_axis.as_dict(),
            "hip_axis": self.hip_axis.as_dict(),
            "longitudinal_axis": self.longitudinal_axis.as_dict(),
            "body_center": self.body_center.as_dict(),
            "orientation_degrees": self.orientation_degrees,
            "orientation_status": self.orientation_status,
            "orientation_confidence": self.orientation_confidence,
            "bbox": self.bbox.as_dict(),
            "bbox_status": self.bbox_status,
            "axis_status": self.axis_status,
            "overall_confidence": self.overall_confidence,
            "overall_status": self.overall_status,
            "total_joints": self.total_joints,
            "valid_joints": self.valid_joints,
            "visible_joints": self.visible_joints,
            "occluded_joints": self.occluded_joints,
            "out_of_bounds_joints": self.out_of_bounds_joints,
            "missing_joints": self.missing_joints,
            "per_joint_validity": self.per_joint_validity,
            "error_codes": list(self.error_codes),
            "left_right_flip_detected": self.left_right_flip_detected,
            "face_up_detected": self.face_up_detected,
            "extreme_frame_jump": self.extreme_frame_jump,
            "anomalous_bone_length": self.anomalous_bone_length,
            "task_id": self.task_id,
            "adapter_version": self.adapter_version,
            "generator": self.generator,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _is_visible(jc: "JointCoords") -> bool:
    return jc.is_valid and not jc.is_occluded


def _is_out_of_bounds(jc: "JointCoords", *, width: int = RGB_WIDTH, height: int = RGB_HEIGHT) -> bool:
    if not jc.is_valid:
        return False
    return jc.x < 0 or jc.x >= width or jc.y < 0 or jc.y >= height


def _valid_joint(jc: "JointCoords", *, width: int = RGB_WIDTH, height: int = RGB_HEIGHT) -> JointValidity:
    """Build a JointValidity record from a JointCoords."""
    idx = 0  # caller fills this
    jv = JointValidity(
        index=idx,
        name=JOINT_NAMES[idx],
        x=jc.x,
        y=jc.y,
        confidence=jc.confidence,
        is_valid=jc.is_valid,
        is_visible=_is_visible(jc),
        is_occluded=jc.is_occluded,
        is_out_of_bounds=_is_out_of_bounds(jc, width=width, height=height),
        status="visible" if _is_visible(jc) else ("occluded" if jc.is_occluded else "invalid"),
    )
    # Fix index and name after construction.
    return jv


def _point(jc: "JointCoords", *, width: int = RGB_WIDTH, height: int = RGB_HEIGHT) -> Point2D:
    """Convert a JointCoords to a Point2D with bounds checking.

    A point is valid for axis computation only when:
    - coordinates are not NaN,
    - confidence != 0 (joint is not occluded).

    Occluded joints (confidence=0) are excluded from axis computation per
    the deterministic rules: key nodes missing → explicit reject/uncertain,
    never silently interpolated.
    """
    if not jc.is_valid or np.isnan(jc.x) or np.isnan(jc.y):
        return Point2D(x=float("nan"), y=float("nan"), valid=False)
    if jc.is_occluded:
        # Occluded: coordinates may be unreliable; exclude from axis.
        return Point2D(x=jc.x, y=jc.y, valid=False)
    oob = jc.x < 0 or jc.x >= width or jc.y < 0 or jc.y >= height
    return Point2D(x=jc.x, y=jc.y, valid=(not oob))


def _axis_from_points(start: Point2D, end: Point2D) -> BodyAxis:
    """Build a BodyAxis from two endpoints."""
    if start.valid and end.valid:
        mx = (start.x + end.x) / 2.0
        my = (start.y + end.y) / 2.0
        return BodyAxis(
            start=start, end=end,
            midpoint=Point2D(x=mx, y=my, valid=True),
            direction_valid=True,
        )
    # Direction invalid but we can still report midpoint if one point valid.
    if start.valid:
        return BodyAxis(start=start, end=end, midpoint=start, direction_valid=False)
    if end.valid:
        return BodyAxis(start=start, end=end, midpoint=end, direction_valid=False)
    return BodyAxis(start=start, end=end,
                    midpoint=Point2D(x=float("nan"), y=float("nan"), valid=False),
                    direction_valid=False)


def _shoulder_axis(joints: list["JointCoords"], *, width: int = RGB_WIDTH, height: int = RGB_HEIGHT) -> BodyAxis:
    """Compute shoulder axis: from left to right shoulder."""
    r_shoulder = _point(joints[J_R_SHOULDER], width=width, height=height)
    l_shoulder = _point(joints[J_L_SHOULDER], width=width, height=height)
    return _axis_from_points(l_shoulder, r_shoulder)


def _hip_axis(joints: list["JointCoords"], *, width: int = RGB_WIDTH, height: int = RGB_HEIGHT) -> BodyAxis:
    """Compute hip axis: from left to right hip."""
    r_hip = _point(joints[J_R_HIP], width=width, height=height)
    l_hip = _point(joints[J_L_HIP], width=width, height=height)
    return _axis_from_points(l_hip, r_hip)


def _longitudinal_axis(
    shoulder_axis: BodyAxis,
    hip_axis: BodyAxis,
    *, width: int = RGB_WIDTH, height: int = RGB_HEIGHT,
) -> BodyAxis:
    """Compute body longitudinal axis: from hip midpoint to shoulder midpoint."""
    if shoulder_axis.midpoint.valid and hip_axis.midpoint.valid:
        return BodyAxis(
            start=hip_axis.midpoint,
            end=shoulder_axis.midpoint,
            midpoint=Point2D(
                x=(hip_axis.midpoint.x + shoulder_axis.midpoint.x) / 2.0,
                y=(hip_axis.midpoint.y + shoulder_axis.midpoint.y) / 2.0,
                valid=True,
            ),
            direction_valid=True,
        )
    if shoulder_axis.midpoint.valid:
        return BodyAxis(
            start=shoulder_axis.midpoint, end=shoulder_axis.midpoint,
            midpoint=shoulder_axis.midpoint, direction_valid=False,
        )
    if hip_axis.midpoint.valid:
        return BodyAxis(
            start=hip_axis.midpoint, end=hip_axis.midpoint,
            midpoint=hip_axis.midpoint, direction_valid=False,
        )
    return BodyAxis(
        start=Point2D(x=float("nan"), y=float("nan"), valid=False),
        end=Point2D(x=float("nan"), y=float("nan"), valid=False),
        midpoint=Point2D(x=float("nan"), y=float("nan"), valid=False),
        direction_valid=False,
    )


def _body_center(
    shoulder_axis: BodyAxis,
    hip_axis: BodyAxis,
) -> Point2D:
    """Compute body center as midpoint of shoulder and hip midpoints."""
    sm = shoulder_axis.midpoint
    hm = hip_axis.midpoint
    if sm.valid and hm.valid:
        return Point2D(
            x=(sm.x + hm.x) / 2.0,
            y=(sm.y + hm.y) / 2.0,
            valid=True,
        )
    if sm.valid:
        return sm
    if hm.valid:
        return hm
    return Point2D(x=float("nan"), y=float("nan"), valid=False)


def _compute_orientation(
    longitudinal_axis: BodyAxis,
    shoulder_axis: BodyAxis,
    hip_axis: BodyAxis,
) -> tuple[float | None, str, float]:
    """Compute body orientation in degrees from horizontal.

    Returns (angle_degrees, status, confidence).
    angle_degrees: clockwise from horizontal axis (0-360).
    """
    if not longitudinal_axis.direction_valid:
        # Try to salvage from individual axes.
        if shoulder_axis.direction_valid:
            dx = shoulder_axis.end.x - shoulder_axis.start.x
            dy = shoulder_axis.end.y - shoulder_axis.start.y
        elif hip_axis.direction_valid:
            dx = hip_axis.end.x - hip_axis.start.x
            dy = hip_axis.end.y - hip_axis.start.y
        else:
            return None, OrientationStatus.REJECT.value, 0.0
        angle = np.degrees(np.arctan2(dy, dx)) % 360
        return float(angle), OrientationStatus.UNCERTAIN.value, 0.5

    # Primary: use longitudinal axis.
    dx = longitudinal_axis.end.x - longitudinal_axis.start.x
    dy = longitudinal_axis.end.y - longitudinal_axis.start.y
    angle = np.degrees(np.arctan2(dy, dx)) % 360

    # Normal orientation: subject lies roughly horizontal → axis near 0° or 180°.
    # In SLP's top-down view, the body runs along the y-axis (bed direction),
    # so the longitudinal axis should be near vertical in image coords.
    # A "normal" bedridden subject has the axis roughly along the y direction.
    # We compute confidence as how close the axis is to vertical.
    # |sin(angle)| near 1 → vertical orientation.
    angle_rad = np.radians(angle)
    vertical_alignment = abs(np.sin(angle_rad))  # 1 = vertical, 0 = horizontal
    confidence = float(vertical_alignment)

    # A08 uses a fixed angle-based orientation classification.
    # The physical face-up/face-down distinction is not reliably detectable
    # from a single static frame without prior knowledge of which end is the head.
    # We flag it as "ambiguous" for now.
    status = OrientationStatus.AMBIGUOUS.value

    return float(angle), status, confidence


def _compute_bbox(
    joints: list["JointCoords"],
    per_joint_validity: list[JointValidity],
    *,
    width: int = RGB_WIDTH,
    height: int = RGB_HEIGHT,
) -> BoundingBox:
    """Compute axis-aligned bounding box from all valid, visible, in-bounds joints."""
    xs = []
    ys = []
    for jv in per_joint_validity:
        if jv.is_visible and not jv.is_out_of_bounds:
            xs.append(jv.x)
            ys.append(jv.y)

    if not xs:
        return BoundingBox(
            x_min=float("nan"), y_min=float("nan"),
            x_max=float("nan"), y_max=float("nan"),
            width=0.0, height=0.0,
            center_x=float("nan"), center_y=float("nan"),
            valid=False,
        )

    x_min_v = min(xs)
    y_min_v = min(ys)
    x_max_v = max(xs)
    y_max_v = max(xs)
    w = x_max_v - x_min_v
    h = y_max_v - y_min_v

    # Sanity check: if bbox is unreasonably large, reject.
    span = max(w, h)
    valid = span <= _MAX_BODY_SPAN_PX and span > 0

    return BoundingBox(
        x_min=x_min_v, y_min=y_min_v,
        x_max=x_max_v, y_max=y_max_v,
        width=w, height=h,
        center_x=(x_min_v + x_max_v) / 2.0,
        center_y=(y_min_v + y_max_v) / 2.0,
        valid=valid,
    )


def _detect_left_right_flip(
    shoulder_axis: BodyAxis,
    hip_axis: BodyAxis,
) -> tuple[bool, list[str]]:
    """Detect potential left/right flip from anatomical consistency.

    In SLP's annotation convention:
    - right_shoulder should be on the RIGHT side of the image (higher x)
      when the subject faces the camera.
    - left_shoulder should be on the LEFT side (lower x).

    If the observed ordering contradicts this, flag it as a potential flip.
    Returns (flip_detected, error_codes).
    """
    error_codes: list[str] = []
    flip = False

    # Check shoulder axis direction.
    if shoulder_axis.direction_valid:
        # Expected: right_shoulder.x > left_shoulder.x → direction should be positive x.
        if shoulder_axis.end.x < shoulder_axis.start.x:
            flip = True
            error_codes.append("left_right_flip_suspected_shoulder")

    # Check hip axis direction.
    if hip_axis.direction_valid:
        if hip_axis.end.x < hip_axis.start.x:
            flip = True
            error_codes.append("left_right_flip_suspected_hip")

    return flip, error_codes


def _detect_face_up(
    longitudinal_axis: BodyAxis,
) -> tuple[bool, list[str]]:
    """Detect 180° rotation (face-up vs face-down) from axis direction.

    In a normal bedridden subject lying supine, the longitudinal axis
    runs roughly from hip (bottom) to shoulder (top), which in image
    coordinates is in the negative y direction (y decreases upward).

    A face-up / face-down flip would reverse this direction.
    We can only detect this reliably if both endpoints of the longitudinal
    axis are valid.
    """
    error_codes: list[str] = []
    face_up = False

    if not longitudinal_axis.direction_valid:
        return face_up, error_codes

    dy = longitudinal_axis.end.y - longitudinal_axis.start.y
    # Normal: shoulder is above hip → shoulder.y < hip.y → dy < 0.
    # Flipped: shoulder.y > hip.y → dy > 0.
    # Use a 200px threshold to only flag truly anomalous rotations,
    # not normal body orientation variation (shoulder-hip y difference
    # in SLP top-down view is typically 100-250px).
    if abs(dy) > 200.0:  # More than 200px axis direction change → suspicious.
        face_up = True
        error_codes.append("face_up_180_rotation_suspected")

    return face_up, error_codes


def _axis_status_from_counts(
    visible_key: int,
    total_key: int,
    visible_total: int,
    total_total: int,
) -> tuple[str, float, list[str]]:
    """Determine axis status and confidence from joint visibility counts.

    Returns (status, confidence, error_codes).
    """
    error_codes: list[str] = []

    # Key joints for axis: shoulders (2) + hips (2) = 4 minimum.
    # Total body: all 14.
    key_visible_frac = visible_key / max(total_key, 1)
    total_visible_frac = visible_total / max(total_total, 1)

    if visible_total == 0:
        return AxisStatus.REJECT_NO_VALID_JOINTS.value, 0.0, ["no_visible_joints"]

    if visible_key == 0:
        error_codes.append("no_visible_key_joints")
        if visible_total >= _MIN_VISIBLE_FOR_AXIS:
            confidence = total_visible_frac * 0.3
            return AxisStatus.UNCERTAIN_MISSING_BOTH.value, confidence, error_codes
        else:
            return AxisStatus.REJECT_INSUFFICIENT_VISIBLE.value, 0.0, error_codes + ["insufficient_visible"]

    if visible_key == 1:
        error_codes.append("only_one_key_joint_visible")
        confidence = key_visible_frac * 0.5
        return AxisStatus.UNCERTAIN_MISSING_BOTH.value, confidence, error_codes

    # 2 or more key joints visible.
    if key_visible_frac >= 1.0:
        return AxisStatus.ACCEPT_FULL.value, 1.0, []
    elif key_visible_frac >= 0.5:
        confidence = key_visible_frac
        return AxisStatus.ACCEPT_PARTIAL.value, confidence, []
    else:
        confidence = key_visible_frac * 0.7
        return AxisStatus.UNCERTAIN_MISSING_BOTH.value, confidence, error_codes


def _bbox_status_from_bbox(
    bbox: BoundingBox,
    visible_count: int,
    total_count: int,
) -> tuple[str, list[str]]:
    """Determine bbox status."""
    error_codes: list[str] = []
    if not bbox.valid:
        return BboxStatus.REJECT.value, ["bbox_invalid_span"]
    if visible_count < _MIN_VISIBLE_FOR_AXIS:
        return BboxStatus.UNCERTAIN.value, ["bbox_low_visibility"]
    visible_frac = visible_count / max(total_count, 1)
    if visible_frac >= 0.5:
        return BboxStatus.ACCEPT.value, []
    else:
        return BboxStatus.UNCERTAIN.value, ["bbox_partial_visibility"]


# ---------------------------------------------------------------------------
# Main computation function
# ---------------------------------------------------------------------------


def compute_body_geometry(
    sample_id: str,
    joints: list["JointCoords"],
    *,
    task_id: str = DEFAULT_TASK_ID,
    adapter_version: str = ADAPTER_VERSION,
    generator: str = DEFAULT_GENERATOR,
    created_at: str | None = None,
    width: int = RGB_WIDTH,
    height: int = RGB_HEIGHT,
    extreme_frame_jump: bool = False,
    anomalous_bone_length: bool = False,
) -> BodyGeometryResult:
    """Compute body axis, bbox, and orientation for one frame.

    Parameters
    ----------
    sample_id:
        Sample identifier for provenance.
    joints:
        List of 14 JointCoords (J0 RGB), ordered per JOINT_NAMES.
    task_id, adapter_version, generator, created_at:
        Provenance metadata.
    width, height:
        Image dimensions for out-of-bounds checking.
    extreme_frame_jump:
        True if this frame was flagged as having an extreme coordinate jump
        in a previous pass (e.g., from A07 anomaly detection).
    anomalous_bone_length:
        True if this frame was flagged as having an anomalous bone segment
        length in a previous pass.

    Returns
    -------
    BodyGeometryResult
        Deterministic geometric result with quality fields.
        Never silently imputes missing joints.
    """
    from datetime import datetime, timezone
    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()

    assert len(joints) == 14, f"Expected 14 joints, got {len(joints)}"

    # Step 1: Compute per-joint validity.
    per_jv: list[JointValidity] = []
    for idx, jc in enumerate(joints):
        jv = JointValidity(
            index=idx,
            name=JOINT_NAMES[idx],
            x=jc.x,
            y=jc.y,
            confidence=jc.confidence,
            is_valid=jc.is_valid,
            is_visible=_is_visible(jc),
            is_occluded=jc.is_occluded,
            is_out_of_bounds=_is_out_of_bounds(jc, width=width, height=height),
            status="visible" if _is_visible(jc) else (
                "occluded" if jc.is_occluded else "invalid"
            ),
        )
        # Override out_of_bounds status after construction.
        if jv.is_out_of_bounds:
            jv.status = "out_of_bounds"
        per_jv.append(jv)

    # Step 2: Aggregate counts.
    total_joints = len(joints)
    valid_joints = sum(1 for jv in per_jv if jv.is_valid)
    visible_joints = sum(1 for jv in per_jv if jv.is_visible)
    occluded_joints = sum(1 for jv in per_jv if jv.is_occluded)
    out_of_bounds_joints = sum(1 for jv in per_jv if jv.is_out_of_bounds)
    missing_joints = sum(1 for jv in per_jv if not jv.is_valid)

    # Step 3: Compute axes.
    shoulder_axis = _shoulder_axis(joints, width=width, height=height)
    hip_axis = _hip_axis(joints, width=width, height=height)
    longitudinal_axis = _longitudinal_axis(shoulder_axis, hip_axis, width=width, height=height)
    body_center = _body_center(shoulder_axis, hip_axis)

    # Step 4: Compute orientation.
    orient_deg, orient_status, orient_conf = _compute_orientation(
        longitudinal_axis, shoulder_axis, hip_axis,
    )

    # Step 5: Compute bbox.
    bbox = _compute_bbox(joints, per_jv, width=width, height=height)

    # Step 6: Detect flips.
    flip_detected, flip_codes = _detect_left_right_flip(shoulder_axis, hip_axis)
    face_up_detected, face_up_codes = _detect_face_up(longitudinal_axis)

    # Step 7: Determine axis status.
    key_joints = [
        per_jv[J_R_SHOULDER], per_jv[J_L_SHOULDER],
        per_jv[J_R_HIP], per_jv[J_L_HIP],
    ]
    visible_key = sum(1 for jv in key_joints if jv.is_visible)
    axis_status_str, axis_conf, axis_codes = _axis_status_from_counts(
        visible_key, 4, visible_joints, total_joints,
    )

    # Step 8: Determine bbox status.
    bbox_status_str, bbox_codes = _bbox_status_from_bbox(bbox, visible_joints, total_joints)

    # Step 9: Assemble error codes.
    error_codes: list[str] = []
    if flip_detected:
        error_codes.extend(flip_codes)
    if face_up_detected:
        error_codes.extend(face_up_codes)
    error_codes.extend(axis_codes)
    error_codes.extend(bbox_codes)
    if extreme_frame_jump:
        error_codes.append("extreme_frame_jump")
    if anomalous_bone_length:
        error_codes.append("anomalous_bone_length")
    if visible_joints < _MIN_VISIBLE_FOR_AXIS:
        error_codes.append("insufficient_visible_joints")
    if out_of_bounds_joints > 0:
        error_codes.append(f"out_of_bounds_joints:{out_of_bounds_joints}")
    if missing_joints > 0:
        error_codes.append(f"missing_joints:{missing_joints}")

    # Step 10: Overall confidence and status.
    # Orientation "ambiguous" is a low-penalty flag (body not perfectly vertical);
    # it is reflected in orient_conf but does not override a high-confidence result.
    overall_conf = (axis_conf + orient_conf + (1.0 if bbox.valid else 0.0)) / 3.0
    if axis_status_str in (AxisStatus.REJECT_NO_VALID_JOINTS.value,
                           AxisStatus.REJECT_INSUFFICIENT_VISIBLE.value):
        overall_status = "reject"
    elif axis_status_str.startswith("reject") or bbox_status_str == BboxStatus.REJECT.value:
        overall_status = "reject"
    elif axis_status_str.startswith("uncertain"):
        # Explicit uncertain axis → overall uncertain regardless of confidence.
        overall_status = "uncertain"
    elif overall_conf >= _CONFIDENCE_ACCEPT:
        overall_status = "accept"
    elif overall_conf >= _CONFIDENCE_UNCERTAIN:
        overall_status = "uncertain"
    else:
        overall_status = "reject"

    return BodyGeometryResult(
        sample_id=sample_id,
        coordinate_frame="J0_RGB",
        shoulder_axis=shoulder_axis,
        hip_axis=hip_axis,
        longitudinal_axis=longitudinal_axis,
        body_center=body_center,
        orientation_degrees=orient_deg,
        orientation_status=orient_status,
        orientation_confidence=orient_conf,
        bbox=bbox,
        bbox_status=bbox_status_str,
        axis_status=axis_status_str,
        overall_confidence=overall_conf,
        overall_status=overall_status,
        total_joints=total_joints,
        valid_joints=valid_joints,
        visible_joints=visible_joints,
        occluded_joints=occluded_joints,
        out_of_bounds_joints=out_of_bounds_joints,
        missing_joints=missing_joints,
        per_joint_validity=[asdict(jv) for jv in per_jv],
        error_codes=tuple(error_codes),
        left_right_flip_detected=flip_detected,
        face_up_detected=face_up_detected,
        extreme_frame_jump=extreme_frame_jump,
        anomalous_bone_length=anomalous_bone_length,
        task_id=task_id,
        adapter_version=adapter_version,
        generator=generator,
        created_at=created_at,
    )


# ---------------------------------------------------------------------------
# Flattened row output for CSV
# ---------------------------------------------------------------------------

def geometry_result_to_csv_row(result: BodyGeometryResult) -> dict:
    """Flatten a BodyGeometryResult into a flat dict for CSV output."""
    sa = result.shoulder_axis
    ha = result.hip_axis
    la = result.longitudinal_axis
    bc = result.body_center
    bb = result.bbox

    return {
        "sample_id": result.sample_id,
        "coordinate_frame": result.coordinate_frame,
        # Shoulder axis
        "shoulder_axis_start_x": sa.start.x if sa.start.valid else None,
        "shoulder_axis_start_y": sa.start.y if sa.start.valid else None,
        "shoulder_axis_end_x": sa.end.x if sa.end.valid else None,
        "shoulder_axis_end_y": sa.end.y if sa.end.valid else None,
        "shoulder_axis_midpoint_x": sa.midpoint.x if sa.midpoint.valid else None,
        "shoulder_axis_midpoint_y": sa.midpoint.y if sa.midpoint.valid else None,
        "shoulder_axis_valid": sa.direction_valid,
        # Hip axis
        "hip_axis_start_x": ha.start.x if ha.start.valid else None,
        "hip_axis_start_y": ha.start.y if ha.start.valid else None,
        "hip_axis_end_x": ha.end.x if ha.end.valid else None,
        "hip_axis_end_y": ha.end.y if ha.end.valid else None,
        "hip_axis_midpoint_x": ha.midpoint.x if ha.midpoint.valid else None,
        "hip_axis_midpoint_y": ha.midpoint.y if ha.midpoint.valid else None,
        "hip_axis_valid": ha.direction_valid,
        # Longitudinal axis
        "longitudinal_axis_start_x": la.start.x if la.start.valid else None,
        "longitudinal_axis_start_y": la.start.y if la.start.valid else None,
        "longitudinal_axis_end_x": la.end.x if la.end.valid else None,
        "longitudinal_axis_end_y": la.end.y if la.end.valid else None,
        "longitudinal_axis_midpoint_x": la.midpoint.x if la.midpoint.valid else None,
        "longitudinal_axis_midpoint_y": la.midpoint.y if la.midpoint.valid else None,
        "longitudinal_axis_valid": la.direction_valid,
        # Body center
        "body_center_x": bc.x if bc.valid else None,
        "body_center_y": bc.y if bc.valid else None,
        "body_center_valid": bc.valid,
        # Orientation
        "orientation_degrees": result.orientation_degrees,
        "orientation_status": result.orientation_status,
        "orientation_confidence": result.orientation_confidence,
        # Bbox
        "bbox_x_min": bb.x_min if bb.valid else None,
        "bbox_y_min": bb.y_min if bb.valid else None,
        "bbox_x_max": bb.x_max if bb.valid else None,
        "bbox_y_max": bb.y_max if bb.valid else None,
        "bbox_width": bb.width if bb.valid else None,
        "bbox_height": bb.height if bb.valid else None,
        "bbox_center_x": bb.center_x if bb.valid else None,
        "bbox_center_y": bb.center_y if bb.valid else None,
        "bbox_valid": bb.valid,
        "bbox_status": result.bbox_status,
        # Quality
        "axis_status": result.axis_status,
        "overall_confidence": result.overall_confidence,
        "overall_status": result.overall_status,
        "total_joints": result.total_joints,
        "valid_joints": result.valid_joints,
        "visible_joints": result.visible_joints,
        "occluded_joints": result.occluded_joints,
        "out_of_bounds_joints": result.out_of_bounds_joints,
        "missing_joints": result.missing_joints,
        # Flags
        "left_right_flip_detected": result.left_right_flip_detected,
        "face_up_detected": result.face_up_detected,
        "extreme_frame_jump": result.extreme_frame_jump,
        "anomalous_bone_length": result.anomalous_bone_length,
        # Error codes
        "error_codes": ";".join(result.error_codes) if result.error_codes else "",
    }


# ---------------------------------------------------------------------------
# CSV column names (in order)
# ---------------------------------------------------------------------------

GEOMETRY_CSV_COLUMNS = [
    "sample_id",
    "coordinate_frame",
    # Shoulder axis
    "shoulder_axis_start_x", "shoulder_axis_start_y",
    "shoulder_axis_end_x", "shoulder_axis_end_y",
    "shoulder_axis_midpoint_x", "shoulder_axis_midpoint_y",
    "shoulder_axis_valid",
    # Hip axis
    "hip_axis_start_x", "hip_axis_start_y",
    "hip_axis_end_x", "hip_axis_end_y",
    "hip_axis_midpoint_x", "hip_axis_midpoint_y",
    "hip_axis_valid",
    # Longitudinal axis
    "longitudinal_axis_start_x", "longitudinal_axis_start_y",
    "longitudinal_axis_end_x", "longitudinal_axis_end_y",
    "longitudinal_axis_midpoint_x", "longitudinal_axis_midpoint_y",
    "longitudinal_axis_valid",
    # Body center
    "body_center_x", "body_center_y", "body_center_valid",
    # Orientation
    "orientation_degrees", "orientation_status", "orientation_confidence",
    # Bbox
    "bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max",
    "bbox_width", "bbox_height", "bbox_center_x", "bbox_center_y",
    "bbox_valid", "bbox_status",
    # Quality
    "axis_status", "overall_confidence", "overall_status",
    "total_joints", "valid_joints", "visible_joints",
    "occluded_joints", "out_of_bounds_joints", "missing_joints",
    # Flags
    "left_right_flip_detected", "face_up_detected",
    "extreme_frame_jump", "anomalous_bone_length",
    # Error codes
    "error_codes",
]


# ---------------------------------------------------------------------------
# Schema export
# ---------------------------------------------------------------------------

def geometry_schema_dict() -> dict:
    """Return the geometry output schema as a dict for JSON serialization."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "SLPBodyGeometrySchema",
        "version": "slp_body_geometry_v0.1",
        "type": "object",
        "description": (
            "Body axis and bounding-box geometry output for one SLP frame (J0 RGB space). "
            "Generated by TASK-SLP-A08-BODY-AXIS-GEOMETRY-v0.1. "
            "NOT a region ground truth."
        ),
        "properties": {
            "sample_id": {"type": "string"},
            "coordinate_frame": {"type": "string", "const": "J0_RGB"},
            "shoulder_axis": {"$ref": "#/$defs/BodyAxis"},
            "hip_axis": {"$ref": "#/$defs/BodyAxis"},
            "longitudinal_axis": {"$ref": "#/$defs/BodyAxis"},
            "body_center": {"$ref": "#/$defs/Point2D"},
            "orientation_degrees": {"type": ["number", "null"]},
            "orientation_status": {"type": "string"},
            "orientation_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "bbox": {"$ref": "#/$defs/BoundingBox"},
            "bbox_status": {"type": "string"},
            "axis_status": {"type": "string"},
            "overall_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "overall_status": {"type": "string", "enum": ["accept", "uncertain", "reject"]},
            "total_joints": {"type": "integer", "minimum": 14, "maximum": 14},
            "valid_joints": {"type": "integer", "minimum": 0, "maximum": 14},
            "visible_joints": {"type": "integer", "minimum": 0, "maximum": 14},
            "occluded_joints": {"type": "integer", "minimum": 0, "maximum": 14},
            "out_of_bounds_joints": {"type": "integer", "minimum": 0, "maximum": 14},
            "missing_joints": {"type": "integer", "minimum": 0, "maximum": 14},
            "per_joint_validity": {
                "type": "array",
                "items": {"$ref": "#/$defs/JointValidity"},
            },
            "error_codes": {
                "type": "array",
                "items": {"type": "string"},
            },
            "left_right_flip_detected": {"type": "boolean"},
            "face_up_detected": {"type": "boolean"},
            "extreme_frame_jump": {"type": "boolean"},
            "anomalous_bone_length": {"type": "boolean"},
            "task_id": {"type": "string"},
            "adapter_version": {"type": "string"},
            "generator": {"type": "string"},
            "created_at": {"type": "string"},
        },
        "required": [
            "sample_id", "coordinate_frame",
            "shoulder_axis", "hip_axis", "longitudinal_axis", "body_center",
            "orientation_degrees", "orientation_status", "orientation_confidence",
            "bbox", "bbox_status",
            "axis_status", "overall_confidence", "overall_status",
            "total_joints", "valid_joints", "visible_joints",
            "occluded_joints", "out_of_bounds_joints", "missing_joints",
            "per_joint_validity", "error_codes",
            "left_right_flip_detected", "face_up_detected",
            "extreme_frame_jump", "anomalous_bone_length",
            "task_id", "adapter_version", "generator", "created_at",
        ],
        "$defs": {
            "Point2D": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "valid": {"type": "boolean"},
                },
                "required": ["x", "y", "valid"],
            },
            "BodyAxis": {
                "type": "object",
                "properties": {
                    "start": {"$ref": "#/$defs/Point2D"},
                    "end": {"$ref": "#/$defs/Point2D"},
                    "midpoint": {"$ref": "#/$defs/Point2D"},
                    "direction_valid": {"type": "boolean"},
                },
                "required": ["start", "end", "midpoint", "direction_valid"],
            },
            "BoundingBox": {
                "type": "object",
                "properties": {
                    "x_min": {"type": "number"},
                    "y_min": {"type": "number"},
                    "x_max": {"type": "number"},
                    "y_max": {"type": "number"},
                    "width": {"type": "number"},
                    "height": {"type": "number"},
                    "center_x": {"type": "number"},
                    "center_y": {"type": "number"},
                    "valid": {"type": "boolean"},
                },
                "required": ["x_min", "y_min", "x_max", "y_max",
                             "width", "height", "center_x", "center_y", "valid"],
            },
            "JointValidity": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "name": {"type": "string"},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "confidence": {"type": "number"},
                    "is_valid": {"type": "boolean"},
                    "is_visible": {"type": "boolean"},
                    "is_occluded": {"type": "boolean"},
                    "is_out_of_bounds": {"type": "boolean"},
                    "status": {"type": "string"},
                },
                "required": ["index", "name", "x", "y", "confidence",
                             "is_valid", "is_visible", "is_occluded",
                             "is_out_of_bounds", "status"],
            },
        },
    }
