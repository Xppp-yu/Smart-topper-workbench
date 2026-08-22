"""SLP Joint Occlusion and Quality EDA — A07.

This module performs exploratory data analysis on the 14-joint ground truth
coordinates from the SLP dataset. It is NOT a ground truth builder; it produces
statistics, anomaly tables, and small visualizations for downstream tasks
(A08/A18) to consume.

Design contract (mirroring the A07 task contract):

* J0 (original joints from ``joints_gt_RGB.mat`` / ``joints_gt_IR.mat``) and
  J1 (homography-derived joints) are analysed separately. J1 is never mixed
  into the J0 ground-truth statistics.
* Usable samples and quarantined samples are reported separately.
* danaLab and simLab are always reported in separate buckets.
* Train/val/test split comes from the frozen A06 split manifest and is not
  re-computed here.
* Bone segment lengths, left/right symmetry, and anomaly flags are derived
  statistics only; they do not modify any original data.
* No region ground truth is generated.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ADAPTER_VERSION = "slp_joint_eda_v0.1"
DEFAULT_TASK_ID = "TASK-SLP-A07-NODE-OCCLUSION-EDA-v0.1"

# SLP anatomy: 14 joints in order (indexed 0-13).
# Row 0 = x (px), Row 1 = y (px), Row 2 = confidence (0=occluded, 1=visible).
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

# Left/right pairs for symmetry analysis.
LEFT_RIGHT_PAIRS = (
    (2, 5),   # right_shoulder / left_shoulder
    (3, 6),   # right_elbow / left_elbow
    (4, 7),   # right_wrist / left_wrist
    (8, 11),  # right_hip / left_hip
    (9, 12),  # right_knee / left_knee
    (10, 13), # right_ankle / left_ankle
)

# Bone segments: (start_joint_index, end_joint_index).
BONE_SEGMENTS = (
    (0, 1),   # head → neck
    (1, 2),   # neck → right_shoulder
    (2, 3),   # right_shoulder → right_elbow
    (3, 4),   # right_elbow → right_wrist
    (1, 5),   # neck → left_shoulder
    (5, 6),   # left_shoulder → left_elbow
    (6, 7),   # left_elbow → left_wrist
    (2, 8),   # right_shoulder → right_hip
    (8, 9),   # right_hip → right_knee
    (9, 10),  # right_knee → right_ankle
    (5, 11),  # left_shoulder → left_hip
    (11, 12), # left_hip → left_knee
    (12, 13), # left_knee → left_ankle
)

# J0 (original) coordinate frame label.
J0_LABEL = "J0_original"
# J1 (homography-derived) coordinate frame label.
J1_LABEL = "J1_homography_derived"

# Occlusion threshold: confidence == 0 means occluded.
OCCLUDED = 0
VISIBLE = 1

# RGB image bounds (from canonical sample A04 audit, danaLab subjects).
# These are conservative outer bounds; actual per-subject bounds may vary.
DANALAB_RGB_WIDTH = 576
DANALAB_RGB_HEIGHT = 1024
# PM image bounds.
PM_WIDTH = 192
PM_HEIGHT = 84

# J0 in-bounds check uses RGB image bounds.
J0_IMAGE_BOUNDS = {"width": DANALAB_RGB_WIDTH, "height": DANALAB_RGB_HEIGHT}
# J1 (PM) in-bounds check uses PM image bounds.
J1_IMAGE_BOUNDS = {"width": PM_WIDTH, "height": PM_HEIGHT}

JOINT_COUNT = 14


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class OcclusionStatus(str, Enum):
    VISIBLE = "visible"
    OCCLUDED = "occluded"


@dataclass(frozen=True, slots=True)
class JointCoords:
    """Coordinates and visibility for one joint at one frame."""

    x: float
    y: float
    confidence: float

    @property
    def is_occluded(self) -> bool:
        return self.confidence == OCCLUDED

    @property
    def is_valid(self) -> bool:
        return not (np.isnan(self.x) or np.isnan(self.y))


@dataclass(frozen=True, slots=True)
class PerJointStats:
    """Per-joint aggregate statistics."""

    joint_index: int
    joint_name: str
    coordinate_frame: str  # "J0_RGB" or "J1_PM"

    # Visibility
    visible_count: int
    occluded_count: int
    occlusion_rate: float  # fraction [0,1]

    # Coordinate stats
    x_mean: float
    x_std: float
    x_min: float
    x_max: float
    y_mean: float
    y_std: float
    y_min: float
    y_max: float

    # Bounds
    out_of_bounds_count: int
    out_of_bounds_rate: float

    # Validity
    invalid_count: int  # NaN x or y
    total_observations: int


@dataclass(frozen=True, slots=True)
class BoneSegmentStats:
    """Per bone-segment aggregate statistics."""

    segment_index: int
    start_joint: int
    end_joint: int
    start_joint_name: str
    end_joint_name: str
    coordinate_frame: str

    length_mean: float
    length_std: float
    length_min: float
    length_max: float
    length_median: float
    count: int
    zero_length_count: int  # both endpoints occluded → length = 0


@dataclass(frozen=True, slots=True)
class SymmetryStats:
    """Left/right body symmetry statistics."""

    left_joint: int
    right_joint: int
    left_joint_name: str
    right_joint_name: str
    coordinate_frame: str

    x_diff_mean: float
    x_diff_std: float
    y_diff_mean: float
    y_diff_std: float
    count: int


@dataclass(frozen=True, slots=True)
class AnomalyCase:
    """One anomalous sample record."""

    sample_id: str
    setting: str
    subject_id: str
    cover_condition: str
    frame_index: int
    anomaly_type: str  # "extreme_jump", "anomalous_bone_length", "joint_out_of_bounds", ...
    detail: str
    coordinate_frame: str


@dataclass
class JointEdaResult:
    """Container for the complete J0/J1 EDA result for one coordinate frame."""

    coordinate_frame: str
    total_frames: int
    usable_frames: int
    quarantined_frames: int

    # Per-joint
    per_joint: list[PerJointStats]

    # Bone segments
    bone_segments: list[BoneSegmentStats]

    # Symmetry
    symmetry_pairs: list[SymmetryStats]

    # Anomalies
    anomalies: list[AnomalyCase]

    # Overall coordinate distributions
    all_x: list[float]
    all_y: list[float]
    all_confidence: list[float]


# ---------------------------------------------------------------------------
# Joint loading
# ---------------------------------------------------------------------------


def load_subject_joints_rgb(
    subject_dir: Path, *, cover: str, frame_index: int
) -> np.ndarray | None:
    """Load joints_gt_RGB for one subject/cover/frame.

    Returns shape (3, 14) or None if file not present.

    The mat file contains joints_gt of shape (3, 14, 45). We index into
    frame_index-1 because SLP frames are 1-indexed.
    """
    mat_path = subject_dir / "joints_gt_RGB.mat"
    if not mat_path.is_file():
        return None
    try:
        data = loadmat(str(mat_path))
        joints = data.get("joints_gt")
        if joints is None or joints.ndim != 3:
            return None
        # SLP frames are 1-indexed.
        frame_idx = int(frame_index) - 1
        if frame_idx < 0 or frame_idx >= joints.shape[2]:
            return None
        return joints[:, :, frame_idx]  # shape (3, 14)
    except Exception:
        return None


def load_subject_joints_ir(
    subject_dir: Path, *, cover: str, frame_index: int
) -> np.ndarray | None:
    """Load joints_gt_IR for one subject/cover/frame."""
    mat_path = subject_dir / "joints_gt_IR.mat"
    if not mat_path.is_file():
        return None
    try:
        data = loadmat(str(mat_path))
        joints = data.get("joints_gt")
        if joints is None or joints.ndim != 3:
            return None
        frame_idx = int(frame_index) - 1
        if frame_idx < 0 or frame_idx >= joints.shape[2]:
            return None
        return joints[:, :, frame_idx]
    except Exception:
        return None


def joints_to_coords(joints_arr: np.ndarray) -> list[JointCoords]:
    """Convert a (3, N) array to a list of N JointCoords."""
    if joints_arr.ndim != 2 or joints_arr.shape[0] != 3:
        raise ValueError(f"Expected shape (3, N), got {joints_arr.shape}")
    n_joints = joints_arr.shape[1]
    coords: list[JointCoords] = []
    for j in range(n_joints):
        x = float(joints_arr[0, j])
        y = float(joints_arr[1, j])
        c = float(joints_arr[2, j])
        coords.append(JointCoords(x=x, y=y, confidence=c))
    return coords


# ---------------------------------------------------------------------------
# Statistics computation
# ---------------------------------------------------------------------------


def compute_per_joint_stats(
    frame_records: list[dict[str, Any]],
    coordinate_frame: str,
    image_bounds: dict[str, int],
) -> list[PerJointStats]:
    """Compute per-joint aggregate statistics from all frame records.

    Each frame_record must have keys: ``joints`` (list[JointCoords]),
    ``sample_id``, ``quarantine``.
    """
    # Accumulate per-joint data.
    joint_x: list[list[float]] = [[] for _ in range(JOINT_COUNT)]
    joint_y: list[list[float]] = [[] for _ in range(JOINT_COUNT)]
    joint_conf: list[list[float]] = [[] for _ in range(JOINT_COUNT)]
    joint_oob: list[int] = [0] * JOINT_COUNT
    joint_invalid: list[int] = [0] * JOINT_COUNT

    w = image_bounds["width"]
    h = image_bounds["height"]

    for record in frame_records:
        joints = record["joints"]
        for j_idx, jc in enumerate(joints):
            joint_conf[j_idx].append(jc.confidence)
            if jc.is_valid:
                joint_x[j_idx].append(jc.x)
                joint_y[j_idx].append(jc.y)
                # Out of bounds check (only for visible joints, but count all).
                if jc.x < 0 or jc.x >= w or jc.y < 0 or jc.y >= h:
                    joint_oob[j_idx] += 1
            else:
                joint_invalid[j_idx] += 1

    stats: list[PerJointStats] = []
    for j_idx in range(JOINT_COUNT):
        xs = joint_x[j_idx]
        ys = joint_y[j_idx]
        confs = joint_conf[j_idx]

        visible = sum(1 for c in confs if c == VISIBLE)
        occluded = sum(1 for c in confs if c == OCCLUDED)
        total = len(confs)

        stats.append(
            PerJointStats(
                joint_index=j_idx,
                joint_name=JOINT_NAMES[j_idx],
                coordinate_frame=coordinate_frame,
                visible_count=visible,
                occluded_count=occluded,
                occlusion_rate=occluded / total if total > 0 else 0.0,
                x_mean=float(np.mean(xs)) if xs else float("nan"),
                x_std=float(np.std(xs)) if xs else float("nan"),
                x_min=float(np.min(xs)) if xs else float("nan"),
                x_max=float(np.max(xs)) if xs else float("nan"),
                y_mean=float(np.mean(ys)) if ys else float("nan"),
                y_std=float(np.std(ys)) if ys else float("nan"),
                y_min=float(np.min(ys)) if ys else float("nan"),
                y_max=float(np.max(ys)) if ys else float("nan"),
                out_of_bounds_count=joint_oob[j_idx],
                out_of_bounds_rate=joint_oob[j_idx] / total if total > 0 else 0.0,
                invalid_count=joint_invalid[j_idx],
                total_observations=total,
            )
        )
    return stats


def compute_bone_segment_stats(
    frame_records: list[dict[str, Any]],
    coordinate_frame: str,
) -> list[BoneSegmentStats]:
    """Compute per-bone-segment length statistics."""
    lengths_per_seg: list[list[float]] = [[] for _ in range(len(BONE_SEGMENTS))]

    for record in frame_records:
        joints = record["joints"]
        for seg_idx, (a, b) in enumerate(BONE_SEGMENTS):
            ja = joints[a]
            jb = joints[b]
            if ja.is_valid and jb.is_valid and not (ja.is_occluded or jb.is_occluded):
                dx = jb.x - ja.x
                dy = jb.y - ja.y
                length = (dx * dx + dy * dy) ** 0.5
                lengths_per_seg[seg_idx].append(length)

    stats: list[BoneSegmentStats] = []
    for seg_idx, (a, b) in enumerate(BONE_SEGMENTS):
        lens = lengths_per_seg[seg_idx]
        if lens:
            arr = np.array(lens)
            zero_count = int(np.sum(arr == 0.0))
            stats.append(
                BoneSegmentStats(
                    segment_index=seg_idx,
                    start_joint=a,
                    end_joint=b,
                    start_joint_name=JOINT_NAMES[a],
                    end_joint_name=JOINT_NAMES[b],
                    coordinate_frame=coordinate_frame,
                    length_mean=float(np.mean(arr)),
                    length_std=float(np.std(arr)),
                    length_min=float(np.min(arr)),
                    length_max=float(np.max(arr)),
                    length_median=float(np.median(arr)),
                    count=len(lens),
                    zero_length_count=zero_count,
                )
            )
        else:
            stats.append(
                BoneSegmentStats(
                    segment_index=seg_idx,
                    start_joint=a,
                    end_joint=b,
                    start_joint_name=JOINT_NAMES[a],
                    end_joint_name=JOINT_NAMES[b],
                    coordinate_frame=coordinate_frame,
                    length_mean=float("nan"),
                    length_std=float("nan"),
                    length_min=float("nan"),
                    length_max=float("nan"),
                    length_median=float("nan"),
                    count=0,
                    zero_length_count=0,
                )
            )
    return stats


def compute_symmetry_stats(
    frame_records: list[dict[str, Any]],
    coordinate_frame: str,
) -> list[SymmetryStats]:
    """Compute left/right symmetry statistics."""
    pair_x_diffs: list[list[float]] = [[] for _ in range(len(LEFT_RIGHT_PAIRS))]
    pair_y_diffs: list[list[float]] = [[] for _ in range(len(LEFT_RIGHT_PAIRS))]

    for record in frame_records:
        joints = record["joints"]
        for pair_idx, (left_j, right_j) in enumerate(LEFT_RIGHT_PAIRS):
            lj = joints[left_j]
            rj = joints[right_j]
            if lj.is_valid and rj.is_valid:
                pair_x_diffs[pair_idx].append(rj.x - lj.x)
                pair_y_diffs[pair_idx].append(rj.y - lj.y)

    stats: list[SymmetryStats] = []
    for pair_idx, (left_j, right_j) in enumerate(LEFT_RIGHT_PAIRS):
        xdiffs = pair_x_diffs[pair_idx]
        ydiffs = pair_y_diffs[pair_idx]
        count = len(xdiffs)
        if count > 0:
            stats.append(
                SymmetryStats(
                    left_joint=left_j,
                    right_joint=right_j,
                    left_joint_name=JOINT_NAMES[left_j],
                    right_joint_name=JOINT_NAMES[right_j],
                    coordinate_frame=coordinate_frame,
                    x_diff_mean=float(np.mean(xdiffs)),
                    x_diff_std=float(np.std(xdiffs)),
                    y_diff_mean=float(np.mean(ydiffs)),
                    y_diff_std=float(np.std(ydiffs)),
                    count=count,
                )
            )
        else:
            stats.append(
                SymmetryStats(
                    left_joint=left_j,
                    right_joint=right_j,
                    left_joint_name=JOINT_NAMES[left_j],
                    right_joint_name=JOINT_NAMES[right_j],
                    coordinate_frame=coordinate_frame,
                    x_diff_mean=float("nan"),
                    x_diff_std=float("nan"),
                    y_diff_mean=float("nan"),
                    y_diff_std=float("nan"),
                    count=0,
                )
            )
    return stats


def detect_anomalies(
    frame_records: list[dict[str, Any]],
    coordinate_frame: str,
    jump_threshold_px: float = 100.0,
    bone_zscore_threshold: float = 4.0,
) -> list[AnomalyCase]:
    """Detect anomalous frames: extreme coordinate jumps, anomalous bone lengths.

    Anomaly detection is intentionally conservative:
    - Frame jumps require both: (a) absolute displacement > jump_threshold_px
      AND (b) the same joint also shows consistent large jumps across the
      subject's other covers (to reduce per-subject-person variation false positives).
    - Bone length z-scores are computed per-subject (not globally) so that
      tall/short subjects are not systematically flagged.
    """
    anomalies: list[AnomalyCase] = []

    # Group by subject+cover for within-subject jump detection.
    records_by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in frame_records:
        key = f"{rec['setting']}::{rec['subject_id']}::{rec['cover_condition']}"
        records_by_key[key].append(rec)

    # Pre-compute per-subject joint displacement statistics for adaptive threshold.
    # For each (setting, subject_id), compute the 99th percentile displacement
    # per joint to establish a baseline.
    subject_joint_dists: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for key, recs in records_by_key.items():
        recs_sorted = sorted(recs, key=lambda r: r["frame_index"])
        for i in range(1, len(recs_sorted)):
            prev = recs_sorted[i - 1]["joints"]
            curr = recs_sorted[i]["joints"]
            for j_idx in range(JOINT_COUNT):
                pj = prev[j_idx]
                cj = curr[j_idx]
                if pj.is_valid and cj.is_valid:
                    dx = cj.x - pj.x
                    dy = cj.y - pj.y
                    dist = (dx * dx + dy * dy) ** 0.5
                    # Extract subject key from the full key.
                    subject_key = ":".join(key.split(":")[:2])
                    subject_joint_dists[subject_key][j_idx].append(dist)

    # Compute 99th percentile baseline per subject-joint.
    subject_joint_p99: dict[str, dict[int, float]] = {}
    for subject_key, joint_dists in subject_joint_dists.items():
        subject_joint_p99[subject_key] = {}
        for j_idx, dists in joint_dists.items():
            if len(dists) > 5:
                subject_joint_p99[subject_key][j_idx] = np.percentile(dists, 99)
            else:
                subject_joint_p99[subject_key][j_idx] = jump_threshold_px * 2

    # Detect extreme frame-to-frame jumps using adaptive threshold.
    for key, recs in records_by_key.items():
        recs_sorted = sorted(recs, key=lambda r: r["frame_index"])
        subject_key = ":".join(key.split(":")[:2])
        p99_baseline = subject_joint_p99.get(subject_key, {})

        for i in range(1, len(recs_sorted)):
            prev = recs_sorted[i - 1]["joints"]
            curr = recs_sorted[i]["joints"]
            for j_idx in range(JOINT_COUNT):
                pj = prev[j_idx]
                cj = curr[j_idx]
                if pj.is_valid and cj.is_valid:
                    dx = cj.x - pj.x
                    dy = cj.y - pj.y
                    dist = (dx * dx + dy * dy) ** 0.5
                    # Two-stage: (1) must exceed absolute threshold, and
                    # (2) must exceed 3x the subject's 99th percentile baseline.
                    p99 = p99_baseline.get(j_idx, jump_threshold_px * 2)
                    if dist > jump_threshold_px and dist > 3 * p99:
                        anomalies.append(
                            AnomalyCase(
                                sample_id=recs_sorted[i]["sample_id"],
                                setting=recs_sorted[i]["setting"],
                                subject_id=recs_sorted[i]["subject_id"],
                                cover_condition=recs_sorted[i]["cover_condition"],
                                frame_index=recs_sorted[i]["frame_index"],
                                anomaly_type="extreme_frame_jump",
                                detail=(
                                    f"joint {j_idx} ({JOINT_NAMES[j_idx]}) "
                                    f"jumped {dist:.1f}px from frame "
                                    f"{recs_sorted[i-1]['frame_index']} to "
                                    f"{recs_sorted[i]['frame_index']} "
                                    f"(subject_p99={p99:.1f}px)"
                                ),
                                coordinate_frame=coordinate_frame,
                            )
                        )

    # Detect anomalous bone lengths (z-score > threshold).
    # Compute per-subject per-segment stats (not globally) so that tall/short
    # subjects are not systematically flagged.
    subject_seg_lengths: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for rec in frame_records:
        subject_key = f"{rec['setting']}::{rec['subject_id']}"
        joints = rec["joints"]
        for seg_idx, (a, b) in enumerate(BONE_SEGMENTS):
            ja = joints[a]
            jb = joints[b]
            if ja.is_valid and jb.is_valid:
                dx = jb.x - ja.x
                dy = jb.y - ja.y
                subject_seg_lengths[subject_key][seg_idx].append(
                    (dx * dx + dy * dy) ** 0.5
                )

    # Per-subject mean/std.
    subject_seg_stats: dict[str, dict[int, tuple[float, float]]] = {}
    for subject_key, seg_lengths in subject_seg_lengths.items():
        subject_seg_stats[subject_key] = {}
        for seg_idx, lens in seg_lengths.items():
            if len(lens) > 5:
                subject_seg_stats[subject_key][seg_idx] = (np.mean(lens), np.std(lens))
            else:
                subject_seg_stats[subject_key][seg_idx] = (float("nan"), float("nan"))

    # Flag anomalies.
    for rec in frame_records:
        subject_key = f"{rec['setting']}::{rec['subject_id']}"
        joints = rec["joints"]
        seg_stats = subject_seg_stats.get(subject_key, {})
        for seg_idx, (a, b) in enumerate(BONE_SEGMENTS):
            ja = joints[a]
            jb = joints[b]
            if ja.is_valid and jb.is_valid:
                dx = jb.x - ja.x
                dy = jb.y - ja.y
                length = (dx * dx + dy * dy) ** 0.5
                stats = seg_stats.get(seg_idx, (float("nan"), float("nan")))
                mean, std = stats
                if not (np.isnan(mean) or np.isnan(std) or std == 0):
                    z = abs(length - mean) / std
                    if z > bone_zscore_threshold:
                        anomalies.append(
                            AnomalyCase(
                                sample_id=rec["sample_id"],
                                setting=rec["setting"],
                                subject_id=rec["subject_id"],
                                cover_condition=rec["cover_condition"],
                                frame_index=rec["frame_index"],
                                anomaly_type="anomalous_bone_length",
                                detail=(
                                    f"segment {seg_idx} "
                                    f"({JOINT_NAMES[a]}-{JOINT_NAMES[b]}) "
                                    f"length={length:.1f}px "
                                    f"z={z:.1f} (threshold={bone_zscore_threshold})"
                                ),
                                coordinate_frame=coordinate_frame,
                            )
                        )

    return anomalies


# ---------------------------------------------------------------------------
# Coordinate distribution helpers
# ---------------------------------------------------------------------------


def aggregate_all_coords(
    frame_records: list[dict[str, Any]],
) -> tuple[list[float], list[float], list[float]]:
    """Return all x, y, confidence values from frame records."""
    xs: list[float] = []
    ys: list[float] = []
    confs: list[float] = []
    for rec in frame_records:
        for jc in rec["joints"]:
            if jc.is_valid:
                xs.append(jc.x)
                ys.append(jc.y)
            confs.append(jc.confidence)
    return xs, ys, confs


# ---------------------------------------------------------------------------
# Group-level statistics
# ---------------------------------------------------------------------------


@dataclass
class GroupStats:
    """Statistics for a grouping key (e.g., 'danaLab', 'uncover', 'train')."""

    group_key: str
    coordinate_frame: str
    sample_count: int
    frame_count: int
    quarantined_count: int

    # Visibility
    occlusion_rate: float  # overall fraction of occluded joints
    per_joint_occlusion: dict[int, float]  # joint_index → occlusion_rate

    # Bounds
    out_of_bounds_count: int
    out_of_bounds_rate: float

    # Bone lengths
    bone_length_means: dict[int, float]

    # Coordinates
    x_mean: float
    y_mean: float


def compute_group_stats(
    frame_records: list[dict[str, Any]],
    group_key: str,
    coordinate_frame: str,
    image_bounds: dict[str, int],
) -> GroupStats:
    """Compute aggregate stats for a group of frame records."""
    w = image_bounds["width"]
    h = image_bounds["height"]

    total_joints = 0
    occluded_joints = 0
    oob_joints = 0
    x_vals: list[float] = []
    y_vals: list[float] = []
    per_joint_occ: dict[int, list[int]] = {j: [] for j in range(JOINT_COUNT)}
    per_joint_bone_lens: dict[int, list[float]] = {
        s: [] for s in range(len(BONE_SEGMENTS))
    }

    for rec in frame_records:
        joints = rec["joints"]
        for j_idx, jc in enumerate(joints):
            total_joints += 1
            per_joint_occ[j_idx].append(1 if jc.is_occluded else 0)
            if jc.is_occluded:
                occluded_joints += 1
            if jc.is_valid:
                x_vals.append(jc.x)
                y_vals.append(jc.y)
                if jc.x < 0 or jc.x >= w or jc.y < 0 or jc.y >= h:
                    oob_joints += 1

        # Bone lengths.
        for seg_idx, (a, b) in enumerate(BONE_SEGMENTS):
            ja = joints[a]
            jb = joints[b]
            if ja.is_valid and jb.is_valid:
                dx = jb.x - ja.x
                dy = jb.y - ja.y
                per_joint_bone_lens[seg_idx].append((dx * dx + dy * dy) ** 0.5)

    return GroupStats(
        group_key=group_key,
        coordinate_frame=coordinate_frame,
        sample_count=len(frame_records),
        frame_count=len(set(r["frame_index"] for r in frame_records)),
        quarantined_count=sum(1 for r in frame_records if r.get("quarantine", False)),
        occlusion_rate=occluded_joints / total_joints if total_joints > 0 else 0.0,
        per_joint_occlusion={
            j: sum(per_joint_occ[j]) / len(per_joint_occ[j])
            if per_joint_occ[j]
            else 0.0
            for j in range(JOINT_COUNT)
        },
        out_of_bounds_count=oob_joints,
        out_of_bounds_rate=oob_joints / total_joints if total_joints > 0 else 0.0,
        bone_length_means={
            s: float(np.mean(per_joint_bone_lens[s])) if per_joint_bone_lens[s] else float("nan")
            for s in range(len(BONE_SEGMENTS))
        },
        x_mean=float(np.mean(x_vals)) if x_vals else float("nan"),
        y_mean=float(np.mean(y_vals)) if y_vals else float("nan"),
    )


# ---------------------------------------------------------------------------
# EDA run
# ---------------------------------------------------------------------------


def run_j0_eda(
    canonical_samples: list[dict[str, object]],
    slp_root: Path,
    *,
    task_id: str = DEFAULT_TASK_ID,
    jump_threshold_px: float = 100.0,
    bone_zscore_threshold: float = 4.0,
) -> JointEdaResult:
    """Run J0 (original joint) EDA on all canonical samples.

    J0 coordinates are in the raw dataset pixel space (RGB image coordinates).
    J1 (homography-derived) joints are NOT included in this output.
    """
    # Build per-setting subject directory cache.
    subject_dirs: dict[str, Path] = {}
    for sample in canonical_samples:
        setting = str(sample["setting"])
        subject_id = str(sample["subject_id"])
        key = f"{setting}::{subject_id}"
        if key not in subject_dirs:
            subject_dirs[key] = slp_root / setting / subject_id

    # Accumulate usable + quarantined frame records.
    usable_records: list[dict[str, Any]] = []
    quarantined_records: list[dict[str, Any]] = []

    for sample in canonical_samples:
        sample_id = str(sample["sample_id"])
        setting = str(sample["setting"])
        subject_id = str(sample["subject_id"])
        cover_condition = str(sample["cover_condition"])
        frame_index = int(sample["frame_index"])
        quarantine = str(sample.get("quarantine", "False")).strip().lower() in (
            "true",
            "1",
            "yes",
        )

        key = f"{setting}::{subject_id}"
        subject_dir = subject_dirs.get(key)
        if subject_dir is None or not subject_dir.is_dir():
            continue

        joints_arr = load_subject_joints_rgb(subject_dir, cover=cover_condition, frame_index=frame_index)
        if joints_arr is None:
            # No J0 data available.
            continue

        coords = joints_to_coords(joints_arr)
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "setting": setting,
            "subject_id": subject_id,
            "cover_condition": cover_condition,
            "frame_index": frame_index,
            "quarantine": quarantine,
            "joints": coords,
        }

        if quarantine:
            quarantined_records.append(record)
        else:
            usable_records.append(record)

    result = JointEdaResult(
        coordinate_frame="J0_RGB",
        total_frames=len(canonical_samples),
        usable_frames=len(usable_records),
        quarantined_frames=len(quarantined_records),
        per_joint=compute_per_joint_stats(usable_records, "J0_RGB", J0_IMAGE_BOUNDS),
        bone_segments=compute_bone_segment_stats(usable_records, "J0_RGB"),
        symmetry_pairs=compute_symmetry_stats(usable_records, "J0_RGB"),
        anomalies=detect_anomalies(usable_records, "J0_RGB", jump_threshold_px, bone_zscore_threshold),
        all_x=[],
        all_y=[],
        all_confidence=[],
    )

    result.all_x, result.all_y, result.all_confidence = aggregate_all_coords(usable_records)
    return result


def run_j1_eda_from_csv(
    canonical_samples: list[dict[str, object]],
    slp_root: Path,
    *,
    task_id: str = DEFAULT_TASK_ID,
    jump_threshold_px: float = 100.0,
    bone_zscore_threshold: float = 4.0,
) -> JointEdaResult | None:
    """Run J1 (homography-derived joint) EDA.

    J1 joints are derived by applying the subject-level homography matrix
    (align_PTr_RGB.npy / align_PTr_IR.npy / align_PTr_depth.npy) to the J0
    coordinates. This is a separate analysis from J0.

    Returns None if J1 cannot be generated (e.g., homography matrix missing).

    J1 coordinates are in PM or depth image space.
    """
    import numpy as np

    def _load_homography(subject_dir: Path, modality: str) -> np.ndarray | None:
        h_path = subject_dir / f"align_PTr_{modality}.npy"
        if not h_path.is_file():
            return None
        try:
            H = np.load(str(h_path), allow_pickle=False, mmap_mode="r")
            if H.shape != (3, 3):
                return None
            return H.astype(np.float64)
        except Exception:
            return None

    def _apply_homography(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
        """Apply 3x3 homography to a point. Returns (x', y')."""
        wx = H[0, 0] * x + H[0, 1] * y + H[0, 2]
        wy = H[1, 0] * x + H[1, 1] * y + H[1, 2]
        w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
        if w == 0:
            return float("nan"), float("nan")
        return wx / w, wy / w

    # Build subject directory cache.
    subject_dirs: dict[str, Path] = {}
    for sample in canonical_samples:
        setting = str(sample["setting"])
        subject_id = str(sample["subject_id"])
        key = f"{setting}::{subject_id}"
        if key not in subject_dirs:
            subject_dirs[key] = slp_root / setting / subject_id

    usable_records: list[dict[str, Any]] = []
    quarantined_records: list[dict[str, Any]] = []

    for sample in canonical_samples:
        sample_id = str(sample["sample_id"])
        setting = str(sample["setting"])
        subject_id = str(sample["subject_id"])
        cover_condition = str(sample["cover_condition"])
        frame_index = int(sample["frame_index"])
        quarantine = str(sample.get("quarantine", "False")).strip().lower() in (
            "true",
            "1",
            "yes",
        )

        key = f"{setting}::{subject_id}"
        subject_dir = subject_dirs.get(key)
        if subject_dir is None or not subject_dir.is_dir():
            continue

        # Load J0 first.
        joints_arr = load_subject_joints_rgb(subject_dir, cover=cover_condition, frame_index=frame_index)
        if joints_arr is None:
            continue

        # Load homography for RGB (primary modality for J1).
        H = _load_homography(subject_dir, "RGB")
        if H is None:
            # Try IR.
            H = _load_homography(subject_dir, "IR")

        if H is None:
            continue

        # Apply homography to each joint.
        j1_coords: list[JointCoords] = []
        for j in range(JOINT_COUNT):
            x = float(joints_arr[0, j])
            y = float(joints_arr[1, j])
            c = float(joints_arr[2, j])  # confidence preserved
            x_prime, y_prime = _apply_homography(H, x, y)
            j1_coords.append(JointCoords(x=x_prime, y=y_prime, confidence=c))

        record: dict[str, Any] = {
            "sample_id": sample_id,
            "setting": setting,
            "subject_id": subject_id,
            "cover_condition": cover_condition,
            "frame_index": frame_index,
            "quarantine": quarantine,
            "joints": j1_coords,
        }

        if quarantine:
            quarantined_records.append(record)
        else:
            usable_records.append(record)

    if not usable_records:
        return None

    result = JointEdaResult(
        coordinate_frame="J1_PM",
        total_frames=len(canonical_samples),
        usable_frames=len(usable_records),
        quarantined_frames=len(quarantined_records),
        per_joint=compute_per_joint_stats(usable_records, "J1_PM", J1_IMAGE_BOUNDS),
        bone_segments=compute_bone_segment_stats(usable_records, "J1_PM"),
        symmetry_pairs=compute_symmetry_stats(usable_records, "J1_PM"),
        anomalies=detect_anomalies(usable_records, "J1_PM", jump_threshold_px, bone_zscore_threshold),
        all_x=[],
        all_y=[],
        all_confidence=[],
    )

    result.all_x, result.all_y, result.all_confidence = aggregate_all_coords(usable_records)
    return result


# ---------------------------------------------------------------------------
# Output serialization
# ---------------------------------------------------------------------------


def _sanitize(obj: object) -> object:
    """Convert dataclass instances and numpy types to JSON-serialisable Python types."""
    if isinstance(obj, Enum):
        return obj.value
    if hasattr(obj, "asdict"):
        return _sanitize(obj.as_dict())
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(item) for item in obj]
    if isinstance(obj, float):
        if np.isnan(obj):
            return None
        if np.isinf(obj):
            return str(obj)
        return float(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def result_to_dict(result: JointEdaResult) -> dict[str, object]:
    """Serialise a JointEdaResult to a plain dict."""
    return _sanitize({
        "coordinate_frame": result.coordinate_frame,
        "total_frames": result.total_frames,
        "usable_frames": result.usable_frames,
        "quarantined_frames": result.quarantined_frames,
        "per_joint": [asdict(s) for s in result.per_joint],
        "bone_segments": [asdict(s) for s in result.bone_segments],
        "symmetry_pairs": [asdict(s) for s in result.symmetry_pairs],
        "anomaly_count": len(result.anomalies),
        "anomalies": [asdict(a) for a in result.anomalies[:100]],  # Cap at 100 for output size.
    })


# ---------------------------------------------------------------------------
# Group-level summaries
# ---------------------------------------------------------------------------


def build_group_summaries(
    canonical_samples: list[dict[str, object]],
    j0_result: JointEdaResult,
    slp_root: Path,
    split_manifest: dict[str, object] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build group-level summaries for setting, cover, subject, and split dimensions.

    Parameters
    ----------
    canonical_samples :
        A05 canonical sample rows.
    j0_result :
        J0 EDA result (usable frame records only).
    split_manifest :
        A06 split manifest dict. If provided, split grouping is included.
    """
    # Reconstruct usable records from j0_result.
    # We need to reconstruct them from the original samples + loaded joints.
    subject_dirs: dict[str, Path] = {}
    for sample in canonical_samples:
        setting = str(sample["setting"])
        subject_id = str(sample["subject_id"])
        key = f"{setting}::{subject_id}"
        if key not in subject_dirs:
            subject_dirs[key] = slp_root / setting / subject_id

    usable_records: list[dict[str, Any]] = []
    for sample in canonical_samples:
        setting = str(sample["setting"])
        subject_id = str(sample["subject_id"])
        cover_condition = str(sample["cover_condition"])
        frame_index = int(sample["frame_index"])
        quarantine = str(sample.get("quarantine", "False")).strip().lower() in ("true", "1", "yes")
        if quarantine:
            continue
        key = f"{setting}::{subject_id}"
        subject_dir = subject_dirs.get(key)
        if subject_dir is None:
            continue
        joints_arr = load_subject_joints_rgb(subject_dir, cover=cover_condition, frame_index=frame_index)
        if joints_arr is None:
            continue
        usable_records.append({
            "sample_id": str(sample["sample_id"]),
            "setting": setting,
            "subject_id": subject_id,
            "cover_condition": cover_condition,
            "frame_index": frame_index,
            "quarantine": False,
            "joints": joints_to_coords(joints_arr),
        })

    # Build a lookup: subject_key → split from A06.
    subject_to_split: dict[str, str] = {}
    if split_manifest:
        for entry in split_manifest.get("subject_entries", []):
            key = f"{entry['setting']}::{entry['subject_id']}"
            subject_to_split[key] = entry["split"]

    summaries: dict[str, dict[str, Any]] = {}

    # 1. By setting (danaLab / simLab).
    for setting in ("danaLab", "simLab"):
        recs = [r for r in usable_records if r["setting"] == setting]
        if recs:
            summaries[f"setting_{setting}"] = _group_stats_to_dict(
                compute_group_stats(recs, f"setting_{setting}", "J0_RGB", J0_IMAGE_BOUNDS)
            )

    # 2. By cover condition.
    for cover in ("uncover", "cover1", "cover2"):
        recs = [r for r in usable_records if r["cover_condition"] == cover]
        if recs:
            summaries[f"cover_{cover}"] = _group_stats_to_dict(
                compute_group_stats(recs, f"cover_{cover}", "J0_RGB", J0_IMAGE_BOUNDS)
            )

    # 3. By setting × cover.
    for setting in ("danaLab", "simLab"):
        for cover in ("uncover", "cover1", "cover2"):
            recs = [r for r in usable_records if r["setting"] == setting and r["cover_condition"] == cover]
            if recs:
                summaries[f"setting_{setting}_cover_{cover}"] = _group_stats_to_dict(
                    compute_group_stats(recs, f"setting_{setting}_cover_{cover}", "J0_RGB", J0_IMAGE_BOUNDS)
                )

    # 4. By subject.
    for rec in usable_records:
        key = f"{rec['setting']}::{rec['subject_id']}"
        subj_recs = [r for r in usable_records if f"{r['setting']}::{r['subject_id']}" == key]
        gk = f"subject_{key}"
        if gk not in summaries:
            summaries[gk] = _group_stats_to_dict(
                compute_group_stats(subj_recs, gk, "J0_RGB", J0_IMAGE_BOUNDS)
            )

    # 5. By train/val/test split.
    if subject_to_split:
        for split_name in ("train", "val", "test"):
            split_subjects = {
                k for k, v in subject_to_split.items() if v == split_name
            }
            recs = [
                r for r in usable_records
                if f"{r['setting']}::{r['subject_id']}" in split_subjects
            ]
            if recs:
                summaries[f"split_{split_name}"] = _group_stats_to_dict(
                    compute_group_stats(recs, f"split_{split_name}", "J0_RGB", J0_IMAGE_BOUNDS)
                )

    return summaries


def _group_stats_to_dict(gs: GroupStats) -> dict[str, Any]:
    """Convert GroupStats to a JSON-serialisable dict."""
    return {
        "group_key": gs.group_key,
        "coordinate_frame": gs.coordinate_frame,
        "sample_count": gs.sample_count,
        "frame_count": gs.frame_count,
        "quarantined_count": gs.quarantined_count,
        "occlusion_rate": gs.occlusion_rate,
        "per_joint_occlusion": {str(j): v for j, v in gs.per_joint_occlusion.items()},
        "out_of_bounds_count": gs.out_of_bounds_count,
        "out_of_bounds_rate": gs.out_of_bounds_rate,
        "bone_length_means": {str(s): v for s, v in gs.bone_length_means.items()},
        "x_mean": gs.x_mean,
        "y_mean": gs.y_mean,
    }


# ---------------------------------------------------------------------------
# CSV QA output
# ---------------------------------------------------------------------------


def write_joint_qa_csv(result: JointEdaResult, output_path: Path) -> None:
    """Write per-joint QA statistics to a CSV file."""
    rows: list[dict[str, object]] = []
    for s in result.per_joint:
        rows.append({
            "coordinate_frame": s.coordinate_frame,
            "joint_index": s.joint_index,
            "joint_name": s.joint_name,
            "total_observations": s.total_observations,
            "visible_count": s.visible_count,
            "occluded_count": s.occluded_count,
            "occlusion_rate": s.occlusion_rate,
            "x_mean": s.x_mean,
            "x_std": s.x_std,
            "x_min": s.x_min,
            "x_max": s.x_max,
            "y_mean": s.y_mean,
            "y_std": s.y_std,
            "y_min": s.y_min,
            "y_max": s.y_max,
            "out_of_bounds_count": s.out_of_bounds_count,
            "out_of_bounds_rate": s.out_of_bounds_rate,
            "invalid_count": s.invalid_count,
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def write_bone_segment_csv(results: list[JointEdaResult], output_path: Path) -> None:
    """Write bone segment statistics to CSV."""
    rows: list[dict[str, object]] = []
    for result in results:
        for s in result.bone_segments:
            rows.append({
                "coordinate_frame": s.coordinate_frame,
                "segment_index": s.segment_index,
                "start_joint": s.start_joint,
                "end_joint": s.end_joint,
                "start_joint_name": s.start_joint_name,
                "end_joint_name": s.end_joint_name,
                "count": s.count,
                "length_mean": s.length_mean,
                "length_std": s.length_std,
                "length_min": s.length_min,
                "length_max": s.length_max,
                "length_median": s.length_median,
                "zero_length_count": s.zero_length_count,
            })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def write_anomaly_csv(anomalies: list[AnomalyCase], output_path: Path) -> None:
    """Write anomaly cases to CSV."""
    rows = [asdict(a) for a in anomalies]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def write_group_stats_csv(
    group_summaries: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Write group-level summaries to CSV."""
    rows: list[dict[str, object]] = []
    for gk, stats in group_summaries.items():
        row: dict[str, object] = {
            "group_key": gk,
            "coordinate_frame": stats.get("coordinate_frame", ""),
            "sample_count": stats.get("sample_count", 0),
            "frame_count": stats.get("frame_count", 0),
            "quarantined_count": stats.get("quarantined_count", 0),
            "occlusion_rate": stats.get("occlusion_rate", 0.0),
            "out_of_bounds_count": stats.get("out_of_bounds_count", 0),
            "out_of_bounds_rate": stats.get("out_of_bounds_rate", 0.0),
            "x_mean": stats.get("x_mean", 0.0),
            "y_mean": stats.get("y_mean", 0.0),
        }
        # Flatten per-joint occlusion.
        pjo = stats.get("per_joint_occlusion", {})
        for j in range(JOINT_COUNT):
            row[f"occ_rate_j{j:02d}"] = pjo.get(str(j), 0.0)
        # Flatten bone lengths.
        bl = stats.get("bone_length_means", {})
        for s in range(len(BONE_SEGMENTS)):
            row[f"bone_len_seg{s:02d}"] = bl.get(str(s), 0.0)
        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------------------------------------------------
# Small visualizations (matplotlib)
# ---------------------------------------------------------------------------

def write_joint_scatter_plot(
    result: JointEdaResult,
    output_path: Path,
    *,
    title: str = "",
) -> None:
    """Write a small joint scatter plot (x vs y, colored by joint index)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = result.all_x
        ys = result.all_y

        fig, ax = plt.subplots(figsize=(6, 8))
        # Color by joint index by re-aggregating.
        # Since all_x/all_y are flat, we can't easily color by joint here.
        # Just plot all points.
        ax.scatter(xs, ys, alpha=0.1, s=2)
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.set_title(title or f"Joint coordinates — {result.coordinate_frame}")
        ax.invert_yaxis()  # Image coordinates have y increasing downward.
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=100)
        plt.close(fig)
    except Exception:
        pass  # Visualizations are best-effort.


def write_occlusion_heatmap(
    per_joint_stats: list[PerJointStats],
    output_path: Path,
    *,
    title: str = "",
) -> None:
    """Write a per-joint occlusion rate heatmap as a bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [s.joint_name for s in per_joint_stats]
        rates = [s.occlusion_rate for s in per_joint_stats]

        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ["#e74c3c" if r > 0.5 else "#3498db" for r in rates]
        ax.bar(names, rates, color=colors)
        ax.set_ylabel("Occlusion Rate")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_title(title or "Per-Joint Occlusion Rate")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=100)
        plt.close(fig)
    except Exception:
        pass  # Visualizations are best-effort.
