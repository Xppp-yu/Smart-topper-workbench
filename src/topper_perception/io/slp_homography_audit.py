"""Dataset-side helpers for the SLP homography audit.

This module reads only small alignment/joint artifacts and PNG headers. It does
not mutate raw SLP data and it deliberately keeps homography direction
unresolved until documentation/overlay review confirms the semantic contract.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np
from scipy.io import loadmat

from topper_perception.geometry.slp_homography import (
    apply_homography,
    homography_diagnostics,
    in_bounds_mask,
    invert_homography,
    roundtrip_errors,
)
from topper_perception.io.slp_inventory import iter_subject_directories, resolve_slp_root


ALIGN_MODALITIES = ("RGB", "IR", "depth")
JOINT_MODALITIES = ("RGB", "IR")
HOMOGRAPHY_AUDIT_COLUMNS = (
    "setting",
    "subject_id",
    "modality",
    "matrix_uri",
    "matrix_present",
    "determinant",
    "condition_number",
    "rank",
    "invertible",
    "source_width",
    "source_height",
    "pm_width",
    "pm_height",
    "probe_roundtrip_mean_error",
    "probe_roundtrip_max_error",
    "joint_points",
    "direct_joint_in_bounds_rate",
    "inverse_joint_in_bounds_rate",
    "direction_status",
    "coordinate_origin_status",
    "error_codes",
)


@dataclass(frozen=True, slots=True)
class SlpHomographyAuditRow:
    values: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {column: self.values.get(column, "") for column in HOMOGRAPHY_AUDIT_COLUMNS}


def read_png_dimensions(path: Path) -> tuple[int, int]:
    """Read PNG width/height from IHDR without decoding the image payload."""
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"not a valid PNG header: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid PNG dimensions: {path}")
    return int(width), int(height)


def load_joint_xy(subject_dir: Path, modality: str) -> np.ndarray:
    """Load J0 x/y coordinates as frames x joints x 2 without origin shifting."""
    if modality not in JOINT_MODALITIES:
        raise ValueError(f"no original J0 joints for modality: {modality}")
    path = subject_dir / f"joints_gt_{modality}.mat"
    value = loadmat(path).get("joints_gt")
    if not isinstance(value, np.ndarray) or value.shape != (3, 14, 45):
        raise ValueError(f"unexpected joints_gt shape for {path}: {getattr(value, 'shape', None)}")
    xy = np.asarray(value[:2, :, :], dtype=np.float64).transpose(2, 1, 0)
    return xy


def source_png_path(subject_dir: Path, modality: str, *, frame_index: int = 1) -> Path:
    return subject_dir / modality / "uncover" / f"image_{frame_index:06d}.png"


def pm_png_path(subject_dir: Path, *, frame_index: int = 1) -> Path:
    return subject_dir / "PM" / "uncover" / f"image_{frame_index:06d}.png"


def _probe_points(width: int, height: int) -> np.ndarray:
    xmax = max(width - 1, 0)
    ymax = max(height - 1, 0)
    return np.asarray(
        [
            [0.0, 0.0],
            [float(xmax), 0.0],
            [0.0, float(ymax)],
            [float(xmax), float(ymax)],
            [float(xmax) / 2.0, float(ymax) / 2.0],
        ],
        dtype=np.float64,
    )


def audit_subject_homographies(
    slp_root: Path,
    *,
    setting: str,
    subject_dir: Path,
) -> Iterator[SlpHomographyAuditRow]:
    """Audit the three published alignment matrices for one SLP subject."""
    pm_dimensions: tuple[int, int] | None = None
    pm_path = pm_png_path(subject_dir)
    if pm_path.is_file():
        pm_dimensions = read_png_dimensions(pm_path)

    for modality in ALIGN_MODALITIES:
        errors: list[str] = []
        matrix_path = subject_dir / f"align_PTr_{modality}.npy"
        source_path = source_png_path(subject_dir, modality)
        values: dict[str, object] = {
            "setting": setting,
            "subject_id": subject_dir.name,
            "modality": modality,
            "matrix_uri": matrix_path.relative_to(slp_root).as_posix(),
            "matrix_present": matrix_path.is_file(),
            "coordinate_origin_status": "UNRESOLVED_RAW_DATASET_COORDINATES_NO_OFFSET_APPLIED",
        }

        if not matrix_path.is_file():
            errors.append("missing_homography")
            values.update(
                {
                    "direction_status": "BLOCKED_MISSING_MATRIX",
                    "error_codes": ";".join(errors),
                }
            )
            yield SlpHomographyAuditRow(values)
            continue

        matrix = np.load(matrix_path, allow_pickle=False)
        try:
            diagnostics = homography_diagnostics(matrix)
        except ValueError:
            errors.append("invalid_homography")
            values.update(
                {
                    "direction_status": "BLOCKED_INVALID_MATRIX",
                    "error_codes": ";".join(errors),
                }
            )
            yield SlpHomographyAuditRow(values)
            continue

        values.update(diagnostics.as_dict())
        if not diagnostics.invertible:
            errors.append("non_invertible_homography")
            values.update(
                {
                    "direction_status": "BLOCKED_NON_INVERTIBLE",
                    "error_codes": ";".join(errors),
                }
            )
            yield SlpHomographyAuditRow(values)
            continue

        if not source_path.is_file():
            errors.append("missing_source_reference_png")
            source_dimensions = None
        else:
            source_dimensions = read_png_dimensions(source_path)
            values["source_width"], values["source_height"] = source_dimensions

        if pm_dimensions is not None:
            values["pm_width"], values["pm_height"] = pm_dimensions

        if source_dimensions is not None:
            probe = _probe_points(*source_dimensions)
            probe_errors = roundtrip_errors(probe, matrix)
            values["probe_roundtrip_mean_error"] = float(probe_errors.mean())
            values["probe_roundtrip_max_error"] = float(probe_errors.max())

        if modality in JOINT_MODALITIES and pm_dimensions is not None:
            joints = load_joint_xy(subject_dir, modality).reshape(-1, 2)
            finite = np.isfinite(joints).all(axis=1)
            joints = joints[finite]
            values["joint_points"] = int(joints.shape[0])
            direct = apply_homography(joints, matrix)
            inverse = apply_homography(joints, invert_homography(matrix))
            values["direct_joint_in_bounds_rate"] = float(
                in_bounds_mask(direct, width=pm_dimensions[0], height=pm_dimensions[1]).mean()
            )
            values["inverse_joint_in_bounds_rate"] = float(
                in_bounds_mask(inverse, width=pm_dimensions[0], height=pm_dimensions[1]).mean()
            )
            values["direction_status"] = "UNRESOLVED_REQUIRES_DOCUMENT_AND_OVERLAY_REVIEW"
        elif modality == "depth":
            values["direction_status"] = "UNRESOLVED_NO_ORIGINAL_DEPTH_J0"
        elif pm_dimensions is None:
            values["direction_status"] = "UNRESOLVED_NO_PM_REFERENCE_IMAGE"
        else:
            values["direction_status"] = "UNRESOLVED_REQUIRES_DOCUMENT_AND_OVERLAY_REVIEW"

        values["error_codes"] = ";".join(errors)
        yield SlpHomographyAuditRow(values)


def audit_slp_homographies(data_root: Path) -> Iterator[SlpHomographyAuditRow]:
    slp_root = resolve_slp_root(data_root)
    for setting, subject_dir in iter_subject_directories(slp_root):
        yield from audit_subject_homographies(
            slp_root,
            setting=setting,
            subject_dir=subject_dir,
        )


def summarise_homography_audit(rows: list[SlpHomographyAuditRow]) -> dict[str, object]:
    materialized = [row.as_dict() for row in rows]
    status_counts = Counter(str(row["direction_status"]) for row in materialized)
    error_counts = Counter(
        code
        for row in materialized
        for code in str(row["error_codes"]).split(";")
        if code
    )
    invertible_count = sum(bool(row["invertible"]) for row in materialized)
    roundtrip_maxima = [
        float(row["probe_roundtrip_max_error"])
        for row in materialized
        if row["probe_roundtrip_max_error"] not in ("", None)
    ]
    return {
        "rows": len(materialized),
        "expected_rows": 109 * len(ALIGN_MODALITIES),
        "invertible_matrices": invertible_count,
        "direction_status_counts": dict(sorted(status_counts.items())),
        "error_counts": dict(sorted(error_counts.items())),
        "max_probe_roundtrip_error": max(roundtrip_maxima) if roundtrip_maxima else None,
        "semantic_direction_auto_selected": False,
        "coordinate_origin_auto_shifted": False,
    }


def select_fixed_danalab_subjects(slp_root: Path, *, count: int = 6) -> list[Path]:
    """Select deterministic, spread-out danaLab subjects for manual overlays."""
    if count <= 0:
        return []
    subjects = [
        subject_dir
        for setting, subject_dir in iter_subject_directories(slp_root)
        if setting == "danaLab"
    ]
    if not subjects:
        return []
    if count >= len(subjects):
        return subjects
    indices = np.linspace(0, len(subjects) - 1, num=count, dtype=int)
    return [subjects[int(index)] for index in indices]
