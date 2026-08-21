"""Structural inventory and annotation audit for the SLP dataset.

The scanner works at subject/modality/cover-condition grain.  It enumerates
files and reads only the small joint, homography, calibration and physique
artifacts; image frames and raw sensor arrays are never loaded into memory.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat


SETTINGS = ("danaLab", "simLab")
COVER_CONDITIONS = ("uncover", "cover1", "cover2")
SETTING_MODALITIES = {
    "danaLab": ("PM", "IR", "IRraw", "RGB", "depth", "depthRaw"),
    "simLab": ("IR", "IRraw", "RGB", "depth", "depthRaw"),
}
EXPECTED_FRAMES_PER_GROUP = 45
JOINT_COUNT = 14

_FRAME_PATTERN = re.compile(r"^(?:image_)?(?P<index>\d{6})\.(?P<extension>png|npy)$", re.IGNORECASE)

INVENTORY_COLUMNS = (
    "group_id",
    "setting",
    "subject_id",
    "cover_condition",
    "modality",
    "source_relative_directory",
    "file_count",
    "total_bytes",
    "minimum_frame_index",
    "maximum_frame_index",
    "missing_frame_indices",
    "duplicate_frame_indices",
    "unexpected_files",
    "status",
    "error_codes",
    "warning_codes",
)

ANNOTATION_COLUMNS = (
    "subject_key",
    "setting",
    "subject_id",
    "joints_gt_rgb_present",
    "joints_gt_rgb_shape",
    "joints_gt_ir_present",
    "joints_gt_ir_shape",
    "rgb_ir_joint_gt_source",
    "mapped_joint_gt_status",
    "align_ptr_rgb_shape",
    "align_ptr_ir_shape",
    "align_ptr_depth_shape",
    "pmcali_present",
    "pmcali_shape",
    "region_ground_truth_present",
    "status",
    "error_codes",
    "warning_codes",
)


@dataclass(frozen=True, slots=True)
class SlpInventoryRow:
    values: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {column: self.values.get(column, "") for column in INVENTORY_COLUMNS}


@dataclass(frozen=True, slots=True)
class SlpAnnotationRow:
    values: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {column: self.values.get(column, "") for column in ANNOTATION_COLUMNS}


def resolve_slp_root(data_root: Path) -> Path:
    """Accept the SLP root, the extracted SLP2022 root, or a parent directory."""
    root = data_root.expanduser()
    candidates = (root, root / "SLP", root / "SLP2022" / "SLP")
    for candidate in candidates:
        if all((candidate / setting).is_dir() for setting in SETTINGS):
            return candidate
    raise FileNotFoundError(f"SLP danaLab/simLab directories were not found under: {data_root}")


def iter_subject_directories(slp_root: Path) -> Iterator[tuple[str, Path]]:
    for setting in SETTINGS:
        setting_root = slp_root / setting
        for subject_dir in sorted(
            (path for path in setting_root.iterdir() if path.is_dir() and path.name.isdigit()),
            key=lambda path: path.name,
        ):
            yield setting, subject_dir


def _status(errors: list[str], warnings: list[str]) -> str:
    if errors:
        return "ERROR"
    if warnings:
        return "WARN"
    return "OK"


def inventory_modality_group(
    slp_root: Path,
    *,
    setting: str,
    subject_dir: Path,
    modality: str,
    cover_condition: str,
    expected_frames: int = EXPECTED_FRAMES_PER_GROUP,
) -> SlpInventoryRow:
    group_dir = subject_dir / modality / cover_condition
    errors: list[str] = []
    warnings: list[str] = []
    frame_indices: list[int] = []
    unexpected_files: list[str] = []
    total_bytes = 0

    if not group_dir.is_dir():
        errors.append("missing_modality_condition_directory")
    else:
        for source_file in sorted(path for path in group_dir.iterdir() if path.is_file()):
            total_bytes += source_file.stat().st_size
            match = _FRAME_PATTERN.fullmatch(source_file.name)
            if match is None:
                unexpected_files.append(source_file.name)
                continue
            frame_indices.append(int(match.group("index")))

    counts = Counter(frame_indices)
    duplicate_indices = sorted(index for index, count in counts.items() if count > 1)
    expected_indices = set(range(1, expected_frames + 1))
    missing_indices = sorted(expected_indices.difference(counts))
    outside_expected = sorted(index for index in counts if index not in expected_indices)

    if len(frame_indices) != expected_frames:
        errors.append("frame_count_mismatch")
    if missing_indices:
        errors.append("missing_frame_indices")
    if duplicate_indices:
        errors.append("duplicate_frame_indices")
    if outside_expected:
        errors.append("frame_index_out_of_range")
    if unexpected_files:
        warnings.append("unexpected_files")

    relative_directory = group_dir.relative_to(slp_root).as_posix()
    values = {
        "group_id": f"slp::{setting}::{subject_dir.name}::{cover_condition}::{modality}",
        "setting": setting,
        "subject_id": subject_dir.name,
        "cover_condition": cover_condition,
        "modality": modality,
        "source_relative_directory": relative_directory,
        "file_count": len(frame_indices),
        "total_bytes": total_bytes,
        "minimum_frame_index": min(frame_indices) if frame_indices else "",
        "maximum_frame_index": max(frame_indices) if frame_indices else "",
        "missing_frame_indices": ";".join(str(index) for index in missing_indices),
        "duplicate_frame_indices": ";".join(str(index) for index in duplicate_indices),
        "unexpected_files": ";".join(unexpected_files),
        "status": _status(errors, warnings),
        "error_codes": ";".join(sorted(set(errors))),
        "warning_codes": ";".join(sorted(set(warnings))),
    }
    return SlpInventoryRow(values)


def inventory_slp_dataset(
    data_root: Path,
    *,
    expected_frames: int = EXPECTED_FRAMES_PER_GROUP,
) -> Iterator[SlpInventoryRow]:
    slp_root = resolve_slp_root(data_root)
    for setting, subject_dir in iter_subject_directories(slp_root):
        for cover_condition in COVER_CONDITIONS:
            for modality in SETTING_MODALITIES[setting]:
                yield inventory_modality_group(
                    slp_root,
                    setting=setting,
                    subject_dir=subject_dir,
                    modality=modality,
                    cover_condition=cover_condition,
                    expected_frames=expected_frames,
                )


def _shape_text(shape: tuple[int, ...] | None) -> str:
    return "x".join(str(value) for value in shape) if shape is not None else ""


def _load_mat_shape(path: Path, key: str) -> tuple[int, ...] | None:
    if not path.is_file():
        return None
    value = loadmat(path).get(key)
    return tuple(int(item) for item in value.shape) if isinstance(value, np.ndarray) else None


def _load_npy_shape(path: Path) -> tuple[int, ...] | None:
    if not path.is_file():
        return None
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    return tuple(int(item) for item in value.shape)


def audit_subject_annotations(setting: str, subject_dir: Path) -> SlpAnnotationRow:
    """Audit original joint labels and mapping artifacts for one subject.

    RGB/IR joints are marked as the original manually labelled truth described
    by the dataset README.  Joints projected to pressure/depth are explicitly
    classified as derived labels because homography mapping can introduce bias.
    """
    errors: list[str] = []
    warnings: list[str] = []

    rgb_shape = _load_mat_shape(subject_dir / "joints_gt_RGB.mat", "joints_gt")
    ir_shape = _load_mat_shape(subject_dir / "joints_gt_IR.mat", "joints_gt")
    expected_joint_shape = (3, JOINT_COUNT, EXPECTED_FRAMES_PER_GROUP)
    if rgb_shape is None:
        errors.append("missing_joints_gt_rgb")
    elif rgb_shape != expected_joint_shape:
        errors.append("unexpected_joints_gt_rgb_shape")
    if ir_shape is None:
        errors.append("missing_joints_gt_ir")
    elif ir_shape != expected_joint_shape:
        errors.append("unexpected_joints_gt_ir_shape")

    homography_shapes = {
        modality: _load_npy_shape(subject_dir / f"align_PTr_{modality}.npy")
        for modality in ("RGB", "IR", "depth")
    }
    for modality, shape in homography_shapes.items():
        if shape is None:
            errors.append(f"missing_align_ptr_{modality.lower()}")
        elif shape != (3, 3):
            errors.append(f"unexpected_align_ptr_{modality.lower()}_shape")

    pmcali_shape = _load_npy_shape(subject_dir / "PMcali.npy")
    if setting == "danaLab":
        if pmcali_shape is None:
            errors.append("missing_pmcali")
        elif pmcali_shape != (3, EXPECTED_FRAMES_PER_GROUP):
            errors.append("unexpected_pmcali_shape")
    elif pmcali_shape is not None:
        warnings.append("unexpected_pmcali_for_simlab")

    # SLP distributes keypoints, not pixel-level anatomical region masks.
    region_ground_truth_present = False
    warnings.append("region_ground_truth_absent")
    values = {
        "subject_key": f"slp::{setting}::{subject_dir.name}",
        "setting": setting,
        "subject_id": subject_dir.name,
        "joints_gt_rgb_present": rgb_shape is not None,
        "joints_gt_rgb_shape": _shape_text(rgb_shape),
        "joints_gt_ir_present": ir_shape is not None,
        "joints_gt_ir_shape": _shape_text(ir_shape),
        "rgb_ir_joint_gt_source": "manual_original",
        "mapped_joint_gt_status": "derived_homography_bias_possible",
        "align_ptr_rgb_shape": _shape_text(homography_shapes["RGB"]),
        "align_ptr_ir_shape": _shape_text(homography_shapes["IR"]),
        "align_ptr_depth_shape": _shape_text(homography_shapes["depth"]),
        "pmcali_present": pmcali_shape is not None,
        "pmcali_shape": _shape_text(pmcali_shape),
        "region_ground_truth_present": region_ground_truth_present,
        "status": _status(errors, warnings),
        "error_codes": ";".join(sorted(set(errors))),
        "warning_codes": ";".join(sorted(set(warnings))),
    }
    return SlpAnnotationRow(values)


def audit_slp_annotations(data_root: Path) -> Iterator[SlpAnnotationRow]:
    slp_root = resolve_slp_root(data_root)
    for setting, subject_dir in iter_subject_directories(slp_root):
        yield audit_subject_annotations(setting, subject_dir)


def summarise_slp_inventory(
    inventory_rows: Iterable[SlpInventoryRow],
    annotation_rows: Iterable[SlpAnnotationRow],
    *,
    slp_root: Path,
) -> dict[str, object]:
    groups = [row.as_dict() for row in inventory_rows]
    annotations = [row.as_dict() for row in annotation_rows]
    subjects = {(str(row["setting"]), str(row["subject_id"])) for row in annotations}
    group_status = Counter(str(row["status"]) for row in groups)
    annotation_status = Counter(str(row["status"]) for row in annotations)
    group_errors = Counter(
        code for row in groups for code in str(row["error_codes"]).split(";") if code
    )
    annotation_errors = Counter(
        code for row in annotations for code in str(row["error_codes"]).split(";") if code
    )
    annotation_warnings = Counter(
        code for row in annotations for code in str(row["warning_codes"]).split(";") if code
    )
    file_counts_by_modality = Counter()
    for row in groups:
        file_counts_by_modality[str(row["modality"])] += int(row["file_count"])

    return {
        "dataset": "SLP",
        "slp_root": str(slp_root.resolve()),
        "grain": "subject x cover_condition x modality",
        "subjects": len(subjects),
        "subjects_by_setting": dict(sorted(Counter(setting for setting, _ in subjects).items())),
        "inventory_groups": len(groups),
        "group_status_counts": dict(sorted(group_status.items())),
        "group_error_counts": dict(sorted(group_errors.items())),
        "annotation_status_counts": dict(sorted(annotation_status.items())),
        "annotation_error_counts": dict(sorted(annotation_errors.items())),
        "annotation_warning_counts": dict(sorted(annotation_warnings.items())),
        "files_by_modality": dict(sorted(file_counts_by_modality.items())),
        "total_frame_files": sum(file_counts_by_modality.values()),
        "joint_ground_truth": {
            "original_manual_modalities": ["RGB", "IR"],
            "joint_count": JOINT_COUNT,
            "mapped_modalities_are_derived": True,
        },
        "region_ground_truth": {
            "present": False,
            "planned_status": "joint_derived_and_image_assisted_labels_require_manual_review",
        },
    }


def canonical_summary_json(summary: dict[str, object]) -> str:
    return json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
