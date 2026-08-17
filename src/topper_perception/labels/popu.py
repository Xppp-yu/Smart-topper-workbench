"""Read-only structural audit helpers for PoPu COCO body-part annotations.

This module deliberately audits *whether* an annotation can supervise a
pressure record.  It does not rasterize polygons into a training target: a
COCO file may correspond to several Tactilus JSON captures, so selecting one
without a documented pairing rule would create invented ground truth.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


BODY_PART_CATEGORIES = {
    1: "Head",
    2: "Lower_Arm",
    3: "Lower_Leg",
    4: "Torso",
    5: "Upper_Arm",
    6: "Upper_Leg",
}

ANNOTATION_NAME = re.compile(
    r"^(?P<subject>[^_]+)_(?P<posture>empty|left|prone|right|supine)(?P<variation>\d+)_annotations\.coco\.json$",
    re.IGNORECASE,
)


def annotation_identity(annotation_path: Path) -> dict[str, str]:
    """Extract the declared subject/posture/variation identity from one file."""
    match = ANNOTATION_NAME.match(annotation_path.name)
    if not match:
        raise ValueError(f"Unexpected PoPu segmentation filename: {annotation_path.name}")
    identity = {key: value.lower() for key, value in match.groupdict().items()}
    if annotation_path.parent.name != identity["subject"]:
        raise ValueError(
            "Segmentation filename subject differs from parent directory: "
            f"{annotation_path}"
        )
    return identity


def _candidate_tactilus_records(
    tactilus_root: Path,
    *,
    subject: str,
    posture: str,
    variation: str,
) -> list[Path]:
    """Find JSON records sharing the annotation's visible identity.

    The source names contain a capture suffix (for example ``left1_0.json``),
    while COCO annotations contain only ``left1``.  Returning every candidate
    makes the unresolved many-to-one relationship measurable rather than
    silently picking a capture.
    """
    subject_root = tactilus_root / subject
    if posture == "empty":
        return sorted(subject_root.glob(f"{posture}{variation}*.json"))
    return sorted(subject_root.glob(f"{posture}{variation}_*.json"))


def _polygon_out_of_bounds_points(
    annotations: list[dict[str, Any]], *, width: int, height: int
) -> int:
    count = 0
    for annotation in annotations:
        for polygon in annotation.get("segmentation", []):
            if not isinstance(polygon, list) or len(polygon) % 2:
                count += 1
                continue
            for x_value, y_value in zip(polygon[::2], polygon[1::2], strict=True):
                try:
                    x, y = float(x_value), float(y_value)
                except (TypeError, ValueError):
                    count += 1
                    continue
                if x < 0 or x > width or y < 0 or y > height:
                    count += 1
    return count


def _declared_category_ids(categories: list[Any]) -> set[int]:
    ids: set[int] = set()
    for category in categories:
        if not isinstance(category, dict):
            continue
        category_id = category.get("id")
        try:
            ids.add(int(category_id))
        except (TypeError, ValueError):
            continue
    return ids


def _declared_image_ids(images: list[Any]) -> list[int]:
    ids: list[int] = []
    for image in images:
        if isinstance(image, dict) and "id" in image:
            try:
                ids.append(int(image["id"]))
            except (TypeError, ValueError):
                continue
    return ids


def _annotation_category_ids(annotations: list[Any]) -> set[int]:
    ids: set[int] = set()
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        category_id = annotation.get("category_id")
        try:
            ids.add(int(category_id))
        except (TypeError, ValueError):
            continue
    return ids


def _category_name_mismatch_count(categories: list[Any]) -> int:
    """Count declared categories whose name disagrees with the id contract."""
    count = 0
    for category in categories:
        if not isinstance(category, dict):
            continue
        category_id = category.get("id")
        name = category.get("name")
        if category_id is None or name is None:
            continue
        try:
            category_id_int = int(category_id)
        except (TypeError, ValueError):
            continue
        expected = BODY_PART_CATEGORIES.get(category_id_int)
        if expected is not None and str(name) != expected:
            count += 1
    return count


def _annotation_image_reference_error_count(
    annotations: list[Any], image_ids: set[int]
) -> int:
    """Count annotations whose ``image_id`` does not reference a declared image."""
    count = 0
    for annotation in annotations:
        if not isinstance(annotation, dict):
            count += 1
            continue
        image_id = annotation.get("image_id")
        try:
            resolved = int(image_id)
        except (TypeError, ValueError):
            count += 1
            continue
        if resolved not in image_ids:
            count += 1
    return count


def _annotation_bbox_error_count(annotations: list[Any]) -> int:
    """Count annotations with a missing, malformed or non-positive bbox."""
    count = 0
    for annotation in annotations:
        if not isinstance(annotation, dict):
            count += 1
            continue
        bbox = annotation.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            count += 1
            continue
        try:
            _, _, width, height = (float(value) for value in bbox)
        except (TypeError, ValueError):
            count += 1
            continue
        if width <= 0 or height <= 0:
            count += 1
    return count


def _annotation_category_error_count(annotations: list[Any]) -> int:
    """Count annotations whose category_id is missing or outside the contract."""
    count = 0
    for annotation in annotations:
        if not isinstance(annotation, dict):
            count += 1
            continue
        category_id = annotation.get("category_id")
        try:
            resolved = int(category_id)
        except (TypeError, ValueError):
            count += 1
            continue
        if resolved not in BODY_PART_CATEGORIES:
            count += 1
    return count


def audit_segmentation_file(annotation_path: Path, tactilus_root: Path) -> dict[str, object]:
    """Audit a single PoPu COCO file without asserting it is frame-level truth."""
    identity = annotation_identity(annotation_path)
    document = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = document.get("images", [])
    annotations = document.get("annotations", [])
    categories = document.get("categories", [])

    image = images[0] if len(images) == 1 and isinstance(images[0], dict) else {}
    width = int(image.get("width", 0)) if image else 0
    height = int(image.get("height", 0)) if image else 0
    image_id = image.get("id", "") if image else ""
    image_file_name = image.get("file_name", "") if image else ""

    category_ids = _declared_category_ids(categories)
    annotation_category_ids = _annotation_category_ids(annotations)
    candidates = _candidate_tactilus_records(tactilus_root, **identity)

    category_name_mismatch_count = _category_name_mismatch_count(categories)
    declared_image_ids = _declared_image_ids(images)
    annotation_image_reference_error_count = _annotation_image_reference_error_count(
        annotations, set(declared_image_ids)
    )
    annotation_bbox_error_count = _annotation_bbox_error_count(annotations)
    annotation_category_error_count = _annotation_category_error_count(annotations)
    invalid_polygon_points = _polygon_out_of_bounds_points(
        [item for item in annotations if isinstance(item, dict)], width=width, height=height
    ) if width > 0 and height > 0 else 0

    structural_errors: list[str] = []
    if len(images) != 1:
        structural_errors.append("expected_exactly_one_image")
    if (height, width) != (64, 27):
        structural_errors.append(f"unexpected_canvas:{height}x{width}")
    if not set(BODY_PART_CATEGORIES).issubset(category_ids):
        structural_errors.append("missing_declared_body_part_category")
    if category_name_mismatch_count:
        structural_errors.append("category_name_mismatch")
    if annotation_image_reference_error_count:
        structural_errors.append("annotation_image_reference_unknown")
    if annotation_bbox_error_count:
        structural_errors.append("annotation_bbox_invalid")
    if annotation_category_error_count:
        structural_errors.append("unknown_annotation_category")
    if invalid_polygon_points:
        structural_errors.append("polygon_coordinate_out_of_bounds")

    if structural_errors:
        alignment_status = "STRUCTURAL_WARN"
    elif len(candidates) == 1:
        alignment_status = "ONE_TO_ONE_CANDIDATE"
    elif len(candidates) == 0:
        alignment_status = "MISSING_TACTILUS_CANDIDATE"
    else:
        alignment_status = "AMBIGUOUS_TACTILUS_CANDIDATES"

    return {
        "annotation_file": str(annotation_path.resolve()),
        "subject_id": identity["subject"],
        "posture": identity["posture"],
        "variation": identity["variation"],
        "image_count": len(images),
        "image_id": str(image_id),
        "image_file_name": str(image_file_name),
        "canvas_height": height,
        "canvas_width": width,
        "annotation_count": len(annotations),
        "annotation_category_ids": ";".join(map(str, sorted(annotation_category_ids))),
        "category_name_mismatch_count": category_name_mismatch_count,
        "annotation_image_reference_error_count": annotation_image_reference_error_count,
        "annotation_bbox_error_count": annotation_bbox_error_count,
        "annotation_category_error_count": annotation_category_error_count,
        "candidate_tactilus_record_count": len(candidates),
        "candidate_tactilus_records": ";".join(str(path.resolve()) for path in candidates),
        "invalid_polygon_point_count": invalid_polygon_points,
        "structural_errors": ";".join(structural_errors),
        "alignment_status": alignment_status,
        "supervision_boundary": (
            "COCO region annotation; not frame-level ground truth until a documented "
            "Tactilus-record and snapshot pairing rule is verified. ONE_TO_ONE_CANDIDATE "
            "is a candidate pairing only, never an asserted frame-level truth."
        ),
    }
