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
    category_ids = {int(item["id"]) for item in categories if isinstance(item, dict) and "id" in item}
    annotation_category_ids = {
        int(item["category_id"])
        for item in annotations
        if isinstance(item, dict) and "category_id" in item
    }
    unknown_category_ids = sorted(annotation_category_ids - set(BODY_PART_CATEGORIES))
    candidates = _candidate_tactilus_records(tactilus_root, **identity)
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
    if unknown_category_ids:
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
        "canvas_height": height,
        "canvas_width": width,
        "annotation_count": len(annotations),
        "annotation_category_ids": ";".join(map(str, sorted(annotation_category_ids))),
        "candidate_tactilus_record_count": len(candidates),
        "candidate_tactilus_records": ";".join(str(path.resolve()) for path in candidates),
        "invalid_polygon_point_count": invalid_polygon_points,
        "structural_errors": ";".join(structural_errors),
        "alignment_status": alignment_status,
        "supervision_boundary": (
            "COCO region annotation; not frame-level ground truth until a documented "
            "Tactilus-record and snapshot pairing rule is verified."
        ),
    }
