from __future__ import annotations

import json
from pathlib import Path

from topper_perception.labels.popu import (
    BODY_PART_CATEGORIES,
    annotation_identity,
    audit_segmentation_file,
)


def _valid_document() -> dict:
    return {
        "categories": [
            {"id": category_id, "name": name}
            for category_id, name in BODY_PART_CATEGORIES.items()
        ],
        "images": [{"id": 1, "file_name": "7_left3.png", "width": 27, "height": 64}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 4,
                "bbox": [1, 1, 3, 4],
                "segmentation": [[1, 1, 2, 1, 2, 2]],
            }
        ],
    }


def _write_document(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


def _annotation_dir(tmp_path: Path, subject: str = "7") -> Path:
    annotation_dir = tmp_path / "segmentation_data" / subject
    annotation_dir.mkdir(parents=True, exist_ok=True)
    return annotation_dir


def _tactilus_dir(tmp_path: Path, subject: str = "7") -> Path:
    tactilus_dir = tmp_path / "tactilus_data" / subject
    tactilus_dir.mkdir(parents=True, exist_ok=True)
    return tactilus_dir


def test_annotation_identity_reads_subject_posture_and_variation(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "7"
    annotation_dir.mkdir()
    annotation = annotation_dir / "7_left3_annotations.coco.json"
    annotation.write_text("{}", encoding="utf-8")
    assert annotation_identity(annotation) == {"subject": "7", "posture": "left", "variation": "3"}


def test_audit_marks_single_matching_record_as_one_to_one_candidate(tmp_path: Path) -> None:
    annotation = _annotation_dir(tmp_path) / "7_left3_annotations.coco.json"
    _write_document(annotation, _valid_document())
    (_tactilus_dir(tmp_path) / "left3_0.json").write_text("{}", encoding="utf-8")

    result = audit_segmentation_file(annotation, tmp_path / "tactilus_data")

    assert result["candidate_tactilus_record_count"] == 1
    assert result["alignment_status"] == "ONE_TO_ONE_CANDIDATE"
    assert "candidate" in result["supervision_boundary"].lower()


def test_audit_marks_zero_candidates_as_missing_tactilus(tmp_path: Path) -> None:
    annotation = _annotation_dir(tmp_path) / "7_left3_annotations.coco.json"
    _write_document(annotation, _valid_document())
    _tactilus_dir(tmp_path)  # directory exists but has no matching capture

    result = audit_segmentation_file(annotation, tmp_path / "tactilus_data")

    assert result["candidate_tactilus_record_count"] == 0
    assert result["alignment_status"] == "MISSING_TACTILUS_CANDIDATE"


def test_audit_marks_multiple_matching_tactilus_records_as_ambiguous(tmp_path: Path) -> None:
    annotation = _annotation_dir(tmp_path) / "7_left3_annotations.coco.json"
    _write_document(annotation, _valid_document())
    tactilus_dir = _tactilus_dir(tmp_path)
    (tactilus_dir / "left3_0.json").write_text("{}", encoding="utf-8")
    (tactilus_dir / "left3_1.json").write_text("{}", encoding="utf-8")

    result = audit_segmentation_file(annotation, tmp_path / "tactilus_data")

    assert result["candidate_tactilus_record_count"] == 2
    assert result["alignment_status"] == "AMBIGUOUS_TACTILUS_CANDIDATES"


def test_audit_flags_category_name_mismatch(tmp_path: Path) -> None:
    document = _valid_document()
    for category in document["categories"]:
        if category["id"] == 4:
            category["name"] = "Wrong_Name"

    annotation = _annotation_dir(tmp_path) / "7_left3_annotations.coco.json"
    _write_document(annotation, document)
    _tactilus_dir(tmp_path)

    result = audit_segmentation_file(annotation, tmp_path / "tactilus_data")

    assert result["category_name_mismatch_count"] == 1
    assert "category_name_mismatch" in result["structural_errors"]
    assert result["alignment_status"] == "STRUCTURAL_WARN"


def test_audit_flags_annotation_image_reference_error(tmp_path: Path) -> None:
    document = _valid_document()
    document["annotations"][0]["image_id"] = 999  # references an undeclared image

    annotation = _annotation_dir(tmp_path) / "7_left3_annotations.coco.json"
    _write_document(annotation, document)
    _tactilus_dir(tmp_path)

    result = audit_segmentation_file(annotation, tmp_path / "tactilus_data")

    assert result["annotation_image_reference_error_count"] == 1
    assert "annotation_image_reference_unknown" in result["structural_errors"]
    assert result["alignment_status"] == "STRUCTURAL_WARN"


def test_audit_flags_invalid_bbox(tmp_path: Path) -> None:
    document = _valid_document()
    document["annotations"][0]["bbox"] = [1, 1, 0, 0]  # non-positive size

    annotation = _annotation_dir(tmp_path) / "7_left3_annotations.coco.json"
    _write_document(annotation, document)
    _tactilus_dir(tmp_path)

    result = audit_segmentation_file(annotation, tmp_path / "tactilus_data")

    assert result["annotation_bbox_error_count"] == 1
    assert "annotation_bbox_invalid" in result["structural_errors"]
    assert result["alignment_status"] == "STRUCTURAL_WARN"


def test_audit_reports_image_id_file_name_and_canvas(tmp_path: Path) -> None:
    annotation = _annotation_dir(tmp_path) / "7_left3_annotations.coco.json"
    _write_document(annotation, _valid_document())
    _tactilus_dir(tmp_path)

    result = audit_segmentation_file(annotation, tmp_path / "tactilus_data")

    assert result["image_id"] == "1"
    assert result["image_file_name"] == "7_left3.png"
    assert result["canvas_width"] == 27
    assert result["canvas_height"] == 64
    assert result["category_name_mismatch_count"] == 0
    assert result["annotation_image_reference_error_count"] == 0
    assert result["annotation_bbox_error_count"] == 0
    assert result["annotation_category_error_count"] == 0
