from __future__ import annotations

import json
from pathlib import Path

from topper_perception.labels.popu import annotation_identity, audit_segmentation_file


def _write_annotation(path: Path) -> None:
    document = {
        "categories": [{"id": category_id, "name": name} for category_id, name in {
            0: "body-parts", 1: "Head", 2: "Lower_Arm", 3: "Lower_Leg",
            4: "Torso", 5: "Upper_Arm", 6: "Upper_Leg",
        }.items()],
        "images": [{"id": 1, "width": 27, "height": 64}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 4, "segmentation": [[1, 1, 2, 1, 2, 2]]}],
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def test_annotation_identity_reads_subject_posture_and_variation(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "7"
    annotation_dir.mkdir()
    annotation = annotation_dir / "7_left3_annotations.coco.json"
    annotation.write_text("{}", encoding="utf-8")
    assert annotation_identity(annotation) == {"subject": "7", "posture": "left", "variation": "3"}


def test_audit_marks_multiple_matching_tactilus_records_as_ambiguous(tmp_path: Path) -> None:
    annotation_dir = tmp_path / "segmentation_data" / "7"
    annotation_dir.mkdir(parents=True)
    annotation = annotation_dir / "7_left3_annotations.coco.json"
    _write_annotation(annotation)
    tactilus_dir = tmp_path / "tactilus_data" / "7"
    tactilus_dir.mkdir(parents=True)
    (tactilus_dir / "left3_0.json").write_text("{}", encoding="utf-8")
    (tactilus_dir / "left3_1.json").write_text("{}", encoding="utf-8")

    result = audit_segmentation_file(annotation, tmp_path / "tactilus_data")

    assert result["candidate_tactilus_record_count"] == 2
    assert result["alignment_status"] == "AMBIGUOUS_TACTILUS_CANDIDATES"
