from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = (
    PROJECT_ROOT / "configs" / "annotations" / "slp_region_annotation_v0.1.schema.json"
)


def _schema() -> dict[str, object]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_slp_region_schema_is_valid_json_and_closed() -> None:
    schema = _schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"]["const"] == "slp_region_annotation_v0.1"


def test_slp_region_schema_requires_provenance_and_review_fields() -> None:
    schema = _schema()
    required = set(schema["required"])

    assert {
        "label_tier",
        "label_source",
        "joint_gt_source",
        "review_status",
        "proposal_polygon",
        "final_polygon",
        "parameter_hash",
        "homography_sha256",
        "reviewer_id",
        "reviewed_at",
        "provenance",
    }.issubset(required)


def test_slp_region_schema_separates_pseudo_and_reviewed_tiers() -> None:
    schema = _schema()
    tiers = schema["properties"]["label_tier"]["enum"]
    sources = schema["properties"]["label_source"]["enum"]
    reviewed_rule = schema["allOf"][0]

    assert tiers == ["R0", "R1", "R2", "R3"]
    assert "opencv_refined" in sources
    assert "human_consensus" in sources
    assert reviewed_rule["if"]["properties"]["label_tier"]["enum"] == ["R2", "R3"]
    assert reviewed_rule["then"]["properties"]["reviewer_id"]["minLength"] == 1


def test_slp_region_schema_uses_coarse_product_relevant_regions() -> None:
    schema = _schema()
    regions = set(schema["properties"]["region_id"]["enum"])

    assert {"shoulder_left", "shoulder_right", "thorax_back"}.issubset(regions)
    assert {"abdomen_waist", "pelvis_hip"}.issubset(regions)
    assert "buttock_exact" not in regions
