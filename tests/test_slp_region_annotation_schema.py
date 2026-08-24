from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
from jsonschema.exceptions import ValidationError
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = (
    PROJECT_ROOT / "configs" / "annotations" / "slp_region_annotation_v0.1.schema.json"
)


def _schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validator():
    schema = _schema()
    # Use default format checker (Draft202012Validator validates date-time format)
    return jsonschema.Draft202012Validator(schema)


def _validate(data: dict[str, Any]) -> list[ValidationError]:
    validator = _validator()
    errors = list(validator.iter_errors(data))
    return errors


def _assert_valid(data: dict[str, Any]) -> None:
    """Assert a data dict passes Draft 2020-12 validation with no errors."""
    errors = _validate(data)
    assert not errors, "\n".join(e.message for e in errors)


def _assert_invalid(data: dict[str, Any]) -> None:
    """Assert a data dict fails Draft 2020-12 validation."""
    errors = _validate(data)
    assert errors, f"Expected validation failure, but data was valid: {data}"


def _minimal_provenance() -> dict[str, Any]:
    return {
        "source_artifacts": ["slp::danaLab::00001::uncover::1::rgb"],
        "generator": "test-fixture",
        "created_at": "2026-08-01T12:00:00Z",
    }


def _sha256() -> str:
    return "A" * 64


def _poly() -> list[list[float]]:
    return [[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]]


def _base(tier: str, source: str, status: str) -> dict[str, Any]:
    return {
        "schema_version": "slp_region_annotation_v0.1",
        "annotation_id": "test-001",
        "sample_id": "slp::danaLab::00001::uncover::1",
        "setting": "danaLab",
        "subject_id": "00001",
        "cover_condition": "uncover",
        "frame_index": 1,
        "coordinate_frame": "rgb",
        "region_id": "head_neck",
        "label_tier": tier,
        "label_source": source,
        "joint_gt_source": "manual_rgb",
        "review_status": status,
        "proposal_polygon": _poly(),
        "algorithm_version": "test-v0.1",
        "parameter_hash": _sha256(),
        "homography_sha256": _sha256(),
        "alignment_confidence": 0.9,
        "anatomical_confidence": 0.8,
        "quality_flags": [],
        "reviewer_id": None,
        "reviewed_at": None,
        "reason_codes": [],
        "provenance": _minimal_provenance(),
    }


# ---------------------------------------------------------------------------
# Schema structural tests (existing, preserved)
# ---------------------------------------------------------------------------

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
        "parameter_hash",
        "homography_sha256",
        "reviewer_id",
        "reviewed_at",
        "provenance",
    }.issubset(required)
    # final_polygon is no longer in top-level required (R0 may be null)
    assert "final_polygon" not in required


def test_slp_region_schema_separates_pseudo_and_reviewed_tiers() -> None:
    schema = _schema()
    tiers = schema["properties"]["label_tier"]["enum"]
    sources = schema["properties"]["label_source"]["enum"]
    # R2/R3 conditional rule present in allOf
    allof_rules = schema.get("allOf", [])
    r2r3_rule = next(
        (r for r in allof_rules if "R2" in str(r.get("if", {}).get("properties", {}).get("label_tier", {}))),
        None,
    )
    assert tiers == ["R0", "R1", "R2", "R3"]
    assert "opencv_refined" in sources
    assert "human_consensus" in sources
    assert r2r3_rule is not None


def test_slp_region_schema_uses_coarse_product_relevant_regions() -> None:
    schema = _schema()
    regions = set(schema["properties"]["region_id"]["enum"])
    assert {"shoulder_left", "shoulder_right", "thorax_back"}.issubset(regions)
    assert {"abdomen_waist", "pelvis_hip"}.issubset(regions)
    assert "buttock_exact" not in regions


# ---------------------------------------------------------------------------
# Draft 2020-12 metaschema compliance
# ---------------------------------------------------------------------------

def test_schema_passes_draft_2020_12_metaschema() -> None:
    """The schema itself must validate against the Draft 2020-12 metaschema."""
    schema = _schema()
    metaschema = jsonschema.Draft202012Validator.META_SCHEMA
    validator = jsonschema.Draft202012Validator(metaschema)
    errors = list(validator.iter_errors(schema))
    assert not errors, f"Schema fails metaschema: {errors}"


# ---------------------------------------------------------------------------
# Tier/source/status compatibility — positive cases
# ---------------------------------------------------------------------------

def test_r0_valid_joint_geometry_pending() -> None:
    """R0 + joint_geometry + pending with null final_polygon is valid."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    _assert_valid(data)


def test_r0_valid_joint_geometry_uncertain() -> None:
    """R0 + joint_geometry + uncertain is valid."""
    data = _base("R0", "joint_geometry", "uncertain")
    data["final_polygon"] = None
    _assert_valid(data)


def test_r0_valid_joint_geometry_rejected() -> None:
    """R0 + joint_geometry + rejected with reviewer is valid."""
    data = _base("R0", "joint_geometry", "rejected")
    data["final_polygon"] = None
    data["reviewer_id"] = "auto-rejector-001"
    data["reviewed_at"] = "2026-08-01T12:00:00Z"
    _assert_valid(data)


def test_r1_valid_opencv_refined_pending() -> None:
    """R1 + opencv_refined + pending with final_polygon is valid."""
    data = _base("R1", "opencv_refined", "pending")
    data["final_polygon"] = _poly()
    _assert_valid(data)


def test_r1_valid_opencv_refined_uncertain() -> None:
    """R1 + opencv_refined + uncertain is valid."""
    data = _base("R1", "opencv_refined", "uncertain")
    data["final_polygon"] = _poly()
    _assert_valid(data)


def test_r1_valid_opencv_refined_rejected() -> None:
    """R1 + opencv_refined + rejected with reviewer is valid."""
    data = _base("R1", "opencv_refined", "rejected")
    data["final_polygon"] = _poly()
    data["reviewer_id"] = "auto-rejector-002"
    data["reviewed_at"] = "2026-08-01T12:00:00Z"
    _assert_valid(data)


def test_r2_valid_human_accepted_accepted() -> None:
    """R2 + human_accepted + accepted with reviewer is valid."""
    data = _base("R2", "human_accepted", "accepted")
    data["final_polygon"] = _poly()
    data["reviewer_id"] = "reviewer-001"
    data["reviewed_at"] = "2026-08-01T12:00:00Z"
    _assert_valid(data)


def test_r2_valid_human_edited_edited() -> None:
    """R2 + human_edited + edited with reviewer is valid."""
    data = _base("R2", "human_edited", "edited")
    data["final_polygon"] = _poly()
    data["reviewer_id"] = "reviewer-002"
    data["reviewed_at"] = "2026-08-02T10:00:00Z"
    _assert_valid(data)


def test_r3_valid_human_consensus_accepted() -> None:
    """R3 + human_consensus + accepted with reviewer is valid."""
    data = _base("R3", "human_consensus", "accepted")
    data["final_polygon"] = _poly()
    data["reviewer_id"] = "reviewer-003"
    data["reviewed_at"] = "2026-08-03T09:00:00Z"
    _assert_valid(data)


def test_r3_valid_human_consensus_adjudicated() -> None:
    """R3 + human_consensus + adjudicated is valid."""
    data = _base("R3", "human_consensus", "adjudicated")
    data["final_polygon"] = _poly()
    data["reviewer_id"] = "adjudicator-001"
    data["reviewed_at"] = "2026-08-03T11:00:00Z"
    _assert_valid(data)


# ---------------------------------------------------------------------------
# Tier/source/status compatibility — negative cases
# ---------------------------------------------------------------------------

def test_reject_unknown_region() -> None:
    """Unknown region_id must be rejected."""
    data = _base("R0", "joint_geometry", "pending")
    data["region_id"] = "buttock_exact"
    data["final_polygon"] = None
    _assert_invalid(data)


def test_reject_unknown_tier() -> None:
    """Unknown label_tier must be rejected."""
    data = _base("R0", "joint_geometry", "pending")
    data["label_tier"] = "R4"
    data["final_polygon"] = None
    _assert_invalid(data)


def test_reject_unknown_source() -> None:
    """Unknown label_source must be rejected."""
    data = _base("R0", "joint_geometry", "pending")
    data["label_source"] = "unknown_source"
    data["final_polygon"] = None
    _assert_invalid(data)


def test_reject_unknown_status() -> None:
    """Unknown review_status must be rejected."""
    data = _base("R0", "joint_geometry", "pending")
    data["review_status"] = "approved"
    data["final_polygon"] = None
    _assert_invalid(data)


def test_reject_r0_wrong_source() -> None:
    """R0 must use joint_geometry, not opencv_refined."""
    data = _base("R0", "opencv_refined", "pending")
    data["final_polygon"] = _poly()
    _assert_invalid(data)


def test_reject_r1_wrong_source() -> None:
    """R1 must use opencv_refined, not joint_geometry."""
    data = _base("R1", "joint_geometry", "pending")
    data["final_polygon"] = _poly()
    _assert_invalid(data)


def test_reject_r2_wrong_source() -> None:
    """R2 must use human_accepted or human_edited, not opencv_refined."""
    data = _base("R2", "opencv_refined", "accepted")
    data["final_polygon"] = _poly()
    data["reviewer_id"] = "r001"
    data["reviewed_at"] = "2026-08-01T12:00:00Z"
    _assert_invalid(data)


def test_reject_r3_wrong_source() -> None:
    """R3 must use human_consensus, not human_edited."""
    data = _base("R3", "human_edited", "accepted")
    data["final_polygon"] = _poly()
    data["reviewer_id"] = "r001"
    data["reviewed_at"] = "2026-08-01T12:00:00Z"
    _assert_invalid(data)


def test_reject_r0_accepted_status() -> None:
    """R0 cannot have accepted status."""
    data = _base("R0", "joint_geometry", "accepted")
    data["final_polygon"] = None
    _assert_invalid(data)


def test_reject_r1_accepted_status() -> None:
    """R1 cannot have accepted status (only pending/uncertain/rejected)."""
    data = _base("R1", "opencv_refined", "accepted")
    data["final_polygon"] = _poly()
    _assert_invalid(data)


def test_reject_r2_rejected_status() -> None:
    """R2 cannot have rejected status (only accepted/edited)."""
    data = _base("R2", "human_accepted", "rejected")
    data["final_polygon"] = _poly()
    data["reviewer_id"] = "r001"
    data["reviewed_at"] = "2026-08-01T12:00:00Z"
    _assert_invalid(data)


def test_reject_r2_pending_status() -> None:
    """R2 cannot have pending status."""
    data = _base("R2", "human_accepted", "pending")
    data["final_polygon"] = _poly()
    data["reviewer_id"] = "r001"
    data["reviewed_at"] = "2026-08-01T12:00:00Z"
    _assert_invalid(data)


def test_reject_r3_rejected_status() -> None:
    """R3 cannot have rejected status (only accepted/adjudicated)."""
    data = _base("R3", "human_consensus", "rejected")
    data["final_polygon"] = _poly()
    data["reviewer_id"] = "r001"
    data["reviewed_at"] = "2026-08-01T12:00:00Z"
    _assert_invalid(data)


def test_reject_r3_pending_status() -> None:
    """R3 cannot have pending status."""
    data = _base("R3", "human_consensus", "pending")
    data["final_polygon"] = _poly()
    data["reviewer_id"] = "r001"
    data["reviewed_at"] = "2026-08-01T12:00:00Z"
    _assert_invalid(data)


# ---------------------------------------------------------------------------
# R1/R2/R3 final_polygon required; R0 may be null
# ---------------------------------------------------------------------------

def test_reject_r1_missing_final_polygon() -> None:
    """R1 must have final_polygon; null is invalid."""
    data = _base("R1", "opencv_refined", "pending")
    data["final_polygon"] = None
    _assert_invalid(data)


def test_reject_r2_missing_final_polygon() -> None:
    """R2 must have final_polygon; null is invalid."""
    data = _base("R2", "human_accepted", "accepted")
    data["final_polygon"] = None
    data["reviewer_id"] = "r001"
    data["reviewed_at"] = "2026-08-01T12:00:00Z"
    _assert_invalid(data)


def test_reject_r3_missing_final_polygon() -> None:
    """R3 must have final_polygon; null is invalid."""
    data = _base("R3", "human_consensus", "accepted")
    data["final_polygon"] = None
    data["reviewer_id"] = "r001"
    data["reviewed_at"] = "2026-08-01T12:00:00Z"
    _assert_invalid(data)


def test_reject_r0_has_final_polygon_still_valid() -> None:
    """R0 MAY have final_polygon (optional), so having it is fine."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = _poly()
    _assert_valid(data)


# ---------------------------------------------------------------------------
# R2/R3 reviewer constraints
# ---------------------------------------------------------------------------

def test_reject_r2_missing_reviewer_id() -> None:
    """R2 must have non-empty reviewer_id."""
    data = _base("R2", "human_accepted", "accepted")
    data["final_polygon"] = _poly()
    data["reviewer_id"] = None
    data["reviewed_at"] = "2026-08-01T12:00:00Z"
    _assert_invalid(data)


def test_reject_r2_empty_reviewer_id() -> None:
    """R2 reviewer_id cannot be empty string."""
    data = _base("R2", "human_accepted", "accepted")
    data["final_polygon"] = _poly()
    data["reviewer_id"] = ""
    data["reviewed_at"] = "2026-08-01T12:00:00Z"
    _assert_invalid(data)


def test_reject_r2_missing_reviewed_at() -> None:
    """R2 must have reviewed_at."""
    data = _base("R2", "human_accepted", "accepted")
    data["final_polygon"] = _poly()
    data["reviewer_id"] = "r001"
    data["reviewed_at"] = None
    _assert_invalid(data)


def test_reject_r3_missing_reviewer_id() -> None:
    """R3 must have non-empty reviewer_id."""
    data = _base("R3", "human_consensus", "accepted")
    data["final_polygon"] = _poly()
    data["reviewer_id"] = None
    data["reviewed_at"] = "2026-08-01T12:00:00Z"
    _assert_invalid(data)


def test_reject_r3_missing_reviewed_at() -> None:
    """R3 must have reviewed_at."""
    data = _base("R3", "human_consensus", "accepted")
    data["final_polygon"] = _poly()
    data["reviewer_id"] = "r001"
    data["reviewed_at"] = None
    _assert_invalid(data)


# ---------------------------------------------------------------------------
# Human conclusions require reviewer/time
# ---------------------------------------------------------------------------

def test_reject_accepted_without_reviewer() -> None:
    """accepted status without reviewer_id must be rejected."""
    data = _base("R2", "human_accepted", "accepted")
    data["final_polygon"] = _poly()
    data["reviewer_id"] = None
    data["reviewed_at"] = "2026-08-01T12:00:00Z"
    _assert_invalid(data)


def test_reject_accepted_without_time() -> None:
    """accepted status without reviewed_at must be rejected."""
    data = _base("R2", "human_accepted", "accepted")
    data["final_polygon"] = _poly()
    data["reviewer_id"] = "r001"
    data["reviewed_at"] = None
    _assert_invalid(data)


def test_reject_edited_without_reviewer() -> None:
    """edited status without reviewer_id must be rejected."""
    data = _base("R2", "human_edited", "edited")
    data["final_polygon"] = _poly()
    data["reviewer_id"] = None
    data["reviewed_at"] = "2026-08-01T12:00:00Z"
    _assert_invalid(data)


def test_reject_adjudicated_without_reviewer() -> None:
    """adjudicated status without reviewer must be rejected."""
    data = _base("R3", "human_consensus", "adjudicated")
    data["final_polygon"] = _poly()
    data["reviewer_id"] = None
    data["reviewed_at"] = "2026-08-01T12:00:00Z"
    _assert_invalid(data)


def test_reject_rejected_without_reviewer() -> None:
    """rejected status without reviewer must be rejected."""
    data = _base("R0", "joint_geometry", "rejected")
    data["final_polygon"] = None
    data["reviewer_id"] = None
    data["reviewed_at"] = None
    _assert_invalid(data)


# ---------------------------------------------------------------------------
# Reject extra fields (closed schema)
# ---------------------------------------------------------------------------

def test_reject_extra_field() -> None:
    """additionalProperties: false rejects unknown fields."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["extra_field"] = "not_allowed"
    _assert_invalid(data)


# ---------------------------------------------------------------------------
# Reject wrong SHA256 patterns
# ---------------------------------------------------------------------------

def test_reject_wrong_sha256_parameter_hash() -> None:
    """parameter_hash must be 64-char hex."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["parameter_hash"] = "not-a-sha256"
    _assert_invalid(data)


def test_reject_short_sha256() -> None:
    """SHA256 must be exactly 64 characters."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["parameter_hash"] = "A" * 63
    _assert_invalid(data)


def test_reject_non_hex_sha256() -> None:
    """SHA256 hex must contain only A-Fa-f0-9."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["parameter_hash"] = "G" * 64  # G is not a valid hex char
    _assert_invalid(data)


def test_reject_wrong_sha256_homography() -> None:
    """homography_sha256 must match 64-char hex pattern."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["homography_sha256"] = "short"
    _assert_invalid(data)


# ---------------------------------------------------------------------------
# Reject illegal polygon
# ---------------------------------------------------------------------------

def test_reject_polygon_too_few_points() -> None:
    """Polygon must have at least 3 points."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["proposal_polygon"] = [[0.0, 0.0], [100.0, 0.0]]  # only 2 points
    _assert_invalid(data)


def test_reject_polygon_not_array_of_arrays() -> None:
    """Polygon points must be [x, y] arrays."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["proposal_polygon"] = ["not", "an", "array"]
    _assert_invalid(data)


def test_reject_polygon_wrong_point_type() -> None:
    """Polygon points must be [number, number]."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["proposal_polygon"] = [["a", "b"], ["c", "d"], ["e", "f"]]
    _assert_invalid(data)


def test_reject_final_polygon_too_few_points() -> None:
    """final_polygon must have >= 3 points if present."""
    data = _base("R1", "opencv_refined", "pending")
    data["final_polygon"] = [[0.0, 0.0], [100.0, 0.0]]
    _assert_invalid(data)


# ---------------------------------------------------------------------------
# Reject NaN/Inf in numeric fields
# Note: JSON Schema Draft 2020-12 cannot natively detect NaN/Inf at schema level
# (they pass type:number and minimum/maximum). Rejection is enforced by a
# Python-level pre-check function used in tests here.
# ---------------------------------------------------------------------------

def _validate_strict(data: dict[str, Any]) -> list[ValidationError]:
    """Draft 2020-12 schema validation + Python-level NaN/Inf and date-time pre-check."""
    import math as _math
    from datetime import datetime
    from re import compile as re_compile
    errors: list[ValidationError] = []

    # Python-level NaN/Inf pre-check (schema cannot catch these)
    for field in ("alignment_confidence", "anatomical_confidence"):
        val = data.get(field)
        if val is not None and isinstance(val, float) and not _math.isfinite(val):
            errors.append(
                ValidationError(
                    f"{field} must be finite, got {val}",
                    instance=val,
                    schema={},
                    validator="finite",
                    validator_value=[],
                    path=f"/{field}",
                    schema_path=f"#/properties/{field}",
                )
            )

    # Python-level ISO date-time pre-check (format:date-time removed from nullable fields)
    ISO8601_RE = re_compile(
        r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)?$"
    )
    for field, instance_path in (
        ("reviewed_at", "/reviewed_at"),
        ("provenance.created_at", "/provenance"),
    ):
        if field == "provenance.created_at":
            val = (data.get("provenance") or {}).get("created_at")
        else:
            val = data.get(field)
        if isinstance(val, str):
            try:
                datetime.fromisoformat(val.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                errors.append(
                    ValidationError(
                        f"{field} must be ISO date-time, got {val!r}",
                        instance=val,
                        schema={},
                        validator="date-time",
                        validator_value=[],
                        path=instance_path,
                        schema_path=f"#/properties/{field}",
                    )
                )

    # Schema validation
    if not errors:
        errors.extend(_validate(data))
    return errors


def test_reject_nan_alignment_confidence() -> None:
    """alignment_confidence must be finite (no NaN). Python-level check."""
    import math
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["alignment_confidence"] = math.nan
    errors = _validate_strict(data)
    assert errors, "NaN alignment_confidence must be rejected by strict validator"


def test_reject_inf_anatomical_confidence() -> None:
    """anatomical_confidence must be finite (no Inf). Python-level check."""
    import math
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["anatomical_confidence"] = math.inf
    errors = _validate_strict(data)
    assert errors, "+Inf anatomical_confidence must be rejected by strict validator"


def test_reject_negative_infinity_alignment() -> None:
    """alignment_confidence must be finite (no -Inf). Python-level check."""
    import math
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["alignment_confidence"] = -math.inf
    errors = _validate_strict(data)
    assert errors, "-Inf alignment_confidence must be rejected by strict validator"


def test_reject_nan_but_valid_float_passes() -> None:
    """Valid floats should pass the strict validator."""
    import math
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["alignment_confidence"] = 0.5
    data["anatomical_confidence"] = 0.0
    errors = _validate_strict(data)
    assert not errors, f"Valid float confidence should pass: {errors}"


# ---------------------------------------------------------------------------
# Reject missing provenance
# ---------------------------------------------------------------------------

def test_reject_missing_provenance() -> None:
    """provenance is required."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    del data["provenance"]
    _assert_invalid(data)


def test_reject_provenance_missing_source_artifacts() -> None:
    """provenance.source_artifacts is required and non-empty."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["provenance"] = {
        "source_artifacts": [],
        "generator": "test",
        "created_at": "2026-08-01T12:00:00Z",
    }
    _assert_invalid(data)


def test_reject_provenance_missing_generator() -> None:
    """provenance.generator is required."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["provenance"] = {
        "source_artifacts": ["artifact1"],
        "created_at": "2026-08-01T12:00:00Z",
    }
    _assert_invalid(data)


def test_reject_provenance_missing_created_at() -> None:
    """provenance.created_at is required."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["provenance"] = {
        "source_artifacts": ["artifact1"],
        "generator": "test",
    }
    _assert_invalid(data)


def test_reject_provenance_extra_field() -> None:
    """provenance.additionalProperties: false rejects unknown fields."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["provenance"] = {
        "source_artifacts": ["artifact1"],
        "generator": "test",
        "created_at": "2026-08-01T12:00:00Z",
        "extra": "not allowed",
    }
    _assert_invalid(data)


# ---------------------------------------------------------------------------
# Reject malformed dates
# ---------------------------------------------------------------------------

def test_reject_bad_reviewed_at_format() -> None:
    """reviewed_at must be ISO date-time (Python-level check)."""
    data = _base("R2", "human_accepted", "accepted")
    data["final_polygon"] = _poly()
    data["reviewer_id"] = "r001"
    data["reviewed_at"] = "not-a-date"
    errors = _validate_strict(data)
    assert errors, "Non-ISO reviewed_at must be rejected by strict validator"


def test_reject_bad_provenance_created_at() -> None:
    """provenance.created_at must be ISO date-time (Python-level check)."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["provenance"]["created_at"] = "2026/08/01"
    errors = _validate_strict(data)
    assert errors, "Non-ISO provenance.created_at must be rejected by strict validator"


def test_reject_provenance_bad_date_time() -> None:
    """provenance.created_at must be a valid date-time (Python-level check)."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["provenance"]["created_at"] = "2026-13-40T99:99:99Z"  # impossible date
    errors = _validate_strict(data)
    assert errors, "Impossible date must be rejected by strict validator"


# ---------------------------------------------------------------------------
# Reject invalid reason_codes
# ---------------------------------------------------------------------------

def test_reject_unknown_reason_code() -> None:
    """reason_codes items must be from the allowed enum."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["reason_codes"] = ["unknown_reason"]
    _assert_invalid(data)


def test_reject_reason_codes_not_unique() -> None:
    """reason_codes must have unique items."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["reason_codes"] = ["alignment", "alignment"]
    _assert_invalid(data)


# ---------------------------------------------------------------------------
# Reject sample_id format
# ---------------------------------------------------------------------------

def test_reject_bad_sample_id_prefix() -> None:
    """sample_id must start with 'slp::'."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["sample_id"] = "bad::prefix"
    _assert_invalid(data)


# ---------------------------------------------------------------------------
# Reject subject_id format
# ---------------------------------------------------------------------------

def test_reject_bad_subject_id_format() -> None:
    """subject_id must be 5 digits."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["subject_id"] = "abc"
    _assert_invalid(data)


def test_reject_subject_id_wrong_length() -> None:
    """subject_id must be exactly 5 digits."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["subject_id"] = "0001"
    _assert_invalid(data)


# ---------------------------------------------------------------------------
# Reject frame_index out of range
# ---------------------------------------------------------------------------

def test_reject_frame_index_zero() -> None:
    """frame_index must be >= 1."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["frame_index"] = 0
    _assert_invalid(data)


def test_reject_frame_index_46() -> None:
    """frame_index must be <= 45."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["frame_index"] = 46
    _assert_invalid(data)


# ---------------------------------------------------------------------------
# Reject quality_flags non-unique
# ---------------------------------------------------------------------------

def test_reject_quality_flags_not_unique() -> None:
    """quality_flags must have unique items."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["quality_flags"] = ["occluded", "occluded"]
    _assert_invalid(data)


# ---------------------------------------------------------------------------
# Reject coverage out of [0, 1] for confidence fields
# ---------------------------------------------------------------------------

def test_reject_alignment_confidence_too_high() -> None:
    """alignment_confidence must be <= 1.0."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["alignment_confidence"] = 1.5
    _assert_invalid(data)


def test_reject_anatomical_confidence_negative() -> None:
    """anatomical_confidence must be >= 0.0."""
    data = _base("R0", "joint_geometry", "pending")
    data["final_polygon"] = None
    data["anatomical_confidence"] = -0.1
    _assert_invalid(data)


# ---------------------------------------------------------------------------
# Training default restriction (R0/R1 not default training labels)
# ---------------------------------------------------------------------------

def test_schema_documents_r0_r1_not_default_training() -> None:
    """Schema description must state R0/R1 are not default training labels."""
    schema = _schema()
    desc = schema.get("description", "")
    assert "R0" in desc and "R1" in desc
    assert "R2" in desc and "R3" in desc
    # The schema description should mention default training labels
    assert "default training" in desc.lower()


# ---------------------------------------------------------------------------
# Reject all 10 authorized regions are present
# ---------------------------------------------------------------------------

def test_all_10_authorized_regions_present() -> None:
    """All 10 authorized region_ids must be defined."""
    schema = _schema()
    regions = set(schema["properties"]["region_id"]["enum"])
    expected = {
        "head_neck",
        "shoulder_left",
        "shoulder_right",
        "thorax_back",
        "abdomen_waist",
        "pelvis_hip",
        "thigh_left",
        "thigh_right",
        "lower_leg_foot_left",
        "lower_leg_foot_right",
    }
    assert regions == expected
