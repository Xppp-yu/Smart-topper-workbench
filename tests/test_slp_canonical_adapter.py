"""Tests for the SLP A05 Canonical Sample / Adapter.

Coverage map (from the A05 task contract):

* single-frame traceability to all raw URIs
* missing modalities are quarantined, not silently imputed
* duplicate frame matches are reported, not sort-paired
* illegal URIs and on-disk absence become quality flags
* joint provenance distinguishes J0 (manual) from J1 (derived) and never
  silently picks a homography direction
* region layer is isolated from the frame layer
* schema round-trips through JSON
* running the adapter does not modify the raw data directory
* the A03 / A04 / S0 / region-schema tests continue to pass (regression)
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from topper_perception.io.slp_canonical import (  # noqa: E402
    CANONICAL_CSV_COLUMNS,
    CANONICAL_SCHEMA_VERSION,
    CanonicalSample,
    FrameLayer,
    HomographyContract,
    J1_STATUS_NOT_GENERATED,
    JOINT_PROVENANCE_J0,
    JointLayer,
    Provenance,
    RAW_COORDINATE_FRAME,
    RAW_COORDINATE_ORIGIN_STATUS,
    REGION_PLACEHOLDER_STATUS,
    REGION_SCHEMA_VERSION,
    RegionLayer,
    SlpCanonicalAdapter,
    _check_uri_existence,
    canonical_sample_to_csv_row,
    load_a03_frame_index_csv,
    load_a04_homography_audit_csv,
    summarise_canonical_samples,
    write_canonical_csv,
    write_canonical_jsonl,
)
from topper_perception.io.slp_frame_index import (  # noqa: E402
    SlpFrameIndexRow,
    build_slp_frame_index,
    build_subject_cover_rows,
)
from topper_perception.io.slp_homography_audit import SlpHomographyAuditRow  # noqa: E402
from topper_perception.io.slp_inventory import (  # noqa: E402
    COVER_CONDITIONS,
    SETTING_MODALITIES,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _make_subject(root: Path, setting: str, subject_id: str, *, frames: int = 2) -> Path:
    subject = root / setting / subject_id
    raw_modalities = {"IRraw", "depthRaw"}
    for modality in SETTING_MODALITIES[setting]:
        for cover in COVER_CONDITIONS:
            group = subject / modality / cover
            group.mkdir(parents=True, exist_ok=True)
            is_raw = modality in raw_modalities
            suffix = ".npy" if is_raw else ".png"
            prefix = "" if is_raw else "image_"
            for index in range(1, frames + 1):
                (group / f"{prefix}{index:06d}{suffix}").write_bytes(b"x")
    return subject


def _make_root(tmp_path: Path) -> Path:
    root = tmp_path / "SLP"
    for setting in ("danaLab", "simLab"):
        (root / setting).mkdir(parents=True)
    return root


def _make_joint_artifacts(subject: Path, *, modalities: tuple[str, ...] = ("RGB", "IR")) -> None:
    for modality in modalities:
        (subject / f"joints_gt_{modality}.mat").write_bytes(b"mat-stub")


def _make_align_artifacts(subject: Path, *, modalities: tuple[str, ...] = ("RGB", "IR", "depth")) -> None:
    for modality in modalities:
        (subject / f"align_PTr_{modality}.npy").write_bytes(b"npy-stub")


def _adapter(
    tmp_path: Path,
    *,
    frames: int = 2,
    missing_depth_raw: bool = False,
) -> SlpCanonicalAdapter:
    root = _make_root(tmp_path)
    subject = _make_subject(root, "danaLab", "00001", frames=frames)
    _make_joint_artifacts(subject)
    _make_align_artifacts(subject)
    if missing_depth_raw:
        for path in (subject / "depthRaw" / "cover2").glob("*.npy"):
            path.unlink()
    rows = list(
        build_slp_frame_index(
            root,
            expected_frames=frames,
        )
    )
    return SlpCanonicalAdapter(
        slp_root=root,
        a03_frame_rows=rows,
        a04_audit_rows=[],
        task_id="TASK-SLP-A05-CANONICAL-ADAPTER-v0.1",
        a03_frame_index_source="synthetic",
        a04_audit_source="none",
    )


def _synthetic_audit_row(
    setting: str,
    subject_id: str,
    modality: str,
    *,
    direction_status: str = "UNRESOLVED_REQUIRES_DOCUMENT_AND_OVERLAY_REVIEW",
    matrix_present: bool = True,
    invertible: bool = True,
    in_bounds: float | None = 0.99,
) -> SlpHomographyAuditRow:
    values = {
        "setting": setting,
        "subject_id": subject_id,
        "modality": modality,
        "matrix_uri": f"{setting}/{subject_id}/align_PTr_{modality}.npy",
        "matrix_present": matrix_present,
        "determinant": 1.0,
        "condition_number": 1.0,
        "rank": 3,
        "invertible": invertible,
        "source_width": 1920,
        "source_height": 1080,
        "pm_width": 192,
        "pm_height": 84,
        "probe_roundtrip_mean_error": 1e-13,
        "probe_roundtrip_max_error": 1e-13,
        "joint_points": 14,
        "direct_joint_in_bounds_rate": in_bounds if modality in ("RGB", "IR") else None,
        "inverse_joint_in_bounds_rate": 0.0 if modality in ("RGB", "IR") else None,
        "direction_status": direction_status,
        "coordinate_origin_status": RAW_COORDINATE_ORIGIN_STATUS,
        "error_codes": "",
    }
    return SlpHomographyAuditRow(values)


# ---------------------------------------------------------------------------
# Single-frame traceability
# ---------------------------------------------------------------------------


def test_canonical_sample_traces_to_all_raw_modality_uris(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, frames=2)
    samples = list(adapter.iter_canonical_samples())
    assert len(samples) == 6  # 3 covers * 2 frames
    first = samples[0]
    assert first.frame.modality_uris["RGB"].endswith("danaLab/00001/RGB/uncover/image_000001.png")
    assert first.frame.modality_uris["IR"].endswith("danaLab/00001/IR/uncover/image_000001.png")
    assert first.frame.modality_uris["IRraw"].endswith("danaLab/00001/IRraw/uncover/000001.npy")
    assert first.frame.modality_uris["depth"].endswith("danaLab/00001/depth/uncover/image_000001.png")
    assert first.frame.modality_uris["depthRaw"].endswith("danaLab/00001/depthRaw/uncover/000001.npy")
    assert first.frame.modality_uris["PM"].endswith("danaLab/00001/PM/uncover/image_000001.png")
    assert first.frame.uri_existence_flags == {
        "RGB": "present",
        "IR": "present",
        "IRraw": "present",
        "depth": "present",
        "depthRaw": "present",
        "PM": "present",
    }


def test_canonical_sample_exposes_joint_provenance_and_homography_contract(
    tmp_path: Path,
) -> None:
    root = _make_root(tmp_path)
    subject = _make_subject(root, "danaLab", "00001", frames=1)
    _make_joint_artifacts(subject)
    _make_align_artifacts(subject)

    rows = list(build_slp_frame_index(root, expected_frames=1))
    audit_rows = [
        _synthetic_audit_row("danaLab", "00001", "RGB"),
        _synthetic_audit_row("danaLab", "00001", "IR"),
        _synthetic_audit_row("danaLab", "00001", "depth"),
    ]
    adapter = SlpCanonicalAdapter(
        slp_root=root,
        a03_frame_rows=rows,
        a04_audit_rows=audit_rows,
    )
    sample = next(adapter.iter_canonical_samples())

    assert sample.joint.j0_source_uris == {
        "RGB": "danaLab/00001/joints_gt_RGB.mat",
        "IR": "danaLab/00001/joints_gt_IR.mat",
    }
    assert sample.joint.j0_present == {"RGB": True, "IR": True}
    assert sample.joint.j0_artifact_count == 14
    assert sample.joint.joint_provenance_status == JOINT_PROVENANCE_J0
    assert sample.joint.j1_status == J1_STATUS_NOT_GENERATED

    rgb_contract = sample.joint.homography_contracts["RGB"]
    assert rgb_contract.matrix_uri == "danaLab/00001/align_PTr_RGB.npy"
    assert rgb_contract.invertible is True
    assert rgb_contract.direction_status == "UNRESOLVED_REQUIRES_DOCUMENT_AND_OVERLAY_REVIEW"
    assert rgb_contract.unresolved_direction is True
    assert rgb_contract.blocked is False
    assert rgb_contract.direct_joint_in_bounds_rate == pytest.approx(0.99)
    assert rgb_contract.coordinate_origin_status == RAW_COORDINATE_ORIGIN_STATUS

    # The adapter must not silently flip the A04 direction.
    assert sample.coordinate_frame == RAW_COORDINATE_FRAME
    assert sample.coordinate_origin_status == RAW_COORDINATE_ORIGIN_STATUS


# ---------------------------------------------------------------------------
# Missing modality -> quarantine
# ---------------------------------------------------------------------------


def test_missing_modality_quarantines_canonical_sample(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    subject = _make_subject(root, "danaLab", "00001", frames=2)
    _make_joint_artifacts(subject)
    _make_align_artifacts(subject)
    for path in (subject / "depthRaw" / "cover2").glob("*.npy"):
        path.unlink()
    rows = list(build_slp_frame_index(root, expected_frames=2))
    audit_rows = [
        _synthetic_audit_row("danaLab", "00001", "RGB"),
        _synthetic_audit_row("danaLab", "00001", "IR"),
        _synthetic_audit_row("danaLab", "00001", "depth"),
    ]
    adapter = SlpCanonicalAdapter(
        slp_root=root,
        a03_frame_rows=rows,
        a04_audit_rows=audit_rows,
    )
    samples = list(adapter.iter_canonical_samples())
    quarantined = [sample for sample in samples if sample.quarantine]
    assert quarantined, "expected at least one quarantined sample"
    depthraw_quarantined = [s for s in quarantined if "depthRaw" in s.frame.missing_modalities]
    assert depthraw_quarantined, "expected quarantined samples to include depthRaw missing"
    for sample in depthraw_quarantined:
        assert "missing_depthRaw" in sample.quality_flags
        assert "missing_modality:depthRaw" in sample.quarantine_reasons
        assert sample.frame.modality_uris["depthRaw"] == ""
        assert sample.frame.uri_existence_flags["depthRaw"] == "absent"
    # Non-quarantined samples from the same subject/cover should not carry
    # the missing depthRaw flag.
    non_quarantined = [s for s in samples if not s.quarantine]
    for sample in non_quarantined:
        assert "missing_depthRaw" not in sample.quality_flags


# ---------------------------------------------------------------------------
# Duplicate frame match -> fail closed (not sort-paired)
# ---------------------------------------------------------------------------


def test_duplicate_frame_match_is_reported_as_quality_flag(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    subject = _make_subject(root, "danaLab", "00001", frames=2)
    _make_joint_artifacts(subject)
    _make_align_artifacts(subject)

    a03_rows = list(build_slp_frame_index(root, expected_frames=2))
    # Build a duplicate A03 row by hand: same primary key, ambiguous RGB slot.
    duplicate_payload = a03_rows[0].as_dict()
    duplicate_payload["ambiguous_modalities"] = "RGB"
    duplicate_payload["quality_flags"] = "ambiguous_RGB;quarantine"
    duplicate_payload["quarantine"] = True
    duplicate_payload["rgb_uri"] = ""
    duplicate_row = SlpFrameIndexRow(duplicate_payload)
    a03_with_dup = a03_rows + [duplicate_row]

    adapter = SlpCanonicalAdapter(
        slp_root=root,
        a03_frame_rows=a03_with_dup,
        a04_audit_rows=[],
    )
    samples = list(adapter.iter_canonical_samples())
    flagged = [s for s in samples if "ambiguous_RGB" in s.quality_flags]
    # The duplicate row keeps the ambiguous marker visible; the adapter does
    # not collapse it to a single row.
    assert flagged, "ambiguous RGB row must remain visible in the canonical sample"
    assert any(s.quarantine for s in flagged)
    # And the ambiguous modality list is exposed on the frame layer.
    for sample in flagged:
        assert "RGB" in sample.frame.ambiguous_modalities


# ---------------------------------------------------------------------------
# Illegal / missing URIs
# ---------------------------------------------------------------------------


def test_uri_pointing_outside_slp_root_is_rejected(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    subject = _make_subject(root, "danaLab", "00001", frames=1)
    _make_joint_artifacts(subject)
    rows = list(build_slp_frame_index(root, expected_frames=1))
    # Inject an absolute URI pointing outside slp_root into the first row.
    payload = rows[0].as_dict()
    payload["rgb_uri"] = "C:/Windows/System32/not-a-real-slp-file.png"
    tampered = SlpFrameIndexRow(payload)
    new_rows = [tampered] + rows[1:]

    adapter = SlpCanonicalAdapter(
        slp_root=root,
        a03_frame_rows=new_rows,
        a04_audit_rows=[],
    )
    samples = list(adapter.iter_canonical_samples())
    target = samples[0]
    assert target.frame.uri_existence_flags["RGB"] == "missing_on_disk"
    assert "uri_missing_on_disk:RGB" in target.quality_flags
    assert "uri_missing_on_disk:RGB" in target.quarantine_reasons
    assert target.quarantine is True


def test_check_uri_existence_reports_unknown_status() -> None:
    root = Path("E:/TeamProjects/datasets/smart-topper/SLP2022/SLP")
    assert _check_uri_existence(root, "") == "absent"
    assert _check_uri_existence(root, "definitely/does/not/exist.png") == "missing_on_disk"
    assert _check_uri_existence(root, "danaLab/00001/RGB/uncover/image_000001.png") == "present"


# ---------------------------------------------------------------------------
# Joint provenance traceability
# ---------------------------------------------------------------------------


def test_j0_missing_is_flagged_and_never_silently_substituted(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    subject = _make_subject(root, "danaLab", "00001", frames=1)
    # Only create RGB joint file; leave IR missing.
    (subject / "joints_gt_RGB.mat").write_bytes(b"mat-stub")
    _make_align_artifacts(subject)
    rows = list(build_slp_frame_index(root, expected_frames=1))
    adapter = SlpCanonicalAdapter(
        slp_root=root,
        a03_frame_rows=rows,
        a04_audit_rows=[
            _synthetic_audit_row("danaLab", "00001", "RGB"),
        ],
    )
    sample = next(adapter.iter_canonical_samples())
    assert sample.joint.j0_present == {"RGB": True, "IR": False}
    assert "j0_missing_IR" in sample.quality_flags
    assert "j0_missing:IR" in sample.quarantine_reasons
    assert sample.quarantine is True


def test_homography_blocked_modality_quarantines_sample(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    subject = _make_subject(root, "danaLab", "00001", frames=1)
    _make_joint_artifacts(subject)
    _make_align_artifacts(subject)
    rows = list(build_slp_frame_index(root, expected_frames=1))
    audit_rows = [
        _synthetic_audit_row("danaLab", "00001", "RGB", direction_status="BLOCKED_MISSING_MATRIX"),
        _synthetic_audit_row("danaLab", "00001", "IR"),
        _synthetic_audit_row("danaLab", "00001", "depth"),
    ]
    adapter = SlpCanonicalAdapter(
        slp_root=root,
        a03_frame_rows=rows,
        a04_audit_rows=audit_rows,
    )
    sample = next(adapter.iter_canonical_samples())
    rgb = sample.joint.homography_contracts["RGB"]
    assert rgb.blocked is True
    assert rgb.direction_status == "BLOCKED_MISSING_MATRIX"
    assert "homography_blocked_RGB" in sample.quality_flags
    assert "homography_blocked:RGB" in sample.quarantine_reasons
    assert sample.quarantine is True


def test_unresolved_direction_is_soft_warning_not_quarantine(tmp_path: Path) -> None:
    """A04 reports direction as ``UNRESOLVED_*``; the adapter must surface
    this as a quality flag but must NOT auto-quarantine every sample on
    that evidence alone (this would be the silent default the A05 contract
    forbids).
    """
    root = _make_root(tmp_path)
    subject = _make_subject(root, "danaLab", "00001", frames=1)
    _make_joint_artifacts(subject)
    _make_align_artifacts(subject)
    rows = list(build_slp_frame_index(root, expected_frames=1))
    audit_rows = [
        _synthetic_audit_row("danaLab", "00001", "RGB"),
        _synthetic_audit_row("danaLab", "00001", "IR"),
        _synthetic_audit_row("danaLab", "00001", "depth"),
    ]
    adapter = SlpCanonicalAdapter(
        slp_root=root,
        a03_frame_rows=rows,
        a04_audit_rows=audit_rows,
    )
    sample = next(adapter.iter_canonical_samples())
    for modality in ("RGB", "IR", "depth"):
        contract = sample.joint.homography_contracts[modality]
        assert contract.unresolved_direction is True
        assert contract.blocked is False
        assert f"homography_unresolved_{modality}" in sample.quality_flags
        # Soft warning must NOT appear in quarantine_reasons.
        assert f"homography_unresolved_direction:{modality}" not in sample.quarantine_reasons
    assert sample.quarantine is False


# ---------------------------------------------------------------------------
# Region layer isolation
# ---------------------------------------------------------------------------


def test_region_layer_is_isolated_from_frame_layer(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, frames=1)
    sample = next(adapter.iter_canonical_samples())
    assert isinstance(sample.frame, FrameLayer)
    assert isinstance(sample.region, RegionLayer)
    assert sample.region.schema_version == REGION_SCHEMA_VERSION
    assert sample.region.annotation_count == 0
    assert sample.region.annotations == ()
    assert sample.region.can_be_used_as_training_truth is False
    assert sample.region.placeholder_status == REGION_PLACEHOLDER_STATUS

    # The adapter must never put per-frame URIs in the region layer.
    region_dict = sample.region.as_dict()
    for modality in ("RGB", "IR", "IRraw", "depth", "depthRaw", "PM"):
        assert modality not in region_dict

    # The frame layer must never carry region annotations.
    frame_dict = sample.frame.as_dict()
    for forbidden in (
        "annotation",
        "label_tier",
        "review_status",
        "polygon",
    ):
        assert forbidden not in frame_dict


# ---------------------------------------------------------------------------
# Schema round-trip
# ---------------------------------------------------------------------------


def test_canonical_sample_serialization_roundtrips_through_json(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, frames=1)
    sample = next(adapter.iter_canonical_samples())
    encoded = json.dumps(sample.as_dict(), ensure_ascii=False, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["sample_id"] == sample.sample_id
    assert decoded["coordinate_frame"] == RAW_COORDINATE_FRAME
    assert decoded["region"]["schema_version"] == REGION_SCHEMA_VERSION
    assert decoded["joint"]["j1_status"] == J1_STATUS_NOT_GENERATED
    # All homography contracts survive the round-trip.
    for modality in ("RGB", "IR", "depth"):
        contract = decoded["joint"]["homography_contracts"][modality]
        assert contract["modality"] == modality


def test_canonical_csv_writes_one_row_per_sample_with_documented_columns(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path, frames=1)
    samples = list(adapter.iter_canonical_samples())
    csv_path = tmp_path / "out.csv"
    write_canonical_csv(samples, csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == len(samples)
    assert tuple(rows[0].keys()) == CANONICAL_CSV_COLUMNS
    for modality in ("RGB", "IR", "IRraw", "depth", "depthRaw", "PM"):
        assert f"{modality.lower()}_uri" in CANONICAL_CSV_COLUMNS or f"{modality}_uri" in CANONICAL_CSV_COLUMNS
    for prefix in ("rgb", "ir", "depth"):
        assert f"{prefix}_homography_direction_status" in CANONICAL_CSV_COLUMNS


def test_canonical_jsonl_writes_one_sample_per_line(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, frames=1)
    samples = list(adapter.iter_canonical_samples())
    jsonl_path = tmp_path / "out.jsonl"
    write_canonical_jsonl(samples, jsonl_path)
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(samples)
    decoded = [json.loads(line) for line in lines]
    assert decoded[0]["region"]["schema_version"] == REGION_SCHEMA_VERSION


def test_canonical_sample_dict_round_trip_preserves_all_layers(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path, frames=1)
    sample = next(adapter.iter_canonical_samples())
    payload = sample.as_dict()
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded["sample_id"] == payload["sample_id"]
    assert decoded["frame"]["modality_uris"] == payload["frame"]["modality_uris"]
    assert decoded["joint"]["j0_present"] == payload["joint"]["j0_present"]
    assert decoded["region"]["placeholder_status"] == REGION_PLACEHOLDER_STATUS
    assert decoded["provenance"]["semantic_direction_auto_selected"] is False
    for modality in ("RGB", "IR", "depth"):
        contract = decoded["joint"]["homography_contracts"][modality]
        assert contract["modality"] == modality
        assert "direction_status" in contract
        assert contract["direction_status"].startswith(
            ("BLOCKED_", "UNRESOLVED_")
        )


# ---------------------------------------------------------------------------
# Raw data integrity
# ---------------------------------------------------------------------------


def _hash_tree(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def test_adapter_does_not_modify_raw_slp_directory(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    subject = _make_subject(root, "danaLab", "00001", frames=2)
    _make_joint_artifacts(subject)
    _make_align_artifacts(subject)

    rows = list(build_slp_frame_index(root, expected_frames=2))
    audit_rows = [
        _synthetic_audit_row("danaLab", "00001", "RGB"),
        _synthetic_audit_row("danaLab", "00001", "IR"),
        _synthetic_audit_row("danaLab", "00001", "depth"),
    ]
    adapter = SlpCanonicalAdapter(
        slp_root=root,
        a03_frame_rows=rows,
        a04_audit_rows=audit_rows,
    )
    before = _hash_tree(root)
    list(adapter.iter_canonical_samples())
    after = _hash_tree(root)
    assert before == after, "adapter must not modify any file in the SLP data root"


# ---------------------------------------------------------------------------
# A03 / A04 / S0 / region-schema regression smoke
# ---------------------------------------------------------------------------


def test_existing_slp_tests_remain_discoverable() -> None:
    """Run the four SLP test files via pytest in a subprocess and assert they
    still pass. This guarantees the adapter does not break A03 / A04 /
    inventory / region-schema tests.
    """
    import subprocess

    targets = [
        "tests/test_slp_frame_index.py",
        "tests/test_slp_inventory.py",
        "tests/test_slp_homography.py",
        "tests/test_slp_region_annotation_schema.py",
    ]
    completed = subprocess.run(
        ["uv", "run", "pytest", "-q", *targets],
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        "Existing SLP tests regressed after A05:\n"
        f"stdout={completed.stdout}\nstderr={completed.stderr}"
    )
    assert "passed" in completed.stdout


# ---------------------------------------------------------------------------
# Summary semantics
# ---------------------------------------------------------------------------


def test_summary_distinguishes_soft_warnings_from_hard_reasons(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    subject = _make_subject(root, "danaLab", "00001", frames=2)
    _make_joint_artifacts(subject)
    _make_align_artifacts(subject)
    for path in (subject / "depthRaw" / "cover2").glob("*.npy"):
        path.unlink()
    rows = list(build_slp_frame_index(root, expected_frames=2))
    audit_rows = [
        _synthetic_audit_row("danaLab", "00001", "RGB"),
        _synthetic_audit_row("danaLab", "00001", "IR"),
        _synthetic_audit_row("danaLab", "00001", "depth"),
    ]
    adapter = SlpCanonicalAdapter(
        slp_root=root,
        a03_frame_rows=rows,
        a04_audit_rows=audit_rows,
    )
    samples = list(adapter.iter_canonical_samples())
    summary = summarise_canonical_samples(samples)

    assert summary["rows"] == len(samples)
    # In our tiny fixture, 1 cover (cover2) has 2 frames with missing depthRaw.
    assert summary["quarantine_rows"] == 2
    assert summary["quarantine_reason_counts"] == {"missing_modality:depthRaw": 2}


# ---------------------------------------------------------------------------
# Schema artifact
# ---------------------------------------------------------------------------


CANONICAL_SCHEMA_PATH = (
    PROJECT_ROOT / "configs" / "annotations" / "slp_canonical_sample_v0.1.schema.json"
)


def test_canonical_schema_is_valid_json_and_closed() -> None:
    schema = json.loads(CANONICAL_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["title"] == "SLP Canonical Sample v0.1"
    assert schema["additionalProperties"] is False
    provenance = schema["$defs"]["provenance"]["properties"]
    assert provenance["canonical_schema_version"]["const"] == CANONICAL_SCHEMA_VERSION
    assert provenance["task_id"]["const"] == "TASK-SLP-A05-CANONICAL-ADAPTER-v0.1"
    assert provenance["adapter_version"]["const"] == "slp_canonical_adapter_v0.1"


def test_canonical_schema_forbids_region_training_truth_and_back_writes() -> None:
    schema = json.loads(CANONICAL_SCHEMA_PATH.read_text(encoding="utf-8"))
    region_def = schema["$defs"]["region_layer"]["properties"]
    assert region_def["annotation_count"]["const"] == 0
    assert region_def["can_be_used_as_training_truth"]["const"] is False
    provenance_def = schema["$defs"]["provenance"]["properties"]
    for forbidden in (
        "subject_split_applied",
        "review_status_applied",
        "model_prediction_applied",
        "semantic_direction_auto_selected",
        "coordinate_origin_auto_shifted",
        "silent_imputation",
    ):
        assert provenance_def[forbidden]["const"] is False


def test_canonical_schema_requires_frame_joint_region_layers() -> None:
    schema = json.loads(CANONICAL_SCHEMA_PATH.read_text(encoding="utf-8"))
    required = set(schema["required"])
    for field in ("frame", "joint", "region", "provenance", "quality_flags", "quarantine"):
        assert field in required


def test_canonical_schema_requires_a04_geometry_contract_per_modality() -> None:
    schema = json.loads(CANONICAL_SCHEMA_PATH.read_text(encoding="utf-8"))
    contract_required = set(schema["$defs"]["homography_contract"]["required"])
    for field in (
        "modality",
        "matrix_uri",
        "matrix_present",
        "invertible",
        "direction_status",
        "coordinate_origin_status",
        "probe_roundtrip_mean_error",
        "probe_roundtrip_max_error",
        "direct_joint_in_bounds_rate",
        "inverse_joint_in_bounds_rate",
        "error_codes",
        "blocked",
        "unresolved_direction",
    ):
        assert field in contract_required
    direction = schema["$defs"]["homography_contract"]["properties"]["direction_status"]
    # A04 direction must be left as BLOCKED_* or UNRESOLVED_*; the schema
    # refuses to encode a confirmed direction into the canonical sample.
    assert direction["pattern"] == "^(BLOCKED_|UNRESOLVED_).+"
