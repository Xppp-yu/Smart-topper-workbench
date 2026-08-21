from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

from topper_perception.io.slp_inventory import (
    COVER_CONDITIONS,
    SETTING_MODALITIES,
    audit_subject_annotations,
    inventory_modality_group,
    inventory_slp_dataset,
    resolve_slp_root,
    summarise_slp_inventory,
    audit_slp_annotations,
)


def _make_subject(root: Path, setting: str, subject_id: str, *, frames: int = 2) -> Path:
    subject = root / setting / subject_id
    for modality in SETTING_MODALITIES[setting]:
        for cover in COVER_CONDITIONS:
            group = subject / modality / cover
            group.mkdir(parents=True, exist_ok=True)
            suffix = ".npy" if modality.endswith("raw") else ".png"
            prefix = "" if suffix == ".npy" else "image_"
            for index in range(1, frames + 1):
                (group / f"{prefix}{index:06d}{suffix}").write_bytes(b"x")

    joints = np.zeros((3, 14, 45), dtype=np.float64)
    savemat(subject / "joints_gt_RGB.mat", {"joints_gt": joints})
    savemat(subject / "joints_gt_IR.mat", {"joints_gt": joints})
    for modality in ("RGB", "IR", "depth"):
        np.save(subject / f"align_PTr_{modality}.npy", np.eye(3))
    if setting == "danaLab":
        np.save(subject / "PMcali.npy", np.ones((3, 45)))
    return subject


def test_resolve_slp_root_accepts_parent_and_direct_root(tmp_path: Path) -> None:
    slp_root = tmp_path / "SLP2022" / "SLP"
    for setting in ("danaLab", "simLab"):
        (slp_root / setting).mkdir(parents=True)

    assert resolve_slp_root(tmp_path) == slp_root
    assert resolve_slp_root(slp_root) == slp_root


def test_modality_group_reports_complete_and_missing_frames(tmp_path: Path) -> None:
    slp_root = tmp_path / "SLP"
    for setting in ("danaLab", "simLab"):
        (slp_root / setting).mkdir(parents=True)
    subject = _make_subject(slp_root, "danaLab", "00001", frames=2)

    complete = inventory_modality_group(
        slp_root,
        setting="danaLab",
        subject_dir=subject,
        modality="PM",
        cover_condition="uncover",
        expected_frames=2,
    ).as_dict()
    assert complete["status"] == "OK"
    assert complete["file_count"] == 2

    (subject / "PM" / "cover1" / "image_000002.png").unlink()
    incomplete = inventory_modality_group(
        slp_root,
        setting="danaLab",
        subject_dir=subject,
        modality="PM",
        cover_condition="cover1",
        expected_frames=2,
    ).as_dict()
    assert incomplete["status"] == "ERROR"
    assert "missing_frame_indices" in str(incomplete["error_codes"])


def test_annotation_audit_preserves_truth_provenance(tmp_path: Path) -> None:
    slp_root = tmp_path / "SLP"
    for setting in ("danaLab", "simLab"):
        (slp_root / setting).mkdir(parents=True)
    subject = _make_subject(slp_root, "danaLab", "00001")

    row = audit_subject_annotations("danaLab", subject).as_dict()

    assert row["joints_gt_rgb_shape"] == "3x14x45"
    assert row["rgb_ir_joint_gt_source"] == "manual_original"
    assert row["mapped_joint_gt_status"] == "derived_homography_bias_possible"
    assert row["region_ground_truth_present"] is False
    assert row["status"] == "WARN"
    assert "region_ground_truth_absent" in str(row["warning_codes"])


def test_dataset_summary_keeps_simlab_without_pressure_as_expected(tmp_path: Path) -> None:
    slp_root = tmp_path / "SLP"
    for setting in ("danaLab", "simLab"):
        (slp_root / setting).mkdir(parents=True)
    _make_subject(slp_root, "danaLab", "00001", frames=2)
    _make_subject(slp_root, "simLab", "00001", frames=2)

    inventory = list(inventory_slp_dataset(slp_root, expected_frames=2))
    annotations = list(audit_slp_annotations(slp_root))
    summary = summarise_slp_inventory(inventory, annotations, slp_root=slp_root)

    assert summary["subjects"] == 2
    assert summary["subjects_by_setting"] == {"danaLab": 1, "simLab": 1}
    assert summary["inventory_groups"] == 33
    assert summary["group_status_counts"] == {"OK": 33}
    assert summary["total_frame_files"] == 66
    assert summary["annotation_warning_counts"] == {"region_ground_truth_absent": 2}
    assert summary["region_ground_truth"]["present"] is False
