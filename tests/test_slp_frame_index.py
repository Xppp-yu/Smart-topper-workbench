from __future__ import annotations

from pathlib import Path

import topper_perception.io.slp_frame_index as slp_frame_index
from topper_perception.io.slp_frame_index import (
    build_slp_frame_index,
    build_subject_cover_rows,
    validate_frame_index_rows,
)
from topper_perception.io.slp_inventory import COVER_CONDITIONS, SETTING_MODALITIES


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


def test_frame_rows_pair_modalities_by_explicit_frame_index(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    subject = _make_subject(root, "danaLab", "00001", frames=2)

    rows = list(
        build_subject_cover_rows(
            root,
            setting="danaLab",
            subject_dir=subject,
            cover_condition="uncover",
            expected_frames=2,
        )
    )

    assert len(rows) == 2
    first = rows[0].as_dict()
    assert first["sample_id"] == "slp::danaLab::00001::uncover::000001"
    assert first["frame_index"] == 1
    assert first["rgb_uri"].endswith("RGB/uncover/image_000001.png")
    assert first["irraw_uri"].endswith("IRraw/uncover/000001.npy")
    assert first["pm_uri"].endswith("PM/uncover/image_000001.png")
    assert first["missing_modalities"] == ""
    assert first["quarantine"] is False


def test_simlab_pressure_absence_is_expected_not_quarantine(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    subject = _make_subject(root, "simLab", "00003", frames=2)

    rows = list(
        build_subject_cover_rows(
            root,
            setting="simLab",
            subject_dir=subject,
            cover_condition="cover1",
            expected_frames=2,
        )
    )

    assert len(rows) == 2
    for row in rows:
        values = row.as_dict()
        assert values["pm_uri"] == ""
        assert values["expected_missing_modalities"] == "PM"
        assert values["missing_modalities"] == ""
        assert values["quarantine"] is False


def test_missing_depthraw_is_preserved_per_frame(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    subject = _make_subject(root, "simLab", "00003", frames=2)
    for path in (subject / "depthRaw" / "cover2").glob("*.npy"):
        path.unlink()

    rows = list(
        build_subject_cover_rows(
            root,
            setting="simLab",
            subject_dir=subject,
            cover_condition="cover2",
            expected_frames=2,
        )
    )

    assert len(rows) == 2
    for row in rows:
        values = row.as_dict()
        assert values["depthraw_uri"] == ""
        assert values["missing_modalities"] == "depthRaw"
        assert values["quarantine"] is True
        assert "missing_depthRaw" in values["quality_flags"]


def test_wrong_depthraw_extension_does_not_satisfy_raw_slot(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    subject = _make_subject(root, "simLab", "00003", frames=1)
    raw_dir = subject / "depthRaw" / "uncover"
    (raw_dir / "000001.npy").unlink()
    (raw_dir / "image_000001.png").write_bytes(b"not-raw")

    row = next(
        build_subject_cover_rows(
            root,
            setting="simLab",
            subject_dir=subject,
            cover_condition="uncover",
            expected_frames=1,
        )
    ).as_dict()

    assert row["depthraw_uri"] == ""
    assert row["missing_modalities"] == "depthRaw"
    assert row["quarantine"] is True


def test_duplicate_frame_file_fails_closed_instead_of_sort_pairing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = _make_root(tmp_path)
    subject = _make_subject(root, "danaLab", "00001", frames=2)
    original_scan = slp_frame_index._scan_modality_directory

    def scan_with_duplicate(group_dir: Path, *, modality: str) -> dict[int, list[Path]]:
        matches = original_scan(group_dir, modality=modality)
        if modality == "RGB" and group_dir.name == "uncover":
            matches[1] = [
                group_dir / "image_000001.png",
                group_dir / "synthetic_duplicate_for_frame_000001.png",
            ]
        return matches

    monkeypatch.setattr(slp_frame_index, "_scan_modality_directory", scan_with_duplicate)

    rows = list(
        build_subject_cover_rows(
            root,
            setting="danaLab",
            subject_dir=subject,
            cover_condition="uncover",
            expected_frames=2,
        )
    )
    first = rows[0].as_dict()

    assert first["rgb_uri"] == ""
    assert first["ambiguous_modalities"] == "RGB"
    assert first["quarantine"] is True
    assert "ambiguous_RGB" in first["quality_flags"]


def test_full_index_has_unique_primary_keys_and_integrity_summary(tmp_path: Path) -> None:
    root = _make_root(tmp_path)
    _make_subject(root, "danaLab", "00001", frames=2)
    _make_subject(root, "simLab", "00001", frames=2)

    rows = list(build_slp_frame_index(root, expected_frames=2))
    summary = validate_frame_index_rows(rows)

    assert len(rows) == 12  # 2 subjects x 3 covers x 2 frames
    assert summary["rows"] == 12
    assert summary["unique_primary_keys"] == 12
    assert summary["duplicate_primary_key_count"] == 0
    assert summary["ambiguous_modality_frame_counts"] == {}
    assert summary["expected_missing_modality_frame_counts"] == {"PM": 6}
    assert summary["pairing_method"] == "explicit_frame_index_join"
    assert summary["silent_imputation"] is False
