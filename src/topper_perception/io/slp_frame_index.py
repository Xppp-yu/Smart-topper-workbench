"""Build the SLP frame-level master index without modifying raw data.

The index uses an explicit key of setting / subject / cover / frame. Modalities
are joined by parsed frame indices; directory ordering is never used for
pairing. Missing and ambiguous modality files remain visible as quality flags
rather than being imputed or silently dropped.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
import re

from .slp_inventory import (
    COVER_CONDITIONS,
    EXPECTED_FRAMES_PER_GROUP,
    SETTINGS,
    SETTING_MODALITIES,
    iter_subject_directories,
    resolve_slp_root,
)


CANONICAL_MODALITIES = ("RGB", "IR", "IRraw", "depth", "depthRaw", "PM")
_FRAME_PATTERN = re.compile(
    r"^(?:image_)?(?P<index>\d{6})\.(?P<extension>png|npy)$",
    re.IGNORECASE,
)

FRAME_INDEX_COLUMNS = (
    "sample_id",
    "setting",
    "subject_id",
    "cover_condition",
    "frame_index",
    "rgb_uri",
    "ir_uri",
    "irraw_uri",
    "depth_uri",
    "depthraw_uri",
    "pm_uri",
    "missing_modalities",
    "expected_missing_modalities",
    "ambiguous_modalities",
    "quarantine",
    "quality_flags",
)

_URI_COLUMN = {
    "RGB": "rgb_uri",
    "IR": "ir_uri",
    "IRraw": "irraw_uri",
    "depth": "depth_uri",
    "depthRaw": "depthraw_uri",
    "PM": "pm_uri",
}


@dataclass(frozen=True, slots=True)
class SlpFrameIndexRow:
    values: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {column: self.values.get(column, "") for column in FRAME_INDEX_COLUMNS}


def _scan_modality_directory(group_dir: Path) -> dict[int, list[Path]]:
    """Return explicit frame-index -> files mapping for one modality directory."""
    files_by_index: dict[int, list[Path]] = {}
    if not group_dir.is_dir():
        return files_by_index

    for source_file in group_dir.iterdir():
        if not source_file.is_file():
            continue
        match = _FRAME_PATTERN.fullmatch(source_file.name)
        if match is None:
            continue
        frame_index = int(match.group("index"))
        files_by_index.setdefault(frame_index, []).append(source_file)
    return files_by_index


def _relative_uri(path: Path, slp_root: Path) -> str:
    return path.relative_to(slp_root).as_posix()


def _expected_modalities(setting: str) -> set[str]:
    return set(SETTING_MODALITIES[setting])


def build_subject_cover_rows(
    slp_root: Path,
    *,
    setting: str,
    subject_dir: Path,
    cover_condition: str,
    expected_frames: int = EXPECTED_FRAMES_PER_GROUP,
) -> Iterator[SlpFrameIndexRow]:
    """Build deterministic frame rows for one subject and cover condition."""
    if setting not in SETTINGS:
        raise ValueError(f"unsupported SLP setting: {setting}")
    if cover_condition not in COVER_CONDITIONS:
        raise ValueError(f"unsupported SLP cover condition: {cover_condition}")

    expected_modalities = _expected_modalities(setting)
    modality_files = {
        modality: _scan_modality_directory(subject_dir / modality / cover_condition)
        for modality in CANONICAL_MODALITIES
    }

    for frame_index in range(1, expected_frames + 1):
        values: dict[str, object] = {
            "sample_id": (
                f"slp::{setting}::{subject_dir.name}::{cover_condition}::{frame_index:06d}"
            ),
            "setting": setting,
            "subject_id": subject_dir.name,
            "cover_condition": cover_condition,
            "frame_index": frame_index,
        }
        missing: list[str] = []
        expected_missing: list[str] = []
        ambiguous: list[str] = []
        quality_flags: list[str] = []

        for modality in CANONICAL_MODALITIES:
            matches = modality_files[modality].get(frame_index, [])
            uri_column = _URI_COLUMN[modality]
            if modality not in expected_modalities:
                values[uri_column] = ""
                expected_missing.append(modality)
                continue
            if len(matches) == 1:
                values[uri_column] = _relative_uri(matches[0], slp_root)
            elif len(matches) == 0:
                values[uri_column] = ""
                missing.append(modality)
                quality_flags.append(f"missing_{modality}")
            else:
                # An ambiguous frame must fail closed. Do not pick one by sort order.
                values[uri_column] = ""
                ambiguous.append(modality)
                quality_flags.append(f"ambiguous_{modality}")

        quarantine = bool(missing or ambiguous)
        if quarantine:
            quality_flags.append("quarantine")

        values.update(
            {
                "missing_modalities": ";".join(sorted(missing)),
                "expected_missing_modalities": ";".join(sorted(expected_missing)),
                "ambiguous_modalities": ";".join(sorted(ambiguous)),
                "quarantine": quarantine,
                "quality_flags": ";".join(sorted(set(quality_flags))),
            }
        )
        yield SlpFrameIndexRow(values)


def build_slp_frame_index(
    data_root: Path,
    *,
    expected_frames: int = EXPECTED_FRAMES_PER_GROUP,
) -> Iterator[SlpFrameIndexRow]:
    """Build the full SLP frame index using explicit frame-index joins."""
    slp_root = resolve_slp_root(data_root)
    for setting, subject_dir in iter_subject_directories(slp_root):
        for cover_condition in COVER_CONDITIONS:
            yield from build_subject_cover_rows(
                slp_root,
                setting=setting,
                subject_dir=subject_dir,
                cover_condition=cover_condition,
                expected_frames=expected_frames,
            )


def validate_frame_index_rows(rows: Iterable[SlpFrameIndexRow]) -> dict[str, object]:
    """Return integrity diagnostics; duplicates are never silently resolved."""
    materialized = [row.as_dict() for row in rows]
    key_counts = Counter(
        (
            str(row["setting"]),
            str(row["subject_id"]),
            str(row["cover_condition"]),
            int(row["frame_index"]),
        )
        for row in materialized
    )
    duplicate_keys = [key for key, count in key_counts.items() if count > 1]

    missing_counts: Counter[str] = Counter()
    expected_missing_counts: Counter[str] = Counter()
    ambiguous_counts: Counter[str] = Counter()
    quarantine_rows = 0
    for row in materialized:
        for modality in str(row["missing_modalities"]).split(";"):
            if modality:
                missing_counts[modality] += 1
        for modality in str(row["expected_missing_modalities"]).split(";"):
            if modality:
                expected_missing_counts[modality] += 1
        for modality in str(row["ambiguous_modalities"]).split(";"):
            if modality:
                ambiguous_counts[modality] += 1
        quarantine_rows += int(bool(row["quarantine"]))

    return {
        "rows": len(materialized),
        "unique_primary_keys": len(key_counts),
        "duplicate_primary_key_count": len(duplicate_keys),
        "duplicate_primary_keys": ["/".join(map(str, key)) for key in duplicate_keys],
        "missing_modality_frame_counts": dict(sorted(missing_counts.items())),
        "expected_missing_modality_frame_counts": dict(sorted(expected_missing_counts.items())),
        "ambiguous_modality_frame_counts": dict(sorted(ambiguous_counts.items())),
        "quarantine_rows": quarantine_rows,
        "pairing_method": "explicit_frame_index_join",
        "silent_imputation": False,
    }
