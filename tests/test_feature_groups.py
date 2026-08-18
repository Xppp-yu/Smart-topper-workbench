"""Tests for deterministic ablation grouping of the 71 P4a feature columns.

The ablation protocol in P5.1-B runs the top-2 round-1 candidates on five
feature subsets (intensity only / mask-geometry-shape only / grid zones only /
intensity+geometry / all).  ``feature_group_columns`` classifies any feature
column list by prefix so the partition is deterministic and can never silently
miss a column.
"""

from __future__ import annotations

from topper_perception.features.groups import feature_group_columns

_INTENSITY = (
    [f"intensity_{stat}" for stat in ("sum", "mean", "std", "min", "max", "p25", "p50", "p75", "p90", "p95", "p99")]
    + ["nonzero_cell_count", "nonzero_fraction", "positive_mean"]
)
_GEOMETRY = [
    "mask_threshold_raw", "mask_cell_count", "mask_fraction", "component_count",
    "bbox_row_min", "bbox_row_max", "bbox_column_min", "bbox_column_max",
    "bbox_height", "bbox_width", "bbox_area",
    "centroid_row_fraction", "centroid_column_fraction",
    "cop_row_fraction", "cop_column_fraction",
    "principal_axis_degrees", "principal_axis_anisotropy",
    "contact_signal_sum", "bbox_aspect_ratio", "mask_extent", "mask_compactness",
]
_ZONES = [
    f"zone_{stat}_r{row}c{col}"
    for row in range(4)
    for col in range(3)
    for stat in ("sum", "fraction", "peak")
]
_ALL = _INTENSITY + _GEOMETRY + _ZONES


def test_all_71_columns_partition_into_three_groups() -> None:
    assert len(_ALL) == 71
    groups = feature_group_columns(_ALL)
    assert set(groups) == {"intensity", "geometry", "zones"}
    assert sum(len(values) for values in groups.values()) == 71
    flattened = [column for values in groups.values() for column in values]
    assert sorted(flattened) == sorted(_ALL)


def test_group_sizes_match_the_frozen_feature_schema() -> None:
    groups = feature_group_columns(_ALL)
    assert len(groups["intensity"]) == 14
    assert len(groups["geometry"]) == 21
    assert len(groups["zones"]) == 36


def test_ablation_group_composition_matches_protocol() -> None:
    groups = feature_group_columns(_ALL)
    assert len(groups["intensity"] + groups["geometry"]) == 35  # intensity + geometry
    assert len(groups["zones"]) == 36  # grid zones only
    assert len(groups["intensity"]) == 14  # intensity only


def test_unknown_column_is_rejected() -> None:
    try:
        feature_group_columns(["mystery_feature"])
    except ValueError as error:
        assert "mystery_feature" in str(error)
    else:
        raise AssertionError("expected ValueError for unknown feature column")
