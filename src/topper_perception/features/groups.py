"""Deterministic ablation grouping of P4a feature columns.

The P5.1-B feature-ablation protocol runs the top-2 round-1 candidates on five
feature subsets (intensity only / mask-geometry-shape only / grid zones only /
intensity+geometry / all).  :func:`feature_group_columns` classifies any feature
column list by prefix, so the partition is deterministic and can never silently
miss a column: an unknown column raises instead of being dropped.
"""

from __future__ import annotations

from typing import Sequence

# Intensity-group columns that do not carry the ``intensity_`` prefix.
_INTENSITY_NON_PREFIX = ("nonzero_cell_count", "nonzero_fraction", "positive_mean")
# Geometry-group columns that do not carry a geometry prefix.
_GEOMETRY_EXTRA = ("component_count", "contact_signal_sum")


def feature_group_columns(feature_columns: Sequence[str]) -> dict[str, list[str]]:
    """Classify feature columns into ``intensity`` / ``geometry`` / ``zones``.

    Order within each group follows the input order.  A column that matches no
    group raises ``ValueError`` so an unforeseen schema change cannot silently
    drop features from an ablation run.
    """
    groups: dict[str, list[str]] = {"intensity": [], "geometry": [], "zones": []}
    for column in feature_columns:
        if column.startswith("intensity_") or column in _INTENSITY_NON_PREFIX:
            groups["intensity"].append(column)
        elif column.startswith("zone_"):
            groups["zones"].append(column)
        elif (
            column.startswith(("mask_", "bbox_", "centroid_", "cop_", "principal_"))
            or column in _GEOMETRY_EXTRA
        ):
            groups["geometry"].append(column)
        else:
            raise ValueError(
                f"Feature column {column!r} does not belong to any ablation group "
                "(intensity / geometry / zones)"
            )
    return groups
