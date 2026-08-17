from __future__ import annotations

import numpy as np

from topper_perception.geometry.popu import build_contact_mask, describe_geometry


def test_mask_removes_tiny_islands_and_keeps_contact_cluster() -> None:
    values = np.zeros((6, 6), dtype=float)
    values[2:4, 2:5] = 10
    values[0, 0] = 10

    mask, threshold, components = build_contact_mask(values, positive_percentile=10, minimum_component_cells=3)

    assert threshold == 10
    assert components == 1
    assert mask.sum() == 6
    assert mask[0, 0] is np.False_


def test_geometry_reports_bbox_and_centre_of_pressure() -> None:
    values = np.zeros((5, 5), dtype=float)
    values[1, 1] = 2
    values[1, 2] = 4
    values[2, 1] = 4
    values[2, 2] = 8

    geometry, mask = describe_geometry(values, positive_percentile=0, minimum_component_cells=1)

    assert geometry["geometry_status"] == "OK"
    assert geometry["bbox_height"] == 2
    assert geometry["bbox_width"] == 2
    assert geometry["cop_row"] > 1.5
    assert geometry["cop_column"] > 1.5
    assert mask.sum() == 4


def test_geometry_can_use_frozen_largest_component_strategy() -> None:
    values = np.zeros((7, 7), dtype=float)
    values[1:4, 1:4] = 10
    values[5, 5] = 10

    geometry, mask = describe_geometry(
        values,
        strategy="largest_component",
        positive_percentile=10,
        minimum_component_cells=1,
    )

    assert geometry["mask_strategy"] == "largest_component"
    assert geometry["component_count"] == 1
    assert geometry["bbox_height"] == 3
    assert geometry["bbox_width"] == 3
    assert mask.sum() == 9


def test_single_cell_geometry_warns_instead_of_emitting_nan_axis() -> None:
    values = np.zeros((4, 4), dtype=float)
    values[2, 1] = 10

    geometry, mask = describe_geometry(
        values,
        strategy="largest_component",
        positive_percentile=10,
        minimum_component_cells=1,
    )

    assert mask.sum() == 1
    assert geometry["geometry_status"] == "WARN"
    assert geometry["geometry_reason"] == "insufficient_cells_for_principal_axis"
    assert geometry["principal_axis_degrees"] == ""
    assert geometry["principal_axis_anisotropy"] == ""
