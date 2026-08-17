from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "compare_popu_mask_strategies.py"
)
SPEC = importlib.util.spec_from_file_location("compare_popu_mask_strategies", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
_select_representative_samples = MODULE._select_representative_samples


def _strategy_rows(
    sample_id: str,
    *,
    p2_status: str = "ACCEPT",
    comparison_statuses: tuple[str, str, str] = ("OK", "OK", "OK"),
    bbox_fractions: tuple[float, float, float] = (0.20, 0.21, 0.22),
) -> list[dict[str, object]]:
    return [
        {
            "sample_id": sample_id,
            "p2_quality_status": p2_status,
            "comparison_status": comparison_status,
            "median_bbox_area_fraction": bbox_fraction,
        }
        for comparison_status, bbox_fraction in zip(
            comparison_statuses, bbox_fractions, strict=True
        )
    ]


def test_representative_sample_categories_are_mutually_exclusive() -> None:
    rows = [
        *_strategy_rows("accept"),
        *_strategy_rows("warn", p2_status="WARN"),
        *_strategy_rows("spread", bbox_fractions=(0.10, 0.20, 0.30)),
        *_strategy_rows(
            "partial",
            comparison_statuses=("OK", "WARN", "OK"),
        ),
    ]
    overlay = {
        "accept_count": 3,
        "warn_count": 3,
        "divergence_count": 3,
        "divergence_bbox_fraction_spread": 0.05,
    }

    selection = _select_representative_samples(rows, overlay)

    assert selection == {
        "accept": ["accept"],
        "warn": ["warn"],
        "divergence": ["partial", "spread"],
    }
    selected_sets = [set(selection[category]) for category in selection]
    assert not (selected_sets[0] & selected_sets[1])
    assert not (selected_sets[0] & selected_sets[2])
    assert not (selected_sets[1] & selected_sets[2])
