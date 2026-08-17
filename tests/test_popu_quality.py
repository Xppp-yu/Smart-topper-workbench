from __future__ import annotations

from topper_perception.quality.popu import RecordMetrics, assess_quality


def _record(*, posture: str, total: float, active: float, cv: float = 0.1) -> RecordMetrics:
    return RecordMetrics(
        {
            "sample_id": f"sample-{posture}-{total}",
            "posture": posture,
            "quality_status": "",
            "quality_reasons": "",
            "median_total_signal": total,
            "median_active_cells": active,
            "temporal_total_cv": cv,
        }
    )


def test_assess_quality_warns_a_clear_per_posture_outlier() -> None:
    records = [
        _record(posture="left", total=100, active=20),
        _record(posture="left", total=101, active=20),
        _record(posture="left", total=99, active=21),
        _record(posture="left", total=500, active=90),
    ]

    assessed, _ = assess_quality(records, robust_z_threshold=4.5)

    assert assessed[-1].as_dict()["quality_status"] == "WARN"
    assert "median_total_signal" in str(assessed[-1].as_dict()["quality_reasons"])


def test_assess_quality_keeps_excluded_rows_excluded() -> None:
    excluded = RecordMetrics(
        {
            "sample_id": "others-1",
            "posture": "",
            "quality_status": "EXCLUDED",
            "quality_reasons": "missing_fixed_posture_label_or_p1_warning",
        }
    )

    assessed, _ = assess_quality([excluded], robust_z_threshold=4.5)

    assert assessed[0].as_dict()["quality_status"] == "EXCLUDED"
