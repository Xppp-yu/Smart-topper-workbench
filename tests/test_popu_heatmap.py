from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)

from topper_perception.io.popu import (  # noqa: E402
    load_tactilus_record,
    select_tactilus_frame,
)
from topper_perception.visualization import render_pressure_heatmap  # noqa: E402


def _write_record(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "volunteer_id": "1",
                "position": "left",
                "variation": "1",
                "tactilus_rows": 2,
                "tactilus_columns": 3,
                "snapshots": {
                    "1": {"id": "later", "tactilus_readings": [6, 7, 8, 9, 10, 11]},
                    "0": {"id": "first", "tactilus_readings": [0, 1, 2, 3, 4, 5]},
                },
            }
        ),
        encoding="utf-8",
    )


def test_load_record_restores_rows_and_columns(tmp_path: Path) -> None:
    source = tmp_path / "left1_0.json"
    _write_record(source)

    frames = load_tactilus_record(source)

    assert [frame.snapshot_key for frame in frames] == ["0", "1"]
    assert frames[0].values.shape == (2, 3)
    np.testing.assert_array_equal(frames[0].values, [[0, 1, 2], [3, 4, 5]])


def test_select_frame_filters_subject_posture_and_variation(tmp_path: Path) -> None:
    subject_dir = tmp_path / "tactilus_data" / "1"
    subject_dir.mkdir(parents=True)
    _write_record(subject_dir / "left1_0.json")

    frame = select_tactilus_frame(
        tmp_path,
        subject_id=1,
        posture="left",
        variation=1,
        record_index=0,
        frame_index=1,
    )

    assert frame.snapshot_id == "later"
    assert frame.posture == "left"
    assert frame.values.shape == (2, 3)


def test_render_heatmap_writes_png(tmp_path: Path) -> None:
    source = tmp_path / "left1_0.json"
    _write_record(source)
    frame = load_tactilus_record(source)[0]
    output = tmp_path / "heatmap.png"

    render_pressure_heatmap(frame, output)

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

