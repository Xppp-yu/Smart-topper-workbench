"""Snapshot-to-record probability aggregation for repeated-measure data.

PoPu JSON records carry 10 highly correlated snapshots.  Reducing per-snapshot
predictions to one record-level prediction by mean probability is generic, so
this module is dataset-agnostic: it takes explicit column names and never parses
PoPu file paths itself.  A record whose snapshots disagree on the true label or
on the subject/group is rejected with a ``ValueError`` rather than silently
resolved.
"""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

# P4a sample-id contract, frozen in ``features/popu.py::extract_row``:
#     sample_id = f"popu-tactilus::{source_relative_path}#frame={snapshot_index}"
# ``source_relative_path`` is the formal record key shared by the 10 snapshots.
_SAMPLE_ID_PATTERN = re.compile(r"^popu-tactilus::(?P<path>.+)#frame=\d+$")


def record_id_from_source_path(source_relative_path: str) -> str:
    """Return the record id for a formal traceability field.

    One PoPu JSON record == one ``source_relative_path``, so the path itself is
    the stable record key; prefer this over string-parsing ``sample_id``.
    """
    return str(source_relative_path)


def record_id_from_sample_id(sample_id: str) -> str:
    """Recover the record id from a P4a ``sample_id`` under its frozen contract.

    Raises ``ValueError`` when the id does not match the P4a contract, so a
    non-conforming id fails loudly instead of being silently truncated.
    """
    match = _SAMPLE_ID_PATTERN.match(sample_id)
    if not match:
        raise ValueError(f"sample_id does not match the P4a contract: {sample_id!r}")
    return match.group("path")


def aggregate_record_predictions(
    predictions: pd.DataFrame,
    *,
    record_id_col: str,
    group_id_col: str,
    y_true_col: str,
    label_columns: Sequence[str],
) -> pd.DataFrame:
    """Aggregate per-snapshot probabilities to one prediction per record.

    ``label_columns`` are probability columns whose names ARE the class labels
    (for example ``"empty"``/``"supine"``).  For each record the snapshot
    probabilities are averaged (mean by default), ``y_pred`` is the argmax class
    and ``confidence`` its mean probability.  A record whose snapshots disagree
    on the true label or the subject/group raises ``ValueError``.
    """
    missing = [
        column
        for column in (record_id_col, group_id_col, y_true_col, *label_columns)
        if column not in predictions.columns
    ]
    if missing:
        raise ValueError(f"Missing columns in predictions: {missing}")
    if not label_columns:
        raise ValueError("label_columns must contain at least one class probability column")

    rows: list[dict[str, object]] = []
    for record_id, group in predictions.groupby(record_id_col, sort=False):
        record_groups = group[group_id_col].unique()
        if len(record_groups) != 1:
            raise ValueError(
                f"record {record_id!r} spans multiple groups: {list(record_groups)}"
            )
        record_labels = group[y_true_col].unique()
        if len(record_labels) != 1:
            raise ValueError(
                f"record {record_id!r} has conflicting labels: {list(record_labels)}"
            )

        means = {column: float(group[column].mean()) for column in label_columns}
        predicted_label = max(label_columns, key=lambda column: means[column])
        row: dict[str, object] = {
            "record_id": str(record_id),
            group_id_col: str(record_groups[0]),
            y_true_col: str(record_labels[0]),
            "y_pred": predicted_label,
            "confidence": means[predicted_label],
            "n_snapshots": int(len(group)),
        }
        row.update({column: means[column] for column in label_columns})
        rows.append(row)

    return pd.DataFrame(rows)
