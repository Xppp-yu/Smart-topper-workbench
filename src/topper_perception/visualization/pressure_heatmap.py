"""Render raw PoPu pressure matrices as two-dimensional heatmaps."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from topper_perception.io.popu import PopuTactilusFrame


def _color_limits(frames: Sequence[PopuTactilusFrame]) -> tuple[float, float]:
    minimum = min(float(np.min(frame.values)) for frame in frames)
    maximum = max(float(np.max(frame.values)) for frame in frames)
    lower = min(0.0, minimum)
    upper = maximum if maximum > lower else lower + 1.0
    return lower, upper


def render_pressure_heatmap(
    frame: PopuTactilusFrame,
    output_path: Path,
    *,
    cmap: str = "magma",
    dpi: int = 180,
) -> Path:
    """Render one raw frame with no smoothing or interpolation."""
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vmin, vmax = _color_limits([frame])

    fig, ax = plt.subplots(figsize=(6.4, 10), constrained_layout=True)
    image = ax.imshow(
        frame.values,
        cmap=cmap,
        origin="upper",
        interpolation="nearest",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    posture = frame.posture if frame.posture is not None else "unlabeled"
    variation = frame.variation if frame.variation is not None else "n/a"
    ax.set_title(
        "PoPu Tactilus pressure map\n"
        f"subject={frame.subject_id} | posture={posture} | variation={variation} | "
        f"frame={frame.snapshot_key} | matrix={frame.rows}×{frame.columns}",
        fontsize=12,
    )
    ax.set_xlabel(f"Sensor column (0–{frame.columns - 1})")
    ax.set_ylabel(f"Sensor row (0–{frame.rows - 1})")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Raw sensor value (dataset units)")
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return output_path


def render_posture_overview(
    frames: Sequence[PopuTactilusFrame],
    output_path: Path,
    *,
    cmap: str = "magma",
    dpi: int = 180,
) -> Path:
    """Render several postures on one shared raw-value color scale."""
    if not frames:
        raise ValueError("At least one frame is required for an overview.")

    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vmin, vmax = _color_limits(frames)
    fig, axes = plt.subplots(
        1,
        len(frames),
        figsize=(3.0 * len(frames), 8.2),
        constrained_layout=True,
        squeeze=False,
    )

    images = []
    for index, (ax, frame) in enumerate(zip(axes[0], frames, strict=True)):
        image = ax.imshow(
            frame.values,
            cmap=cmap,
            origin="upper",
            interpolation="nearest",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )
        images.append(image)
        ax.set_title(frame.posture or "unlabeled")
        ax.set_xlabel("Column")
        if index == 0:
            ax.set_ylabel("Sensor row")
        else:
            ax.set_yticklabels([])

    fig.suptitle(
        f"PoPu Tactilus pressure-map examples | subject={frames[0].subject_id} | "
        f"shared scale | matrix={frames[0].rows}×{frames[0].columns}",
        fontsize=13,
    )
    colorbar = fig.colorbar(images[0], ax=list(axes[0]), fraction=0.025, pad=0.02)
    colorbar.set_label("Raw sensor value (dataset units)")
    fig.savefig(output_path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return output_path

