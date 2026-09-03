"""CPU-only, one-batch smoke for the B11F real production wiring."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TASK_ID = "TASK-SLP-B11F-PRODUCTION-WIRING-SMOKE-v0.1"
SMOKE_ID = "SMOKE-SLP-B11F-PRODUCTION-WIRING-20260904-R07"
SMOKE_BATCH_SIZE = 1
STATISTICS_SAMPLE_COUNT = 128
CONFIG = ROOT / "configs/experiments/slp8_pm_final_development_fit_v0.1.json"


class SmokeError(RuntimeError):
    """Raised when the smoke contract is violated."""


def _atomic_write_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise SmokeError(f"refusing to overwrite existing smoke output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # The temporary file is complete and fsynced. Publishing it with a
            # hard link is atomic and, unlike a replacing rename, never clobbers a
            # target created by a competing process after the initial check.
            os.link(temporary, path)
        except FileExistsError as exc:
            raise SmokeError(f"refusing to overwrite existing smoke output: {path}") from exc
        os.unlink(temporary)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def run_smoke(*, b01_freeze_dir: Path, dataset_root: Path) -> dict[str, Any]:
    # Hide every CUDA device before importing torch or production modules. This
    # process must prove CPU wiring only and must not initialize a GPU context.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    import numpy as np
    import torch
    from torch import optim

    from topper_perception.neural.slp8_region_dataset import Slp8RegionDataset, build_dataloader
    from topper_perception.neural.slp8_region_final_fit import (
        BATCH_SIZE,
        LEARNING_RATE,
        MODEL,
        OPTIMIZER,
        WEIGHT_DECAY,
        class_weights_to_tensor,
        load_development_samples,
        load_protocol,
        sha256_file,
    )
    from topper_perception.neural.slp8_region_full import (
        build_model,
        compute_fold_class_weights_from_samples,
        compute_fold_normalization_from_samples,
        deterministic_cross_entropy_2d,
    )
    from topper_perception.neural.slp8_region_determinism import apply_settings

    if torch.cuda.is_available():
        raise SmokeError("CUDA remained visible after CPU-only isolation")
    if not b01_freeze_dir.is_dir() or not dataset_root.is_dir():
        raise SmokeError("B01 freeze directory or dataset root is missing")

    print("stage=protocol", flush=True)
    protocol = load_protocol(CONFIG, ROOT)
    if protocol.optimizer != OPTIMIZER or OPTIMIZER != "AdamW":
        raise SmokeError("optimizer contract drift")
    settings = apply_settings(42)
    print("stage=development_loader", flush=True)
    samples = load_development_samples(b01_freeze_dir)
    if len(samples) != 4095 or len({sample.subject_id for sample in samples}) != 91:
        raise SmokeError("development pool cardinality drift")
    if any(sample.ml_split not in {"train", "val"} for sample in samples):
        raise SmokeError("non-development sample entered smoke")

    statistics_samples = samples[:STATISTICS_SAMPLE_COUNT]
    print("stage=smoke_subset_normalization", flush=True)
    normalization = compute_fold_normalization_from_samples(statistics_samples, data_root=dataset_root)
    print("stage=smoke_subset_class_weights", flush=True)
    class_weights = compute_fold_class_weights_from_samples(statistics_samples, data_root=dataset_root)
    weights_numpy = class_weights_to_tensor(class_weights)
    if not isinstance(weights_numpy, np.ndarray):
        raise SmokeError("class_weights_to_tensor did not return numpy.ndarray")
    if weights_numpy.shape != (9,) or weights_numpy.dtype != np.float64:
        raise SmokeError("class weight NumPy shape/dtype drift")
    if not np.isfinite(weights_numpy).all():
        raise SmokeError("class weight NumPy vector is non-finite")
    expected_weights = np.asarray([class_weights.weights[class_id] for class_id in range(9)])
    if not np.array_equal(weights_numpy, expected_weights):
        raise SmokeError("class weight order is not class ID 0..8")
    weight_tensor = torch.from_numpy(weights_numpy).to("cpu").to(torch.float32)
    if weight_tensor.device.type != "cpu" or weight_tensor.dtype != torch.float32:
        raise SmokeError("class weight Torch device/dtype drift")

    print("stage=real_microbatch", flush=True)
    dataset = Slp8RegionDataset(samples, dataset_root, normalization)
    loader = build_dataloader(dataset, batch_size=SMOKE_BATCH_SIZE, shuffle=True, drop_last=False)
    model = build_model(MODEL, "cpu")
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    before = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}

    batch = next(iter(loader))
    pressure = batch["pressure"].to("cpu")
    labels = batch["label"].to("cpu")
    if tuple(pressure.shape) != (SMOKE_BATCH_SIZE, 1, 192, 84) or pressure.dtype != torch.float32:
        raise SmokeError(f"real pressure batch shape/dtype drift: {tuple(pressure.shape)} / {pressure.dtype}")
    if tuple(labels.shape) != (SMOKE_BATCH_SIZE, 192, 84) or labels.dtype != torch.int64:
        raise SmokeError(f"real label batch shape/dtype drift: {tuple(labels.shape)} / {labels.dtype}")
    if set(batch["ml_split"]) - {"train", "val"}:
        raise SmokeError("forbidden split entered real batch")

    print("stage=forward_backward_step", flush=True)
    optimizer.zero_grad(set_to_none=True)
    logits = model(pressure)
    loss = deterministic_cross_entropy_2d(logits, labels, weight=weight_tensor)
    if not math.isfinite(float(loss.detach().item())):
        raise SmokeError("one-batch loss is non-finite")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
    if not gradients or any(not torch.isfinite(gradient).all().item() for gradient in gradients):
        raise SmokeError("missing or non-finite gradients")
    optimizer.step()
    changed_names = [
        name for name, value in model.named_parameters()
        if value.requires_grad and not torch.equal(before[name], value.detach())
    ]
    if not changed_names:
        raise SmokeError("one AdamW step changed no trainable parameter")

    freeze_manifest = b01_freeze_dir / "freeze_manifest.json"
    return {
        "task_id": TASK_ID,
        "smoke_id": SMOKE_ID,
        "result": "PASS",
        "scope": "LOCAL_CPU_REAL_DATA_ONE_BATCH_ONLY",
        "formal_experiment_id": None,
        "checkpoint_created": False,
        "resume_used": False,
        "autodl_connected": False,
        "gpu_training_run": False,
        "device": "cpu",
        "cuda_visible_devices": "-1",
        "config_sha256": protocol.sha256,
        "candidate_config_sha256": sha256_file(protocol.candidate_contract),
        "data_manifest_sha256": sha256_file(freeze_manifest),
        "development_subjects": 91,
        "development_samples": 4095,
        "development_splits": ["train", "val"],
        "statistics_scope": "DETERMINISTIC_REAL_DEVELOPMENT_SUBSET_NOT_PRODUCTION_ESTIMATE",
        "statistics_samples": STATISTICS_SAMPLE_COUNT,
        "test_access": False,
        "test_rows": 0,
        "test_labels": 0,
        "test_onehot": 0,
        "model": MODEL,
        "seed": 42,
        "production_batch_size": BATCH_SIZE,
        "smoke_microbatch_size": SMOKE_BATCH_SIZE,
        "optimizer": OPTIMIZER,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "optimizer_steps": 1,
        "class_weight_order": list(range(9)),
        "class_weight_numpy_type": type(weights_numpy).__name__,
        "class_weight_numpy_dtype": str(weights_numpy.dtype),
        "class_weight_tensor_dtype": str(weight_tensor.dtype),
        "class_weight_tensor_device": weight_tensor.device.type,
        "class_weight_shape": list(weight_tensor.shape),
        "pressure_shape": list(pressure.shape),
        "pressure_dtype": str(pressure.dtype),
        "label_shape": list(labels.shape),
        "label_dtype": str(labels.dtype),
        "logits_shape": list(logits.shape),
        "loss_diagnostic_not_validation_metric": float(loss.detach().item()),
        "gradient_tensor_count": len(gradients),
        "changed_parameter_tensor_count": len(changed_names),
        "parameters_changed": True,
        "determinism": settings.as_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b01-freeze-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = args.output_json.resolve()
        if output.exists():
            raise SmokeError(f"refusing to overwrite existing smoke output: {output}")
        summary = run_smoke(
            b01_freeze_dir=args.b01_freeze_dir.resolve(),
            dataset_root=args.dataset_root.resolve(),
        )
        _atomic_write_new(output, summary)
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
        print("B11F_PRODUCTION_WIRING_SMOKE_PASSED TEST=0 GPU_NOT_RUN AUTODL_NOT_CONNECTED")
        return 0
    except Exception as exc:
        print(f"ERR: {exc}")
        print("B11F_PRODUCTION_WIRING_SMOKE_FAILED TEST=0 GPU_NOT_RUN AUTODL_NOT_CONNECTED")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
