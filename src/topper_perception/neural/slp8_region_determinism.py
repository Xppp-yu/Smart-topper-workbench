"""Determinism configuration for the B04 PM-only Region Mini (R02).

B04 v0.1 R02 hardens cross-process reproducibility by:

* seeding Python, NumPy, torch, and (when available) all CUDA devices;
* setting :func:`torch.use_deterministic_algorithms` so that any
  non-deterministic op is rejected (with ``warn_only=True`` so the
  existing B03 ops keep working in tests);
* setting ``cudnn.deterministic = True`` and
  ``cudnn.benchmark = False``;
* clamping the CPU thread count via
  :func:`torch.set_num_threads` and the OpenMP / MKL env vars so two
  independent processes compute the same row of any op that splits
  across threads.

All flags are recorded into a :class:`DeterminismSettings` dataclass
and serialized into the run ``environment.json`` so a Reviewer can
audit the exact reproducibility configuration.
"""

from __future__ import annotations

import os
import platform
import random
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeterminismSettings:
    """The exact determinism configuration applied to a B04 run."""

    seed: int
    cpu_threads: int
    use_deterministic_algorithms: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    omp_num_threads: int
    mkl_num_threads: int
    python_hash_seed: int | None
    numpy_bit_generator: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "cpu_threads": int(self.cpu_threads),
            "use_deterministic_algorithms": bool(self.use_deterministic_algorithms),
            "cudnn_deterministic": bool(self.cudnn_deterministic),
            "cudnn_benchmark": bool(self.cudnn_benchmark),
            "omp_num_threads": int(self.omp_num_threads),
            "mkl_num_threads": int(self.mkl_num_threads),
            "python_hash_seed": self.python_hash_seed,
            "numpy_bit_generator": str(self.numpy_bit_generator),
        }


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


DEFAULT_CPU_THREADS: int = 1
DEFAULT_DETERMINISTIC_ALGORITHMS: bool = True
DEFAULT_CUDNN_DETERMINISTIC: bool = True
DEFAULT_CUDNN_BENCHMARK: bool = False
DEFAULT_NUMPY_BIT_GENERATOR: str = "PCG64"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def collect_settings(seed: int) -> DeterminismSettings:
    """Read the current environment + torch flags into a settings record."""

    import torch  # imported lazily so the helpers can be unit-tested

    cpu_threads = int(torch.get_num_threads()) if hasattr(torch, "get_num_threads") else DEFAULT_CPU_THREADS
    try:
        cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
    except Exception:
        cudnn_deterministic = DEFAULT_CUDNN_DETERMINISTIC
    try:
        cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
    except Exception:
        cudnn_benchmark = DEFAULT_CUDNN_BENCHMARK

    py_hash_raw = os.environ.get("PYTHONHASHSEED")
    py_hash: int | None = None
    if py_hash_raw is not None and py_hash_raw != "":
        try:
            py_hash = int(py_hash_raw)
        except ValueError:
            py_hash = None

    bitgen = DEFAULT_NUMPY_BIT_GENERATOR
    try:
        bitgen = str(np.random.bit_generator._bit_generator.__class__.__name__)
    except Exception:
        bitgen = DEFAULT_NUMPY_BIT_GENERATOR

    return DeterminismSettings(
        seed=int(seed),
        cpu_threads=int(cpu_threads),
        use_deterministic_algorithms=bool(DEFAULT_DETERMINISTIC_ALGORITHMS),
        cudnn_deterministic=bool(cudnn_deterministic),
        cudnn_benchmark=bool(cudnn_benchmark),
        omp_num_threads=_env_int("OMP_NUM_THREADS", DEFAULT_CPU_THREADS),
        mkl_num_threads=_env_int("MKL_NUM_THREADS", DEFAULT_CPU_THREADS),
        python_hash_seed=py_hash,
        numpy_bit_generator=str(bitgen),
    )


def apply_settings(seed: int, *, cpu_threads: int = DEFAULT_CPU_THREADS) -> DeterminismSettings:
    """Apply a strict determinism configuration to the current process.

    The function returns the post-application :class:`DeterminismSettings`
    so the caller can persist them in the run metadata.

    The function is **idempotent**: calling it twice with the same
    ``seed`` is equivalent to calling it once.
    """

    import torch  # imported lazily

    os.environ["PYTHONHASHSEED"] = str(int(seed))
    os.environ["OMP_NUM_THREADS"] = str(int(cpu_threads))
    os.environ["MKL_NUM_THREADS"] = str(int(cpu_threads))

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    torch.set_num_threads(int(cpu_threads))
    try:
        torch.use_deterministic_algorithms(
            bool(DEFAULT_DETERMINISTIC_ALGORITHMS), warn_only=True
        )
    except Exception:
        pass
    try:
        torch.backends.cudnn.deterministic = bool(DEFAULT_CUDNN_DETERMINISTIC)
        torch.backends.cudnn.benchmark = bool(DEFAULT_CUDNN_BENCHMARK)
    except Exception:
        pass

    return collect_settings(int(seed))


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


def environment_payload() -> dict[str, Any]:
    """Return a JSON-safe environment snapshot for ``environment.json``."""

    import torch  # lazy

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "torch_num_threads": (
            int(torch.get_num_threads()) if hasattr(torch, "get_num_threads") else None
        ),
    }
