"""Determinism configuration for the B04 PM-only Region Mini (R03).

B04 v0.1 R03 hardens cross-process reproducibility by:

* seeding Python, NumPy, torch, and (when available) all CUDA devices;
* setting :func:`torch.use_deterministic_algorithms` so that any
  non-deterministic op is rejected — ``warn_only=False`` when CUDA is
  available (fail-closed) and ``warn_only=True`` on CPU so the
  synthetic CPU smoke can still complete on PyTorch versions whose
  CPU kernels are not strictly deterministic;
* setting ``cudnn.deterministic = True`` and
  ``cudnn.benchmark = False``;
* exporting ``CUBLAS_WORKSPACE_CONFIG = ":4096:8"`` and
  ``CUBLASLT_WORKSPACE_CONFIG`` **before** any CUDA tensor is created
  so cuBLAS / cuBLASLt pick the deterministic workspace;
* clamping the CPU thread count via
  :func:`torch.set_num_threads` and the OpenMP / MKL env vars so two
  independent processes compute the same row of any op that splits
  across threads.

All flags are recorded into a :class:`DeterminismSettings` dataclass
and serialized into the run ``environment.json`` and
``budget_report.json`` so a Reviewer can audit the exact
reproducibility configuration.  R03 also records the resulting
``run_mode`` (``"cpu_synthetic_reproducible"`` or
``"cuda_determinism_unverified"``) because CUDA determinism is still
**NOT RUN**: the current experiment has no CUDA device, so we can only
*declare* the configuration, not *verify* the contract end-to-end.
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
    deterministic_algorithms_warn_only: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cublas_workspace_config: str
    cublaslt_workspace_config: str
    omp_num_threads: int
    mkl_num_threads: int
    python_hash_seed: int | None
    numpy_bit_generator: str
    run_mode: str  # "cpu_synthetic_reproducible" | "cuda_determinism_unverified"

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "cpu_threads": int(self.cpu_threads),
            "use_deterministic_algorithms": bool(self.use_deterministic_algorithms),
            "deterministic_algorithms_warn_only": bool(self.deterministic_algorithms_warn_only),
            "cudnn_deterministic": bool(self.cudnn_deterministic),
            "cudnn_benchmark": bool(self.cudnn_benchmark),
            "cublas_workspace_config": str(self.cublas_workspace_config),
            "cublaslt_workspace_config": str(self.cublaslt_workspace_config),
            "omp_num_threads": int(self.omp_num_threads),
            "mkl_num_threads": int(self.mkl_num_threads),
            "python_hash_seed": self.python_hash_seed,
            "numpy_bit_generator": str(self.numpy_bit_generator),
            "run_mode": str(self.run_mode),
        }


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


DEFAULT_CPU_THREADS: int = 1
DEFAULT_DETERMINISTIC_ALGORITHMS: bool = True
DEFAULT_CUDNN_DETERMINISTIC: bool = True
DEFAULT_CUDNN_BENCHMARK: bool = False
DEFAULT_NUMPY_BIT_GENERATOR: str = "PCG64"

# cuBLAS / cuBLASLt deterministic workspace size.  Must be set BEFORE
# any CUDA tensor is created (i.e. before the first torch.cuda call).
# 4096 bytes per dimension is the smallest cuBLAS-recommended value.
DEFAULT_CUBLAS_WORKSPACE_CONFIG: str = ":4096:8"


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


def _export_cublas_workspace_configs() -> tuple[str, str]:
    """Set the cuBLAS / cuBLASLt deterministic workspace env vars.

    The values are *also* returned so the caller can persist them in
    :class:`DeterminismSettings`.  Exporting must happen **before** any
    CUDA tensor is created so the libraries read the env at init time.
    """

    cublas_value = os.environ.get("CUBLAS_WORKSPACE_CONFIG") or DEFAULT_CUBLAS_WORKSPACE_CONFIG
    cublaslt_value = os.environ.get("CUBLASLT_WORKSPACE_CONFIG") or DEFAULT_CUBLAS_WORKSPACE_CONFIG
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(cublas_value)
    os.environ["CUBLASLT_WORKSPACE_CONFIG"] = str(cublaslt_value)
    return str(cublas_value), str(cublaslt_value)


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

    cublas_value = os.environ.get("CUBLAS_WORKSPACE_CONFIG") or DEFAULT_CUBLAS_WORKSPACE_CONFIG
    cublaslt_value = os.environ.get("CUBLASLT_WORKSPACE_CONFIG") or DEFAULT_CUBLAS_WORKSPACE_CONFIG
    cuda_available = bool(torch.cuda.is_available())
    run_mode = (
        "cuda_determinism_unverified"
        if cuda_available
        else "cpu_synthetic_reproducible"
    )

    return DeterminismSettings(
        seed=int(seed),
        cpu_threads=int(cpu_threads),
        use_deterministic_algorithms=bool(DEFAULT_DETERMINISTIC_ALGORITHMS),
        deterministic_algorithms_warn_only=bool(not cuda_available),
        cudnn_deterministic=bool(cudnn_deterministic),
        cudnn_benchmark=bool(cudnn_benchmark),
        cublas_workspace_config=str(cublas_value),
        cublaslt_workspace_config=str(cublaslt_value),
        omp_num_threads=_env_int("OMP_NUM_THREADS", DEFAULT_CPU_THREADS),
        mkl_num_threads=_env_int("MKL_NUM_THREADS", DEFAULT_CPU_THREADS),
        python_hash_seed=py_hash,
        numpy_bit_generator=str(bitgen),
        run_mode=run_mode,
    )


def apply_settings(seed: int, *, cpu_threads: int = DEFAULT_CPU_THREADS) -> DeterminismSettings:
    """Apply a strict determinism configuration to the current process.

    CUDA determinism: when a CUDA device is available, any
    non-deterministic op causes a hard ``RuntimeError`` (no
    ``warn_only``).  On CPU we accept ``warn_only=True`` so the
    existing CPU kernels can complete; the test
    ``TestDeterminism::test_apply_settings_pins_env`` verifies the
    configuration regardless.

    The function returns the post-application :class:`DeterminismSettings`
    so the caller can persist them in the run metadata.

    The function is **idempotent**: calling it twice with the same
    ``seed`` is equivalent to calling it once.
    """

    import torch  # imported lazily

    # 1) cuBLAS / cuBLASLt workspace env vars MUST be exported before
    #    any CUDA tensor is created.
    cublas_value, cublaslt_value = _export_cublas_workspace_configs()

    # 2) Seed env vars.
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    os.environ["OMP_NUM_THREADS"] = str(int(cpu_threads))
    os.environ["MKL_NUM_THREADS"] = str(int(cpu_threads))

    # 3) Seed Python / NumPy / torch.
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    # 4) Threading + deterministic algorithms.
    torch.set_num_threads(int(cpu_threads))
    cuda_available = bool(torch.cuda.is_available())
    warn_only = bool(not cuda_available)  # CUDA: fail-closed
    try:
        torch.use_deterministic_algorithms(
            bool(DEFAULT_DETERMINISTIC_ALGORITHMS), warn_only=warn_only
        )
    except Exception as exc:
        if cuda_available:
            raise RuntimeError(
                "CUDA run cannot enable deterministic PyTorch algorithms"
            ) from exc
    try:
        torch.backends.cudnn.deterministic = bool(DEFAULT_CUDNN_DETERMINISTIC)
        torch.backends.cudnn.benchmark = bool(DEFAULT_CUDNN_BENCHMARK)
    except Exception as exc:
        if cuda_available:
            raise RuntimeError(
                "CUDA run cannot configure deterministic cuDNN settings"
            ) from exc

    return collect_settings(int(seed))


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


def environment_payload() -> dict[str, Any]:
    """Return a JSON-safe environment snapshot for ``environment.json``."""

    import torch  # lazy

    cuda_available = bool(torch.cuda.is_available())
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "torch_num_threads": (
            int(torch.get_num_threads()) if hasattr(torch, "get_num_threads") else None
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cublaslt_workspace_config": os.environ.get("CUBLASLT_WORKSPACE_CONFIG"),
        "run_mode": (
            "cuda_determinism_unverified"
            if cuda_available
            else "cpu_synthetic_reproducible"
        ),
    }
