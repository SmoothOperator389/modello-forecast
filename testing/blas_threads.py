"""Limita i thread nativi (OpenBLAS/NumPy) prima di importare numpy.

Su Mac, OpenBLAS multi-thread può fare SIGSEGV (gemm_thread_n) se lo stack
è piccolo — tipico quando il processo parte da Cursor. 1 thread è stabile.
"""

from __future__ import annotations

import os

_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def limit_native_threads(n: int = 1) -> None:
    val = str(n)
    for key in _THREAD_ENV:
        os.environ.setdefault(key, val)
