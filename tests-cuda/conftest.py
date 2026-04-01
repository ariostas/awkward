# BSD 3-Clause License; see https://github.com/scikit-hep/awkward/blob/main/LICENSE

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _warmup_cuda_compute_kernels():
    """
    Pre-compile all cuda.compute JIT kernels once before the test session starts.

    The reducer kernels in awkward._connect.cuda._compute use
    cuda.compute.unary_transform, which JIT-compiles Python closures via Numba
    CUDA on first call.  Each unique (kernel, result_dtype, input_dtype)
    combination triggers a separate compilation that can take several seconds.

    Running every combination here fills the in-process Numba cache before any
    test executes, so tests are not slowed by JIT compilation overhead.
    """
    import cupy as cp

    from awkward._connect.cuda import _compute as cuda_compute

    parents = cp.array([0], dtype=cp.int64)
    offsets = cp.array([0, 1], dtype=cp.int64)
    n = 1  # single-element arrays — enough to trigger JIT without real work

    # ------------------------------------------------------------------
    # awkward_reduce_sum
    # ------------------------------------------------------------------
    for result_t, input_t in [
        (cp.int32, cp.int8),
        (cp.int32, cp.int16),
        (cp.int32, cp.int32),
        (cp.int64, cp.int8),
        (cp.int64, cp.int16),
        (cp.int64, cp.int32),
        (cp.int64, cp.int64),
        (cp.uint32, cp.uint8),
        (cp.uint32, cp.uint16),
        (cp.uint32, cp.uint32),
        (cp.uint64, cp.uint8),
        (cp.uint64, cp.uint16),
        (cp.uint64, cp.uint32),
        (cp.uint64, cp.uint64),
        (cp.float32, cp.float32),
        (cp.float64, cp.float64),
    ]:
        cuda_compute.awkward_reduce_sum(
            cp.zeros(n, result_t), cp.ones(n, input_t), parents, offsets, n, n
        )

    # ------------------------------------------------------------------
    # awkward_reduce_sum_bool  (result = bool_)
    # ------------------------------------------------------------------
    for input_t in [
        cp.int8,
        cp.int16,
        cp.int32,
        cp.int64,
        cp.uint8,
        cp.uint16,
        cp.uint32,
        cp.uint64,
        cp.float32,
        cp.float64,
        cp.bool_,
    ]:
        cuda_compute.awkward_reduce_sum_bool(
            cp.zeros(n, cp.bool_), cp.ones(n, input_t), parents, offsets, n, n
        )

    # ------------------------------------------------------------------
    # awkward_reduce_sum_int32_bool_64 / awkward_reduce_sum_int64_bool_64
    # (both map to the same _compute function; result dtype differs)
    # ------------------------------------------------------------------
    for result_t in (cp.int32, cp.int64):
        cuda_compute.awkward_reduce_sum_int32_bool_64(
            cp.zeros(n, result_t), cp.ones(n, cp.bool_), parents, offsets, n, n
        )

    # ------------------------------------------------------------------
    # awkward_reduce_max / awkward_reduce_min
    # ------------------------------------------------------------------
    for dtype in [
        cp.int8,
        cp.int16,
        cp.int32,
        cp.int64,
        cp.uint8,
        cp.uint16,
        cp.uint32,
        cp.uint64,
        cp.float32,
        cp.float64,
    ]:
        identity = dtype(0)
        cuda_compute.awkward_reduce_max(
            cp.zeros(n, dtype), cp.ones(n, dtype), parents, offsets, n, n, identity
        )
        cuda_compute.awkward_reduce_min(
            cp.zeros(n, dtype), cp.ones(n, dtype), parents, offsets, n, n, identity
        )

    # ------------------------------------------------------------------
    # awkward_reduce_prod
    # ------------------------------------------------------------------
    for result_t, input_t in [
        (cp.int32, cp.int8),
        (cp.int32, cp.int16),
        (cp.int32, cp.int32),
        (cp.int64, cp.int8),
        (cp.int64, cp.int16),
        (cp.int64, cp.int32),
        (cp.int64, cp.int64),
        (cp.uint32, cp.uint8),
        (cp.uint32, cp.uint16),
        (cp.uint32, cp.uint32),
        (cp.uint64, cp.uint8),
        (cp.uint64, cp.uint16),
        (cp.uint64, cp.uint32),
        (cp.uint64, cp.uint64),
        (cp.float32, cp.float32),
        (cp.float64, cp.float64),
    ]:
        cuda_compute.awkward_reduce_prod(
            cp.zeros(n, result_t), cp.ones(n, input_t), parents, offsets, n, n
        )

    # ------------------------------------------------------------------
    # awkward_reduce_prod_bool  (result = bool_)
    # ------------------------------------------------------------------
    for input_t in [
        cp.int8,
        cp.int16,
        cp.int32,
        cp.int64,
        cp.uint8,
        cp.uint16,
        cp.uint32,
        cp.uint64,
        cp.float32,
        cp.float64,
        cp.bool_,
    ]:
        cuda_compute.awkward_reduce_prod_bool(
            cp.zeros(n, cp.bool_), cp.ones(n, input_t), parents, offsets, n, n
        )

    # ------------------------------------------------------------------
    # awkward_reduce_count_64  (no input data, always int64)
    # ------------------------------------------------------------------
    cuda_compute.awkward_reduce_count_64(cp.zeros(n, cp.int64), parents, n, n)

    # ------------------------------------------------------------------
    # awkward_reduce_countnonzero  (result = int64, no offsets arg)
    # ------------------------------------------------------------------
    for input_t in [
        cp.int8,
        cp.int16,
        cp.int32,
        cp.int64,
        cp.uint8,
        cp.uint16,
        cp.uint32,
        cp.uint64,
        cp.float32,
        cp.float64,
        cp.bool_,
    ]:
        cuda_compute.awkward_reduce_countnonzero(
            cp.zeros(n, cp.int64), cp.ones(n, input_t), parents, n, n
        )
