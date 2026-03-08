"""Shared WGSL kernels — used by all model formats.

Loads WGSL source from compiler/kernels/shared/*.wgsl.
These are the canonical kernel definitions shared by all runtimes.
"""

import os

_KERNEL_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'compiler', 'kernels', 'shared')


def _load_wgsl(name: str) -> str:
    """Load a .wgsl kernel from compiler/kernels/shared/."""
    path = os.path.join(_KERNEL_DIR, name + '.wgsl')
    with open(path) as f:
        return f.read()


# Lazy-loaded kernel sources
def get_shared_kernel(name: str) -> str:
    """Get WGSL source for a shared kernel by name."""
    return _load_wgsl(name)


# Re-export from the legacy Python kernel module for backward compat
from common.wgsl_kernels import (
    WGSL_GQA_CHUNKED_PASS1,
    WGSL_GQA_CHUNKED_PASS2,
    GQA_CHUNKED_PASS1_BINDINGS,
    GQA_CHUNKED_PASS2_BINDINGS,
    GQA_CHUNK_SIZE,
    pack_gqa_chunked_params,
    WGSL_FP16_GEMM_KERNEL,
    FP16_GEMM_BINDINGS,
    pack_fp16_gemm_params,
)
