"""GGUF K-quant tensor helpers for runtime/reference execution.

This module provides reusable containers and reference CPU paths for
Q4_K/Q5_K/Q6_K tensors. It is designed as a foundation for future Triton/WGSL
native K-quant kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class GGUFKQuantTensor:
    """Container for a GGUF K-quantized 2D tensor."""

    name: str
    qtype_name: str
    shape: Tuple[int, ...]
    data_u8: np.ndarray


def _get_dequant_backend():
    import gguf
    from diffusers.quantizers.gguf.utils import GGML_QUANT_SIZES, dequantize_functions

    return gguf, GGML_QUANT_SIZES, dequantize_functions


def make_kquant_tensor(gguf_module, gguf_tensor) -> GGUFKQuantTensor:
    """Build a K-quant container from a GGUF tensor object."""
    qtype = gguf_module.GGMLQuantizationType(int(gguf_tensor.tensor_type))
    qname = qtype.name
    if qname not in {"Q4_K", "Q5_K", "Q6_K"}:
        raise ValueError(f"Tensor is not K-quantized: {gguf_tensor.name} ({qname})")

    data = np.asarray(gguf_tensor.data).astype(np.uint8, copy=False)
    return GGUFKQuantTensor(
        name=gguf_tensor.name,
        qtype_name=qname,
        shape=tuple(gguf_tensor.shape),
        data_u8=np.ascontiguousarray(data),
    )


def dequantize_kquant_tensor(kqt: GGUFKQuantTensor, out_dtype=np.float32) -> np.ndarray:
    """Dequantize K-quant tensor into dense matrix."""
    import torch

    gguf, quant_sizes, dequantize_functions = _get_dequant_backend()
    qtype = gguf.GGMLQuantizationType[kqt.qtype_name]

    block_size, type_size = quant_sizes[qtype]
    blocks_np = kqt.data_u8.reshape(-1, type_size)
    blocks = torch.from_numpy(np.ascontiguousarray(blocks_np).copy())
    deq = dequantize_functions[qtype](
        blocks, block_size, type_size, dtype=torch.float32
    )
    arr = deq.reshape(kqt.shape[0], -1).cpu().numpy()
    if out_dtype is not None:
        arr = arr.astype(out_dtype, copy=False)
    return arr


def matvec_reference(kqt: GGUFKQuantTensor, x: np.ndarray) -> np.ndarray:
    """Reference y = W @ x using dequantized K-quant tensor."""
    w = dequantize_kquant_tensor(kqt, out_dtype=np.float32)
    xv = x.astype(np.float32, copy=False).reshape(-1)
    return w @ xv


def parity_check(kqt: GGUFKQuantTensor, atol: float = 1e-4):
    """Basic parity self-check using two independent code paths.

    Returns a metrics dict.
    """
    import torch

    gguf, quant_sizes, dequantize_functions = _get_dequant_backend()
    qtype = gguf.GGMLQuantizationType[kqt.qtype_name]
    block_size, type_size = quant_sizes[qtype]

    # Path A: module dequant helper
    w_a = dequantize_kquant_tensor(kqt, out_dtype=np.float32)

    # Path B: direct dequant function invocation
    blocks_np = kqt.data_u8.reshape(-1, type_size)
    blocks = torch.from_numpy(np.ascontiguousarray(blocks_np).copy())
    deq_b = dequantize_functions[qtype](
        blocks, block_size, type_size, dtype=torch.float32
    )
    w_b = deq_b.reshape(kqt.shape[0], -1).cpu().numpy().astype(np.float32)

    max_abs = float(np.max(np.abs(w_a - w_b)))
    ok = bool(max_abs <= atol)
    return {
        "ok": ok,
        "qtype": kqt.qtype_name,
        "shape": kqt.shape,
        "max_abs": max_abs,
        "atol": float(atol),
    }


def dequantize_raw_kquant_bytes(
    raw_u8: np.ndarray,
    qtype_name: str,
    N: int,
    out_dtype=np.float16,
) -> np.ndarray:
    """Dequantize raw K-quant bytes (Q4_K/Q5_K/Q6_K) to dense matrix.

    Args:
        raw_u8: (N, bytes_per_row) uint8 array of raw GGUF block data.
        qtype_name: one of 'Q4_K', 'Q5_K', 'Q6_K'.
        N: number of rows (output dimension).
        out_dtype: output dtype.
    Returns:
        (N, K) dense matrix.
    """
    kqt = GGUFKQuantTensor(
        name="__dequant__", qtype_name=qtype_name,
        shape=(N, 0), data_u8=np.ascontiguousarray(raw_u8.ravel()),
    )
    return dequantize_kquant_tensor(kqt, out_dtype=out_dtype)
