"""Shared GGUF utilities for Backpack model converters.

These helpers centralize GGUF reading/dequant behavior so model-specific
converters can focus on tensor name mapping.
"""

from __future__ import annotations

import mmap
import os
import struct
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Fast mmap-based GGUF parser (llama.cpp style)
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_DEFAULT_ALIGNMENT = 32

# GGML quantization type → (block_size_elements, block_size_bytes)
GGML_BLOCK_SIZES = {
    0:  (1,  4),     # F32
    1:  (1,  2),     # F16
    2:  (32, 18),    # Q4_0
    3:  (32, 20),    # Q4_1
    6:  (32, 22),    # Q5_0
    7:  (32, 24),    # Q5_1
    8:  (32, 34),    # Q8_0
    9:  (32, 36),    # Q8_1
    10: (256, 84),   # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    28: (32,  18),   # IQ4_NL
    30: (256, 36),   # TQ1_0
    31: (256, 66),   # TQ2_0
}

GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
    18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
    22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 31: "TQ1_0", 32: "TQ2_0",
}

# GGUF metadata value type sizes (fixed-size types only)
_KV_FIXED_SIZES = {
    0: 1,   # UINT8
    1: 1,   # INT8
    2: 2,   # UINT16
    3: 2,   # INT16
    4: 4,   # UINT32
    5: 4,   # INT32
    6: 4,   # FLOAT32
    7: 1,   # BOOL
    # 8: STRING — variable
    # 9: ARRAY — variable
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}


@dataclass
class GGUFTensorInfo:
    """Lightweight tensor descriptor from GGUF header."""
    name: str
    shape: Tuple[int, ...]
    qtype: int       # GGML quantization type enum
    qtype_name: str
    offset: int      # byte offset relative to data section start
    data_size: int   # total bytes of tensor data


class GGUFFile:
    """Fast mmap-based GGUF file reader.

    Like llama.cpp, this mmaps the entire file and parses only the
    header + tensor info entries.  Metadata KV pairs are skipped
    without creating Python objects.  Tensor data is accessed as
    zero-copy mmap views.
    """

    def __init__(self, path: str):
        self.path = path
        self._file = open(path, 'rb')
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._parse()

    def _read_u32(self, pos: int) -> Tuple[int, int]:
        return struct.unpack_from('<I', self._mm, pos)[0], pos + 4

    def _read_u64(self, pos: int) -> Tuple[int, int]:
        return struct.unpack_from('<Q', self._mm, pos)[0], pos + 8

    def _read_i64(self, pos: int) -> Tuple[int, int]:
        return struct.unpack_from('<q', self._mm, pos)[0], pos + 8

    def _read_string(self, pos: int) -> Tuple[str, int]:
        slen, pos = self._read_u64(pos)
        s = self._mm[pos:pos + slen].decode('utf-8')
        return s, pos + slen

    def _skip_kv_value(self, vtype: int, pos: int) -> int:
        """Advance past a metadata value without creating Python objects."""
        if vtype in _KV_FIXED_SIZES:
            return pos + _KV_FIXED_SIZES[vtype]
        if vtype == 8:  # STRING
            slen, pos = self._read_u64(pos)
            return pos + slen
        if vtype == 9:  # ARRAY
            elem_type, pos = self._read_u32(pos)
            count, pos = self._read_u64(pos)
            if elem_type in _KV_FIXED_SIZES:
                return pos + count * _KV_FIXED_SIZES[elem_type]
            # Array of strings or nested arrays
            for _ in range(count):
                pos = self._skip_kv_value(elem_type, pos)
            return pos
        raise ValueError(f"Unknown GGUF KV value type: {vtype}")

    def _read_kv_value(self, vtype: int, pos: int):
        """Read a metadata value and return (value, new_pos)."""
        if vtype == 0:   return self._mm[pos], pos + 1                   # UINT8
        if vtype == 1:   return struct.unpack_from('<b', self._mm, pos)[0], pos + 1  # INT8
        if vtype == 2:   return struct.unpack_from('<H', self._mm, pos)[0], pos + 2  # UINT16
        if vtype == 3:   return struct.unpack_from('<h', self._mm, pos)[0], pos + 2  # INT16
        if vtype == 4:   return struct.unpack_from('<I', self._mm, pos)[0], pos + 4  # UINT32
        if vtype == 5:   return struct.unpack_from('<i', self._mm, pos)[0], pos + 4  # INT32
        if vtype == 6:   return struct.unpack_from('<f', self._mm, pos)[0], pos + 4  # FLOAT32
        if vtype == 7:   return bool(self._mm[pos]), pos + 1                         # BOOL
        if vtype == 8:   return self._read_string(pos)                               # STRING
        if vtype == 10:  return struct.unpack_from('<Q', self._mm, pos)[0], pos + 8  # UINT64
        if vtype == 11:  return struct.unpack_from('<q', self._mm, pos)[0], pos + 8  # INT64
        if vtype == 12:  return struct.unpack_from('<d', self._mm, pos)[0], pos + 8  # FLOAT64
        if vtype == 9:   # ARRAY
            elem_type, pos = self._read_u32(pos)
            count, pos = self._read_u64(pos)
            if elem_type == 8:  # string array
                arr = []
                for _ in range(count):
                    s, pos = self._read_string(pos)
                    arr.append(s)
                return arr, pos
            if elem_type in _KV_FIXED_SIZES:
                sz = _KV_FIXED_SIZES[elem_type]
                # Skip fixed-size arrays efficiently
                pos_end = pos + count * sz
                return None, pos_end  # don't materialize large int arrays
            for _ in range(count):
                _, pos = self._read_kv_value(elem_type, pos)
            return None, pos
        raise ValueError(f"Unknown GGUF KV value type: {vtype}")

    def _parse(self):
        mm = self._mm
        # Header
        magic, pos = self._read_u32(0)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file: magic=0x{magic:08x}")
        version, pos = self._read_u32(pos)
        if version < 2:
            raise ValueError(f"Unsupported GGUF version: {version}")
        n_tensors, pos = self._read_u64(pos)
        n_kv, pos = self._read_u64(pos)

        # Parse metadata KV pairs
        self.metadata: Dict[str, any] = {}
        for _ in range(n_kv):
            key, pos = self._read_string(pos)
            vtype, pos = self._read_u32(pos)
            val, pos = self._read_kv_value(vtype, pos)
            if val is not None:
                self.metadata[key] = val

        # Parse tensor info entries
        self.tensors: Dict[str, GGUFTensorInfo] = {}
        tensor_infos: List[GGUFTensorInfo] = []
        for _ in range(n_tensors):
            name, pos = self._read_string(pos)
            n_dims, pos = self._read_u32(pos)
            shape = []
            for _ in range(n_dims):
                dim, pos = self._read_u64(pos)
                shape.append(int(dim))
            qtype, pos = self._read_u32(pos)
            offset, pos = self._read_u64(pos)

            # Compute data size
            n_elements = 1
            for d in shape:
                n_elements *= d
            if qtype in GGML_BLOCK_SIZES:
                blk_elems, blk_bytes = GGML_BLOCK_SIZES[qtype]
                n_blocks = (n_elements + blk_elems - 1) // blk_elems
                data_size = n_blocks * blk_bytes
            else:
                # Unknown type — estimate from element count
                data_size = n_elements * 4  # assume f32

            qtype_name = GGML_TYPE_NAMES.get(qtype, f"TYPE_{qtype}")
            info = GGUFTensorInfo(
                name=name, shape=tuple(shape), qtype=qtype,
                qtype_name=qtype_name, offset=offset,
                data_size=data_size)
            tensor_infos.append(info)
            self.tensors[name] = info

        # Data section starts after tensor infos, aligned
        self._data_offset = ((pos + GGUF_DEFAULT_ALIGNMENT - 1)
                             // GGUF_DEFAULT_ALIGNMENT
                             * GGUF_DEFAULT_ALIGNMENT)

    def tensor_data(self, name: str) -> np.ndarray:
        """Return raw tensor data as a uint8 numpy view (zero-copy)."""
        info = self.tensors[name]
        start = self._data_offset + info.offset
        end = start + info.data_size
        return np.frombuffer(self._mm, dtype=np.uint8,
                             count=info.data_size, offset=start)

    def tensor_data_f32(self, name: str) -> np.ndarray:
        """Return F32 tensor data as float32 numpy view (zero-copy)."""
        info = self.tensors[name]
        assert info.qtype == 0, f"Expected F32, got {info.qtype_name}"
        start = self._data_offset + info.offset
        n_elements = 1
        for d in info.shape:
            n_elements *= d
        return np.frombuffer(self._mm, dtype=np.float32,
                             count=n_elements, offset=start)

    def tensor_data_f16(self, name: str) -> np.ndarray:
        """Return F16 tensor data as float16 numpy view (zero-copy)."""
        info = self.tensors[name]
        assert info.qtype == 1, f"Expected F16, got {info.qtype_name}"
        start = self._data_offset + info.offset
        n_elements = 1
        for d in info.shape:
            n_elements *= d
        return np.frombuffer(self._mm, dtype=np.float16,
                             count=n_elements, offset=start)

    def tensor_data_raw_blocks(self, name: str) -> np.ndarray:
        """Return K-quant tensor as (N_rows, bytes_per_row) uint8 view.

        For Q4_K/Q5_K/Q6_K tensors, the first dimension is the row count
        and the second is n_blocks_per_row × block_bytes.
        """
        info = self.tensors[name]
        if info.qtype not in GGML_BLOCK_SIZES:
            raise ValueError(f"Not a block-quantized type: {info.qtype_name}")
        blk_elems, blk_bytes = GGML_BLOCK_SIZES[info.qtype]
        raw = self.tensor_data(name)
        # GGUF shape is (ne0, ne1) = (K, N) where K is the reduction dim.
        # Data is laid out as N rows of K elements each.
        if len(info.shape) == 2:
            K = info.shape[0]
            N = info.shape[1]
            n_blocks_per_row = (K + blk_elems - 1) // blk_elems
            bytes_per_row = n_blocks_per_row * blk_bytes
            return raw.reshape(N, bytes_per_row)
        return raw

    def close(self):
        if self._mm:
            self._mm.close()
            self._mm = None
        if self._file:
            self._file.close()
            self._file = None


def dequantize_q8_0(raw: np.ndarray, shape: tuple,
                    out_dtype=np.float16) -> np.ndarray:
    """Dequantize Q8_0 tensor using pure numpy (no torch/diffusers needed).

    Q8_0 block layout (34 bytes per 32 elements):
      - 2 bytes: float16 scale (d)
      - 32 bytes: int8 quantized values (qs)
    Dequantization: weight[i] = qs[i] * d
    """
    block_size = 32
    block_bytes = 34
    n_blocks = len(raw) // block_bytes
    blocks = raw.reshape(n_blocks, block_bytes)

    # Extract scales (first 2 bytes of each block) and int8 values (remaining 32)
    scales = blocks[:, :2].view(np.float16).astype(np.float32).ravel()
    qs = blocks[:, 2:].view(np.int8).astype(np.float32)

    # Dequantize: each block's 32 values are multiplied by that block's scale
    result = qs * scales[:, None]
    result = result.ravel()

    # Reshape: GGUF shape is (ne0=K, ne1=N) → output is (N, K) row-major
    if len(shape) == 2:
        K, N = shape
        result = result.reshape(N, K)
    elif len(shape) == 1:
        result = result[:shape[0]]
    return result.astype(out_dtype, copy=False)


def repack_q8_0_for_gpu(raw: np.ndarray, shape: tuple):
    """Repack Q8_0 raw bytes for GPU: packed u32 weights + fp16 scales.

    Q8_0 block layout: [2-byte fp16 scale][32 bytes int8 values]
    Output:
      weights_u32: (N, K//4) as uint32 — 4 int8 values packed per u32
      scales_fp16: (N, K//32) as float16 — one scale per 32-element block
    """
    block_bytes = 34

    if len(shape) == 2:
        K, N = shape  # GGUF convention: ne0=K, ne1=N
    elif len(shape) == 1:
        K = shape[0]
        N = 1
    else:
        raise ValueError(f"Unexpected shape: {shape}")

    n_blocks_per_row = K // 32
    blocks = raw.reshape(N, n_blocks_per_row, block_bytes)

    # Scales: first 2 bytes of each block → fp16
    scales = blocks[:, :, :2].copy().view(np.float16).reshape(N, n_blocks_per_row)

    # Int8 values: bytes 2:34 of each block, contiguous per row
    qs = blocks[:, :, 2:].reshape(N, K)

    # Pack 4 consecutive bytes as u32 (preserving int8 bit patterns)
    weights = qs.copy().view(np.uint32).reshape(N, K // 4)

    return weights, scales


def open_gguf_reader(gguf_path: str):
    """Open GGUF file and return (gguf_module, reader)."""
    import gguf
    from gguf import GGUFReader

    return gguf, GGUFReader(gguf_path)


def build_tensor_dict(reader) -> Dict[str, object]:
    """Return tensor-name -> GGUF tensor object mapping."""
    return {t.name: t for t in reader.tensors}


def quant_type_histogram(gguf_module, tensors: Dict[str, object]) -> Dict[str, int]:
    """Count GGUF tensor quantization types by name."""
    hist = Counter()
    for tensor in tensors.values():
        qtype = gguf_module.GGMLQuantizationType(int(tensor.tensor_type))
        hist[qtype.name] += 1
    return dict(sorted(hist.items(), key=lambda kv: kv[0]))


def dequantize_tensor(
    gguf_module,
    tensors: Dict[str, object],
    name: str,
    out_dtype=np.float16,
) -> Tuple[np.ndarray, object]:
    """Load/dequantize a GGUF tensor to NumPy.

    Returns (array, quant_type_enum).
    """
    if name not in tensors:
        raise KeyError(f"Missing GGUF tensor: {name}")

    tensor = tensors[name]
    qtype = gguf_module.GGMLQuantizationType(int(tensor.tensor_type))
    data = np.asarray(tensor.data)

    if qtype == gguf_module.GGMLQuantizationType.F16:
        arr = data.astype(np.float16, copy=False)
    elif qtype == gguf_module.GGMLQuantizationType.F32:
        arr = data.astype(np.float32, copy=False)
    elif qtype == gguf_module.GGMLQuantizationType.BF16:
        # Convert BF16 payload to FP32 then cast as requested.
        u16 = data.view(np.uint16)
        f32 = np.empty(u16.shape, dtype=np.float32)
        f32.view(np.uint32)[:] = u16.astype(np.uint32) << 16
        arr = f32
    else:
        try:
            import torch
            from diffusers.quantizers.gguf.utils import (
                GGML_QUANT_SIZES,
                dequantize_functions,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Dequantizing GGUF quantized tensors requires diffusers + torch."
            ) from exc

        if qtype not in dequantize_functions:
            raise NotImplementedError(f"Unsupported GGUF quant type: {qtype}")

        block_size, type_size = GGML_QUANT_SIZES[qtype]
        blocks_np = data.reshape(-1, type_size).astype(np.uint8, copy=False)
        # Ensure writable contiguous storage for torch.from_numpy.
        blocks = torch.from_numpy(np.ascontiguousarray(blocks_np).copy())
        deq = dequantize_functions[qtype](
            blocks, block_size, type_size, dtype=torch.float32
        )
        arr = deq.reshape(data.shape[0], -1).cpu().numpy()
        del blocks, deq

    if out_dtype is not None:
        arr = arr.astype(out_dtype, copy=False)
    return arr, qtype
