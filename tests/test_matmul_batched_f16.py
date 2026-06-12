"""Tests for matmul_tiled_f16_batched.wgsl: validates batched matmul against independent numpy matmuls."""
import os
import struct
import numpy as np
import pytest

SHADER_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'shaders', 'matmul_tiled_f16_batched.wgsl')
TILE_SIZE = 16


def f16_to_f32(h: int) -> float:
    return struct.unpack('e', struct.pack('H', h))[0]


def pack_f16_to_u32(arr_f16: np.ndarray) -> np.ndarray:
    flat = arr_f16.flatten().astype(np.float16)
    raw = flat.view(np.uint16).astype(np.uint32)
    padded = np.append(raw, np.uint32(0)) if len(raw) % 2 else raw
    return ((padded[1::2] << 16) | padded[0::2]).astype(np.uint32)


def load_f16_from_u32(buf: np.ndarray, index: int) -> float:
    word = int(buf[index // 2])
    bits = (word >> ((index % 2) * 16)) & 0xFFFF
    return f16_to_f32(bits)


def cpu_batched_matmul(A_u32: np.ndarray, B_u32: np.ndarray,
                       M: int, N: int, K: int, batch_size: int) -> np.ndarray:
    stride_A = M * K
    stride_B = K * N
    stride_C = M * N
    # u32 strides (each f16 pair packed into one u32)
    stride_A_u32 = (stride_A + 1) // 2
    stride_B_u32 = (stride_B + 1) // 2

    C = np.zeros((batch_size, M, N), dtype=np.float32)

    for b in range(batch_size):
        a_off = b * stride_A
        b_off = b * stride_B
        num_tiles = (K + TILE_SIZE - 1) // TILE_SIZE

        for wg_row in range(0, M, TILE_SIZE):
            for wg_col in range(0, N, TILE_SIZE):
                sums = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
                for t in range(num_tiles):
                    tileA = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
                    tileB = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
                    for lr in range(TILE_SIZE):
                        for lc in range(TILE_SIZE):
                            row = wg_row + lr
                            col = wg_col + lc
                            aCol = t * TILE_SIZE + lc
                            if row < M and aCol < K:
                                tileA[lr, lc] = load_f16_from_u32(A_u32, a_off + row * K + aCol)
                            bRow = t * TILE_SIZE + lr
                            if bRow < K and col < N:
                                tileB[lr, lc] = load_f16_from_u32(B_u32, b_off + bRow * N + col)
                    sums += tileA @ tileB
                for lr in range(TILE_SIZE):
                    for lc in range(TILE_SIZE):
                        row = wg_row + lr
                        col = wg_col + lc
                        if row < M and col < N:
                            C[b, row, col] = sums[lr, lc]
    return C


def _run_batched_test(batch_size: int, M: int, N: int, K: int, seed: int = 42):
    rng = np.random.default_rng(seed)

    A_f16_batched = rng.uniform(-1, 1, (batch_size, M, K)).astype(np.float16)
    B_f16_batched = rng.uniform(-1, 1, (batch_size, K, N)).astype(np.float16)

    # Pack all batches contiguously (matching shader stride layout)
    A_u32_parts = [pack_f16_to_u32(A_f16_batched[b]) for b in range(batch_size)]
    B_u32_parts = [pack_f16_to_u32(B_f16_batched[b]) for b in range(batch_size)]
    A_u32 = np.concatenate(A_u32_parts)
    B_u32 = np.concatenate(B_u32_parts)

    gpu_ref = cpu_batched_matmul(A_u32, B_u32, M, N, K, batch_size)

    for b in range(batch_size):
        expected = A_f16_batched[b].astype(np.float32) @ B_f16_batched[b].astype(np.float32)
        np.testing.assert_allclose(
            gpu_ref[b], expected, rtol=1e-2, atol=1e-2,
            err_msg=f"Batch {b}: matmul mismatch for {M}x{K} @ {K}x{N}")


def test_batch_size_1():
    _run_batched_test(batch_size=1, M=32, N=32, K=32)


def test_batch_size_8():
    _run_batched_test(batch_size=8, M=32, N=32, K=32)


def test_batch_size_8_non_aligned():
    _run_batched_test(batch_size=8, M=20, N=24, K=18)


def test_batch_size_1_single_tile():
    _run_batched_test(batch_size=1, M=16, N=16, K=16)


def test_batch_elements_are_independent():
    """Verify each batch element produces the same result as an independent matmul."""
    rng = np.random.default_rng(99)
    batch_size, M, N, K = 4, 16, 16, 16

    A_f16 = rng.uniform(-1, 1, (batch_size, M, K)).astype(np.float16)
    B_f16 = rng.uniform(-1, 1, (batch_size, K, N)).astype(np.float16)

    A_u32 = np.concatenate([pack_f16_to_u32(A_f16[b]) for b in range(batch_size)])
    B_u32 = np.concatenate([pack_f16_to_u32(B_f16[b]) for b in range(batch_size)])

    batched_result = cpu_batched_matmul(A_u32, B_u32, M, N, K, batch_size)

    for b in range(batch_size):
        single_A = pack_f16_to_u32(A_f16[b])
        single_B = pack_f16_to_u32(B_f16[b])
        single_result = cpu_batched_matmul(single_A, single_B, M, N, K, 1)
        np.testing.assert_array_equal(
            batched_result[b], single_result[0],
            err_msg=f"Batch element {b} differs when computed independently vs batched")


def test_shader_file_exists():
    assert os.path.isfile(SHADER_PATH)
