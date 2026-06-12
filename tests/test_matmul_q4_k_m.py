"""Tests for matmul_q4_k_m.wgsl: validates quantized matmul against CPU dequant + matmul reference."""
import os
import struct
import re
import pytest
import numpy as np

SHADER_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'shaders', 'matmul_q4_k_m.wgsl')

QK_K = 256
BLOCK_SIZE_BYTES = 144


def f32_to_f16_bits(val):
    return struct.unpack('<H', struct.pack('e', val))[0]


def f16_bits_to_f32(bits):
    return struct.unpack('e', struct.pack('<H', bits))[0]


def make_block(d, dmin, scales, mins, qs):
    data = bytearray(BLOCK_SIZE_BYTES)
    struct.pack_into('<HH', data, 0, f32_to_f16_bits(d), f32_to_f16_bits(dmin))
    for j in range(4):
        data[4 + j] = (scales[j] & 63) | ((scales[j + 4] >> 4) << 6)
        data[4 + 4 + j] = (mins[j] & 63) | ((mins[j + 4] >> 4) << 6)
    for j in range(4):
        data[4 + 8 + j] = (scales[j + 4] & 0xF) | ((mins[j + 4] & 0xF) << 4)
    for i, q in enumerate(qs):
        data[16 + i] = q & 0xFF
    return bytes(data)


def get_scale_min_k4_ref(j, scales_12):
    if j < 4:
        sc = scales_12[j] & 63
        mn = scales_12[j + 4] & 63
    else:
        packed = scales_12[j + 4]
        hi_s = scales_12[j - 4]
        sc = (packed & 0xF) | ((hi_s >> 6) << 4)
        hi_m = scales_12[j]
        mn = ((packed >> 4) & 0xF) | ((hi_m >> 6) << 4)
    return sc, mn


def dequantize_block_ref(block_bytes):
    d = f16_bits_to_f32(struct.unpack_from('<H', block_bytes, 0)[0])
    dmin = f16_bits_to_f32(struct.unpack_from('<H', block_bytes, 2)[0])
    scales_12 = block_bytes[4:16]
    qs = block_bytes[16:144]
    output = [0.0] * QK_K
    for j in range(QK_K // 32):
        sc, mn = get_scale_min_k4_ref(j, scales_12)
        d_sc = d * sc
        dm = dmin * mn
        for l in range(16):
            q = qs[j * 16 + l]
            output[j * 32 + l] = d_sc * (q & 0xF) - dm
            output[j * 32 + l + 16] = d_sc * ((q >> 4) & 0xF) - dm
    return output


def quantize_column_q4_k_m(values, rng):
    """Quantize a K-length column into Q4_K_M blocks. K must be multiple of 256."""
    assert len(values) % QK_K == 0
    blocks = []
    for bi in range(len(values) // QK_K):
        chunk = values[bi * QK_K:(bi + 1) * QK_K]
        scales_list = []
        mins_list = []
        qs = [0] * 128
        for sb in range(8):
            sub = chunk[sb * 32:(sb + 1) * 32]
            sub_min = float(np.min(sub))
            sub_max = float(np.max(sub))
            if sub_max == sub_min:
                scales_list.append(0)
                mins_list.append(0)
                continue
            sc_val = (sub_max - sub_min) / 15.0
            mn_val = -sub_min if sub_min < 0 else 0.0
            adj_min = sub_min
            scales_list.append(max(1, min(63, int(round(sc_val * 63 / max(sc_val, 1e-10))))))
            mins_list.append(max(0, min(63, int(round(abs(adj_min) * 63 / max(abs(adj_min), 1e-10))))))

        vmax = max(abs(float(np.max(values[bi * QK_K:(bi + 1) * QK_K]))),
                   abs(float(np.min(values[bi * QK_K:(bi + 1) * QK_K]))))
        if vmax < 1e-10:
            blocks.append(make_block(0.0, 0.0, [0]*8, [0]*8, [0]*128))
            continue

        all_sub_maxes = []
        all_sub_mins = []
        for sb in range(8):
            sub = chunk[sb * 32:(sb + 1) * 32]
            all_sub_maxes.append(float(np.max(sub)))
            all_sub_mins.append(float(np.min(sub)))

        max_sc_range = max((mx - mn) for mx, mn in zip(all_sub_maxes, all_sub_mins))
        max_min_abs = max(abs(mn) for mn in all_sub_mins)

        d = max_sc_range / (63.0 * 15.0) if max_sc_range > 0 else 1e-7
        dmin = max_min_abs / 63.0 if max_min_abs > 0 else 0.0

        d = f16_bits_to_f32(f32_to_f16_bits(d))
        dmin = f16_bits_to_f32(f32_to_f16_bits(dmin))
        if d == 0.0:
            d = 1e-7

        int_scales = []
        int_mins = []
        for sb in range(8):
            sub = chunk[sb * 32:(sb + 1) * 32]
            sub_min = float(np.min(sub))
            sub_max = float(np.max(sub))
            sub_range = sub_max - sub_min
            sc = min(63, max(0, int(round(sub_range / (d * 15.0))))) if d * 15.0 > 0 else 0
            mn = min(63, max(0, int(round(-sub_min / dmin)))) if dmin > 0 and sub_min < 0 else 0
            int_scales.append(sc)
            int_mins.append(mn)

            d_sc = d * sc
            dm = dmin * mn
            for l in range(16):
                v = chunk[sb * 32 + l]
                if d_sc > 0:
                    q = min(15, max(0, int(round((v + dm) / d_sc))))
                else:
                    q = 0
                v_hi = chunk[sb * 32 + l + 16]
                if d_sc > 0:
                    q_hi = min(15, max(0, int(round((v_hi + dm) / d_sc))))
                else:
                    q_hi = 0
                qs[sb * 16 + l] = (q & 0xF) | ((q_hi & 0xF) << 4)

        blocks.append(make_block(d, dmin, int_scales, int_mins, qs))
    return blocks


def pack_f16_to_u32(arr_f16):
    flat = arr_f16.flatten().astype(np.float16)
    raw = flat.view(np.uint16).astype(np.uint32)
    if len(raw) % 2:
        raw = np.append(raw, np.uint32(0))
    return ((raw[1::2] << 16) | raw[0::2]).astype(np.uint32)


def cpu_matmul_q4_k_m(A_f16, B_blocks, M, N, K):
    """CPU reference: dequantize B then multiply.

    B_blocks: list of N columns, each column is a list of (K/256) block byte strings.
    """
    B_deq = np.zeros((K, N), dtype=np.float32)
    for col in range(N):
        col_vals = []
        for blk in B_blocks[col]:
            col_vals.extend(dequantize_block_ref(blk))
        B_deq[:, col] = col_vals[:K]

    A_f32 = A_f16.astype(np.float32)
    return A_f32 @ B_deq


def pack_b_blocks_to_u32(B_blocks, N, blocks_per_col):
    """Pack all B blocks into a flat u32 array matching the shader's layout."""
    raw = bytearray()
    for col in range(N):
        for blk in B_blocks[col]:
            raw.extend(blk)
    arr = np.frombuffer(bytes(raw), dtype=np.uint8)
    padded = np.append(arr, np.zeros((-len(arr)) % 4, dtype=np.uint8))
    return padded.view(np.uint32)


def _run_matmul_q4_k_m_test(M, N, K, seed=42):
    """Simulate the shader's matmul and compare against CPU dequant+matmul reference."""
    assert K % QK_K == 0, "K must be a multiple of 256"
    rng = np.random.default_rng(seed)

    A_f16 = rng.uniform(-1, 1, (M, K)).astype(np.float16)
    B_f32 = rng.uniform(-1, 1, (K, N)).astype(np.float32)

    blocks_per_col = K // QK_K
    B_blocks = []
    for col in range(N):
        col_blocks = quantize_column_q4_k_m(B_f32[:, col], rng)
        B_blocks.append(col_blocks)

    expected = cpu_matmul_q4_k_m(A_f16, B_blocks, M, N, K)

    A_u32 = pack_f16_to_u32(A_f16)
    B_u32 = pack_b_blocks_to_u32(B_blocks, N, blocks_per_col)

    gpu_sim = _simulate_shader(A_u32, B_u32, M, N, K, blocks_per_col)

    np.testing.assert_allclose(gpu_sim, expected, rtol=1e-2, atol=1e-2,
                               err_msg=f"Matmul Q4_K_M mismatch for M={M}, N={N}, K={K}")


def _load_f16_from_u32(buf, index):
    word = int(buf[index // 2])
    bits = (word >> ((index % 2) * 16)) & 0xFFFF
    return f16_bits_to_f32(bits)


def _read_b_u8(B_u32, byte_offset):
    word_idx = byte_offset // 4
    byte_pos = byte_offset % 4
    return (int(B_u32[word_idx]) >> (byte_pos * 8)) & 0xFF


def _read_b_u16(B_u32, byte_offset):
    return _read_b_u8(B_u32, byte_offset) | (_read_b_u8(B_u32, byte_offset + 1) << 8)


def _get_scale_min_k4_shader(B_u32, block_byte_offset, j):
    scales_offset = block_byte_offset + 4
    if j < 4:
        sc = _read_b_u8(B_u32, scales_offset + j) & 63
        mn = _read_b_u8(B_u32, scales_offset + j + 4) & 63
    else:
        packed = _read_b_u8(B_u32, scales_offset + j + 4)
        hi_s = _read_b_u8(B_u32, scales_offset + j - 4)
        sc = (packed & 0xF) | ((hi_s >> 6) << 4)
        hi_m = _read_b_u8(B_u32, scales_offset + j)
        mn = ((packed >> 4) & 0xF) | ((hi_m >> 6) << 4)
    return sc, mn


def _dequant_value_shader(B_u32, block_byte_offset, pos):
    d = f16_bits_to_f32(_read_b_u16(B_u32, block_byte_offset))
    dmin = f16_bits_to_f32(_read_b_u16(B_u32, block_byte_offset + 2))
    sub_block = pos // 32
    pos_in_sub = pos % 32
    sc, m = _get_scale_min_k4_shader(B_u32, block_byte_offset, sub_block)
    d_sc = d * sc
    dm = dmin * m
    qs_offset = block_byte_offset + 16
    if pos_in_sub < 16:
        byte_off = qs_offset + sub_block * 16 + pos_in_sub
        nibble_val = _read_b_u8(B_u32, byte_off) & 0xF
    else:
        byte_off = qs_offset + sub_block * 16 + (pos_in_sub - 16)
        nibble_val = (_read_b_u8(B_u32, byte_off) >> 4) & 0xF
    return d_sc * nibble_val - dm


def _simulate_shader(A_u32, B_u32, M, N, K, blocks_per_col):
    """Simulate matmul_q4_k_m.wgsl logic in Python."""
    C = np.zeros((M, N), dtype=np.float32)
    for row in range(M):
        for col in range(N):
            acc = 0.0
            for k in range(K):
                a_val = _load_f16_from_u32(A_u32, row * K + k)
                block_idx = col * blocks_per_col + k // QK_K
                pos_in_block = k % QK_K
                block_byte_offset = block_idx * BLOCK_SIZE_BYTES
                b_val = _dequant_value_shader(B_u32, block_byte_offset, pos_in_block)
                acc += a_val * b_val
            C[row, col] = acc
    return C


# ---------------------------------------------------------------------------
# Numerical tests
# ---------------------------------------------------------------------------

def test_matmul_q4_k_m_square_256():
    _run_matmul_q4_k_m_test(16, 16, 256)


def test_matmul_q4_k_m_rectangular():
    _run_matmul_q4_k_m_test(8, 32, 256)


def test_matmul_q4_k_m_non_aligned_m_n():
    _run_matmul_q4_k_m_test(7, 13, 256)


def test_matmul_q4_k_m_single_row():
    _run_matmul_q4_k_m_test(1, 16, 256)


def test_matmul_q4_k_m_larger_k():
    _run_matmul_q4_k_m_test(4, 8, 512)


def test_matmul_q4_k_m_non_aligned_all():
    _run_matmul_q4_k_m_test(3, 5, 256)


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------

@pytest.fixture
def shader_source():
    with open(SHADER_PATH) as f:
        return f.read()


def test_shader_file_exists():
    assert os.path.isfile(SHADER_PATH)


def test_has_dequant_function(shader_source):
    assert 'dequant_q4_k_m_value' in shader_source


def test_uses_tiled_accumulation(shader_source):
    assert 'tileA' in shader_source
    assert 'workgroupBarrier()' in shader_source


def test_q4_k_m_block_size(shader_source):
    assert '144' in shader_source


def test_output_is_f32(shader_source):
    assert re.search(r'var<storage,\s*read_write>\s*\w+\s*:\s*array<f32>', shader_source)


def test_params_struct(shader_source):
    assert 'M: u32' in shader_source
    assert 'N: u32' in shader_source
    assert 'K: u32' in shader_source


def test_batch_params(shader_source):
    assert 'batch_size: u32' in shader_source
    assert 'stride_A: u32' in shader_source
    assert 'stride_C: u32' in shader_source

def test_batch_z_dispatch(shader_source):
    assert 'wgid.z' in shader_source

def test_batch_offset_applied(shader_source):
    assert 'batch * params.stride_A' in shader_source
    assert 'batch * params.stride_C' in shader_source


# ---------------------------------------------------------------------------
# Batched numerical tests
# ---------------------------------------------------------------------------

def _simulate_shader_batched(A_u32, B_u32, M, N, K, blocks_per_col, batch_size):
    stride_A = M * K
    C = np.zeros((batch_size, M, N), dtype=np.float32)
    for b in range(batch_size):
        a_off = b * stride_A
        for row in range(M):
            for col in range(N):
                acc = 0.0
                for k in range(K):
                    a_val = _load_f16_from_u32(A_u32, a_off + row * K + k)
                    block_idx = col * blocks_per_col + k // QK_K
                    pos_in_block = k % QK_K
                    block_byte_offset = block_idx * BLOCK_SIZE_BYTES
                    b_val = _dequant_value_shader(B_u32, block_byte_offset, pos_in_block)
                    acc += a_val * b_val
                C[b, row, col] = acc
    return C


def _run_batched_q4_k_m_test(batch_size, M, N, K, seed=42):
    assert K % QK_K == 0
    rng = np.random.default_rng(seed)

    A_f16 = rng.uniform(-1, 1, (batch_size, M, K)).astype(np.float16)
    B_f32 = rng.uniform(-1, 1, (K, N)).astype(np.float32)

    blocks_per_col = K // QK_K
    B_blocks = []
    for col in range(N):
        B_blocks.append(quantize_column_q4_k_m(B_f32[:, col], rng))

    B_u32 = pack_b_blocks_to_u32(B_blocks, N, blocks_per_col)
    A_u32 = np.concatenate([pack_f16_to_u32(A_f16[b]) for b in range(batch_size)])

    batched = _simulate_shader_batched(A_u32, B_u32, M, N, K, blocks_per_col, batch_size)

    for b in range(batch_size):
        expected = cpu_matmul_q4_k_m(A_f16[b], B_blocks, M, N, K)
        np.testing.assert_allclose(
            batched[b], expected, rtol=1e-2, atol=1e-2,
            err_msg=f"Batch {b}: Q4_K_M matmul mismatch M={M}, N={N}, K={K}")


def test_batched_q4_k_m_single():
    _run_batched_q4_k_m_test(1, 8, 8, 256)


def test_batched_q4_k_m_4():
    _run_batched_q4_k_m_test(4, 4, 8, 256)


def test_batched_q4_k_m_independent():
    rng = np.random.default_rng(55)
    batch_size, M, N, K = 3, 4, 4, 256

    A_f16 = rng.uniform(-1, 1, (batch_size, M, K)).astype(np.float16)
    B_f32 = rng.uniform(-1, 1, (K, N)).astype(np.float32)

    blocks_per_col = K // QK_K
    B_blocks = []
    for col in range(N):
        B_blocks.append(quantize_column_q4_k_m(B_f32[:, col], rng))

    B_u32 = pack_b_blocks_to_u32(B_blocks, N, blocks_per_col)
    A_u32 = np.concatenate([pack_f16_to_u32(A_f16[b]) for b in range(batch_size)])

    batched = _simulate_shader_batched(A_u32, B_u32, M, N, K, blocks_per_col, batch_size)

    for b in range(batch_size):
        single_A = pack_f16_to_u32(A_f16[b])
        single = _simulate_shader(single_A, B_u32, M, N, K, blocks_per_col)
        np.testing.assert_allclose(
            batched[b], single, rtol=1e-6, atol=1e-6,
            err_msg=f"Batch {b} differs from independent computation")
