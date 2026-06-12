"""Tests for fused_qkv.wgsl: structural checks + numerical validation against numpy."""
import os
import re
import struct
import pytest
import numpy as np

SHADER_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'shaders', 'fused_qkv.wgsl')


@pytest.fixture
def shader_source():
    assert os.path.exists(SHADER_PATH), f"Shader file not found: {SHADER_PATH}"
    with open(SHADER_PATH) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------

def test_file_exists():
    assert os.path.exists(SHADER_PATH)


def test_has_input_binding(shader_source):
    assert re.search(r'@binding\(0\)\s+var<storage,\s*read>\s+input\s*:\s*array<u32>', shader_source)


def test_has_wq_binding(shader_source):
    assert re.search(r'@binding\(1\)\s+var<storage,\s*read>\s+Wq\s*:\s*array<u32>', shader_source)


def test_has_wk_binding(shader_source):
    assert re.search(r'@binding\(2\)\s+var<storage,\s*read>\s+Wk\s*:\s*array<u32>', shader_source)


def test_has_wv_binding(shader_source):
    assert re.search(r'@binding\(3\)\s+var<storage,\s*read>\s+Wv\s*:\s*array<u32>', shader_source)


def test_has_outQ_binding(shader_source):
    assert re.search(r'@binding\(4\)\s+var<storage,\s*read_write>\s+outQ\s*:\s*array<f32>', shader_source)


def test_has_outK_binding(shader_source):
    assert re.search(r'@binding\(5\)\s+var<storage,\s*read_write>\s+outK\s*:\s*array<f32>', shader_source)


def test_has_outV_binding(shader_source):
    assert re.search(r'@binding\(6\)\s+var<storage,\s*read_write>\s+outV\s*:\s*array<f32>', shader_source)


def test_has_params_binding(shader_source):
    assert re.search(r'@binding\(7\)\s+var<uniform>\s+params\s*:\s*Params', shader_source)


def test_is_compute_shader(shader_source):
    assert re.search(r'@compute\s+@workgroup_size\(16,\s*16\)', shader_source)


def test_uses_shared_memory(shader_source):
    assert 'var<workgroup>' in shader_source


def test_single_dispatch_entry_point(shader_source):
    assert shader_source.count('fn main(') == 1


def test_all_three_outputs_stored(shader_source):
    assert 'outQ[' in shader_source or 'outQ [' in shader_source
    assert 'outK[' in shader_source or 'outK [' in shader_source
    assert 'outV[' in shader_source or 'outV [' in shader_source


# ---------------------------------------------------------------------------
# CPU reference helpers
# ---------------------------------------------------------------------------

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


def cpu_matmul(A_u32: np.ndarray, B_u32: np.ndarray, M: int, N: int, K: int) -> np.ndarray:
    A_f32 = np.array([load_f16_from_u32(A_u32, i) for i in range(M * K)], dtype=np.float32).reshape(M, K)
    B_f32 = np.array([load_f16_from_u32(B_u32, i) for i in range(K * N)], dtype=np.float32).reshape(K, N)
    return A_f32 @ B_f32


# ---------------------------------------------------------------------------
# Numerical validation: fused QKV should match 3 separate matmuls
# ---------------------------------------------------------------------------

def _run_fused_qkv_test(M: int, N: int, K: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    input_f16 = rng.uniform(-1, 1, (M, K)).astype(np.float16)
    Wq_f16 = rng.uniform(-1, 1, (K, N)).astype(np.float16)
    Wk_f16 = rng.uniform(-1, 1, (K, N)).astype(np.float16)
    Wv_f16 = rng.uniform(-1, 1, (K, N)).astype(np.float16)

    input_u32 = pack_f16_to_u32(input_f16)
    Wq_u32 = pack_f16_to_u32(Wq_f16)
    Wk_u32 = pack_f16_to_u32(Wk_f16)
    Wv_u32 = pack_f16_to_u32(Wv_f16)

    Q_ref = cpu_matmul(input_u32, Wq_u32, M, N, K)
    K_ref = cpu_matmul(input_u32, Wk_u32, M, N, K)
    V_ref = cpu_matmul(input_u32, Wv_u32, M, N, K)

    Q_direct = input_f16.astype(np.float32) @ Wq_f16.astype(np.float32)
    K_direct = input_f16.astype(np.float32) @ Wk_f16.astype(np.float32)
    V_direct = input_f16.astype(np.float32) @ Wv_f16.astype(np.float32)

    np.testing.assert_allclose(Q_ref, Q_direct, rtol=1e-2, atol=1e-2,
                               err_msg=f"Q mismatch for {M}x{K} @ {K}x{N}")
    np.testing.assert_allclose(K_ref, K_direct, rtol=1e-2, atol=1e-2,
                               err_msg=f"K mismatch for {M}x{K} @ {K}x{N}")
    np.testing.assert_allclose(V_ref, V_direct, rtol=1e-2, atol=1e-2,
                               err_msg=f"V mismatch for {M}x{K} @ {K}x{N}")


def test_fused_qkv_square_128():
    _run_fused_qkv_test(128, 128, 128)


def test_fused_qkv_non_aligned():
    _run_fused_qkv_test(100, 100, 100)


def test_fused_qkv_rectangular():
    _run_fused_qkv_test(80, 64, 48)


def test_fused_qkv_single_row():
    _run_fused_qkv_test(1, 64, 128)
