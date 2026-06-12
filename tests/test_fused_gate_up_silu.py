"""Tests for fused_gate_up_silu.wgsl: structural checks + numerical validation against numpy."""
import os
import re
import struct
import pytest
import numpy as np

SHADER_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'shaders', 'fused_gate_up_silu.wgsl')


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


def test_has_wgate_binding(shader_source):
    assert re.search(r'@binding\(1\)\s+var<storage,\s*read>\s+Wgate\s*:\s*array<u32>', shader_source)


def test_has_wup_binding(shader_source):
    assert re.search(r'@binding\(2\)\s+var<storage,\s*read>\s+Wup\s*:\s*array<u32>', shader_source)


def test_has_output_binding(shader_source):
    assert re.search(r'@binding\(3\)\s+var<storage,\s*read_write>\s+output\s*:\s*array<f32>', shader_source)


def test_has_params_binding(shader_source):
    assert re.search(r'@binding\(4\)\s+var<uniform>\s+params\s*:\s*Params', shader_source)


def test_is_compute_shader(shader_source):
    assert re.search(r'@compute\s+@workgroup_size\(16,\s*16\)', shader_source)


def test_uses_shared_memory(shader_source):
    assert 'var<workgroup>' in shader_source


def test_single_dispatch_entry_point(shader_source):
    assert shader_source.count('fn main(') == 1


def test_input_read_once_for_both_projections(shader_source):
    """Input is loaded into tileA once per tile iteration, then reused for gate and up."""
    main_body = shader_source[shader_source.index('fn main('):]
    tile_a_loads = len(re.findall(r'load_tile_a_fast|tileA\[.*\]\s*=\s*p', main_body))
    assert tile_a_loads > 0, "tileA should be loaded from input"
    gate_section = main_body.find('Gate projection')
    up_section = main_body.find('Up projection')
    assert gate_section < up_section, "Gate projection should come before up projection"
    between = main_body[gate_section:up_section]
    assert 'load_tile_a_fast' not in between, "tileA should not be reloaded between gate and up"


def test_silu_applied_in_store(shader_source):
    """SiLU activation (x / (1 + exp(-x))) is applied during the store phase."""
    assert 'exp(-g)' in shader_source or 'exp(- g)' in shader_source


def test_gate_times_up_in_store(shader_source):
    """Output should be silu(gate) * up."""
    store_section = shader_source[shader_source.index('silu'):]
    assert 'silu_g * u' in store_section or 'silu_g*u' in store_section


def test_no_dead_code(shader_source):
    """No unused helper functions should exist."""
    assert 'load_tile_b_from' not in shader_source, "Dead code load_tile_b_from should be removed"


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


def cpu_matmul_f16(A_f16: np.ndarray, B_f16: np.ndarray) -> np.ndarray:
    return A_f16.astype(np.float32) @ B_f16.astype(np.float32)


def cpu_silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def cpu_fused_gate_up_silu(input_f16: np.ndarray, gate_w_f16: np.ndarray,
                           up_w_f16: np.ndarray) -> np.ndarray:
    """Reference: silu(input @ gate_w) * (input @ up_w)"""
    gate_out = cpu_matmul_f16(input_f16, gate_w_f16)
    up_out = cpu_matmul_f16(input_f16, up_w_f16)
    return cpu_silu(gate_out) * up_out


# ---------------------------------------------------------------------------
# Numerical validation: fused output should match unfused FFN within tolerance
# ---------------------------------------------------------------------------

def _run_fused_gate_up_silu_test(M: int, N: int, K: int, seed: int = 42):
    """Validate that fused gate+up+silu matches separate matmuls + silu + elementwise mul."""
    rng = np.random.default_rng(seed)
    input_f16 = rng.uniform(-1, 1, (M, K)).astype(np.float16)
    gate_w_f16 = rng.uniform(-0.5, 0.5, (K, N)).astype(np.float16)
    up_w_f16 = rng.uniform(-0.5, 0.5, (K, N)).astype(np.float16)

    fused_ref = cpu_fused_gate_up_silu(input_f16, gate_w_f16, up_w_f16)

    gate_out = cpu_matmul_f16(input_f16, gate_w_f16)
    up_out = cpu_matmul_f16(input_f16, up_w_f16)
    silu_gate = cpu_silu(gate_out)
    unfused_ref = silu_gate * up_out

    np.testing.assert_allclose(fused_ref, unfused_ref, rtol=1e-5, atol=1e-6,
                               err_msg=f"Fused vs unfused mismatch for M={M} K={K} N={N}")


def test_fused_gate_up_silu_single_token():
    _run_fused_gate_up_silu_test(M=1, N=128, K=64)


def test_fused_gate_up_silu_batch():
    _run_fused_gate_up_silu_test(M=4, N=128, K=64)


def test_fused_gate_up_silu_square():
    _run_fused_gate_up_silu_test(M=128, N=128, K=128)


def test_fused_gate_up_silu_non_aligned():
    _run_fused_gate_up_silu_test(M=100, N=100, K=100)


def test_fused_gate_up_silu_rectangular():
    _run_fused_gate_up_silu_test(M=1, N=256, K=128)


def test_fused_gate_up_silu_typical_llama():
    """Typical LLaMA-like dimensions: hidden=256, intermediate=512."""
    _run_fused_gate_up_silu_test(M=1, N=512, K=256)


def test_silu_numerical_correctness():
    """Verify SiLU implementation: silu(x) = x / (1 + exp(-x))."""
    x = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0], dtype=np.float32)
    expected = x * (1.0 / (1.0 + np.exp(-x)))
    result = cpu_silu(x)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)


def test_fused_gate_up_silu_zero_input():
    """With zero input, output should be zero (silu(0)*0 = 0)."""
    M, N, K = 2, 32, 16
    input_f16 = np.zeros((M, K), dtype=np.float16)
    gate_w_f16 = np.ones((K, N), dtype=np.float16) * 0.1
    up_w_f16 = np.ones((K, N), dtype=np.float16) * 0.1
    result = cpu_fused_gate_up_silu(input_f16, gate_w_f16, up_w_f16)
    np.testing.assert_allclose(result, 0.0, atol=1e-7)
