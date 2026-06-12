"""Tests for rope.wgsl: Rotary Position Embedding compute shader."""
import os
import re
import struct
import numpy as np
import pytest

SHADER_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'shaders', 'rope.wgsl')


def rope_ref(x, head_dim, seq_len, theta_base=10000.0):
    """CPU reference: standard RoPE formula.

    x: shape (seq_len, head_dim), float32
    Returns rotated x with same shape.
    """
    half_dim = head_dim // 2
    out = np.copy(x)
    for pos in range(seq_len):
        for d in range(half_dim):
            freq = 1.0 / (theta_base ** (d / half_dim))
            angle = pos * freq
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            x0 = x[pos, d]
            x1 = x[pos, d + half_dim]
            out[pos, d] = x0 * cos_a - x1 * sin_a
            out[pos, d + half_dim] = x0 * sin_a + x1 * cos_a
    return out


@pytest.fixture
def shader_source():
    assert os.path.exists(SHADER_PATH), f"Shader file not found: {SHADER_PATH}"
    with open(SHADER_PATH) as f:
        return f.read()


def test_shader_file_exists():
    assert os.path.isfile(SHADER_PATH)


def test_shader_is_compute(shader_source):
    assert '@compute' in shader_source
    assert '@workgroup_size' in shader_source
    assert 'global_invocation_id' in shader_source


def test_shader_has_bindings(shader_source):
    assert re.search(r'@binding\(0\)\s*var<storage,\s*read>', shader_source)
    assert re.search(r'@binding\(1\)\s*var<storage,\s*read_write>', shader_source)
    assert re.search(r'@binding\(2\)', shader_source)


def test_shader_implements_rope(shader_source):
    assert 'cos(' in shader_source
    assert 'sin(' in shader_source
    assert 'pow(' in shader_source


def test_rope_cpu_reference_basic():
    head_dim = 8
    seq_len = 4
    x = np.ones((seq_len, head_dim), dtype=np.float32)
    result = rope_ref(x, head_dim, seq_len)
    assert result.shape == (seq_len, head_dim)
    np.testing.assert_allclose(result[0], x[0], atol=1e-6,
                               err_msg="Position 0 should have no rotation (angle=0)")


def test_rope_cpu_reference_position_zero_identity():
    head_dim = 16
    rng = np.random.default_rng(42)
    x = rng.standard_normal((1, head_dim)).astype(np.float32)
    result = rope_ref(x, head_dim, 1)
    np.testing.assert_allclose(result, x, atol=1e-6)


def test_rope_cpu_reference_configurable_theta():
    head_dim = 8
    seq_len = 4
    rng = np.random.default_rng(123)
    x = rng.standard_normal((seq_len, head_dim)).astype(np.float32)
    r1 = rope_ref(x, head_dim, seq_len, theta_base=10000.0)
    r2 = rope_ref(x, head_dim, seq_len, theta_base=500000.0)
    assert not np.allclose(r1, r2), "Different theta bases should produce different results"


def test_rope_cpu_reference_preserves_norm():
    """RoPE is a rotation — it should preserve the L2 norm of each (x0, x1) pair."""
    head_dim = 16
    seq_len = 8
    rng = np.random.default_rng(7)
    x = rng.standard_normal((seq_len, head_dim)).astype(np.float32)
    result = rope_ref(x, head_dim, seq_len)
    half = head_dim // 2
    for pos in range(seq_len):
        for d in range(half):
            orig_norm = np.sqrt(x[pos, d] ** 2 + x[pos, d + half] ** 2)
            new_norm = np.sqrt(result[pos, d] ** 2 + result[pos, d + half] ** 2)
            np.testing.assert_allclose(new_norm, orig_norm, atol=1e-5)


def test_rope_cpu_reference_f16_precision():
    """Validate that RoPE with f16 inputs has max absolute error < 1e-2 vs f32 reference."""
    head_dim = 64
    seq_len = 32
    rng = np.random.default_rng(42)
    x_f32 = rng.standard_normal((seq_len, head_dim)).astype(np.float32)

    ref_f32 = rope_ref(x_f32, head_dim, seq_len)

    x_f16_roundtrip = x_f32.astype(np.float16).astype(np.float32)
    result_from_f16 = rope_ref(x_f16_roundtrip, head_dim, seq_len)

    max_abs_error = np.max(np.abs(result_from_f16 - ref_f32))
    assert max_abs_error < 1e-2, f"Max absolute error {max_abs_error} exceeds 1e-2 tolerance"


def test_rope_cpu_reference_different_head_dims():
    """Verify RoPE works for various head_dim sizes."""
    for head_dim in [8, 32, 64, 128]:
        seq_len = 4
        rng = np.random.default_rng(head_dim)
        x = rng.standard_normal((seq_len, head_dim)).astype(np.float32)
        result = rope_ref(x, head_dim, seq_len)
        assert result.shape == (seq_len, head_dim)
        assert np.all(np.isfinite(result))
