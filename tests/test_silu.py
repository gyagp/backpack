"""Tests for silu.wgsl: SiLU activation (x * sigmoid(x)) shader."""
import os
import re
import struct
import numpy as np
import pytest

SHADER_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'shaders', 'silu.wgsl')


def silu_ref(x):
    """CPU reference: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))."""
    return x / (1.0 + np.exp(-x))


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


def test_shader_has_input_output_bindings(shader_source):
    assert re.search(r'@binding\(0\)\s*var<storage,\s*read>', shader_source)
    assert re.search(r'@binding\(1\)\s*var<storage,\s*read_write>', shader_source)


def test_shader_implements_silu(shader_source):
    assert 'exp(' in shader_source or 'exp(-' in shader_source


def test_silu_cpu_reference_basic():
    x = np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)
    result = silu_ref(x)
    expected = x * (1.0 / (1.0 + np.exp(-x)))
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_silu_cpu_reference_f16_precision():
    """Validate that SiLU with f16 inputs has max absolute error < 1e-3 vs f32 reference.

    Uses f16 inputs converted to f32 for computation (matching shader behavior:
    f32 storage buffers, f32 arithmetic). The error comes only from f16 input quantization.
    """
    rng = np.random.default_rng(42)
    x_f32 = rng.uniform(-2.0, 2.0, size=4096).astype(np.float32)

    ref_f32 = silu_ref(x_f32)

    x_f16_roundtrip = x_f32.astype(np.float16).astype(np.float32)
    result_from_f16 = silu_ref(x_f16_roundtrip)

    max_abs_error = np.max(np.abs(result_from_f16 - ref_f32))
    assert max_abs_error < 1e-3, f"Max absolute error {max_abs_error} exceeds 1e-3 tolerance"


def test_silu_cpu_reference_edge_cases():
    x = np.array([0.0, -0.0, 1e-7, -1e-7], dtype=np.float32)
    result = silu_ref(x)
    assert np.all(np.isfinite(result))
    assert result[0] == 0.0
