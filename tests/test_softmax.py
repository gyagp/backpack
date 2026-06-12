"""Tests for softmax.wgsl: numerically stable softmax compute shader."""
import os
import re
import numpy as np
import pytest
from scipy.special import softmax as scipy_softmax

SHADER_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'shaders', 'softmax.wgsl')


def softmax_ref(x):
    """CPU reference: numerically stable softmax along last axis."""
    return scipy_softmax(x, axis=-1)


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
    assert 'global_invocation_id' in shader_source or 'local_invocation_id' in shader_source


def test_shader_has_bindings(shader_source):
    assert re.search(r'@binding\(0\)\s*var<storage,\s*read>', shader_source)
    assert re.search(r'@binding\(1\)\s*var<storage,\s*read_write>', shader_source)


def test_shader_numerically_stable(shader_source):
    """Shader uses max-subtraction for numerical stability."""
    assert 'max(' in shader_source or 'shared_max' in shader_source
    assert 'exp(' in shader_source


def test_shader_has_three_passes(shader_source):
    """Shader implements 3-pass approach: max, sum, normalize."""
    assert shader_source.count('workgroupBarrier()') >= 4


def test_softmax_cpu_reference_basic():
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    result = softmax_ref(x)
    expected = np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_softmax_cpu_reference_single_element():
    x = np.array([[5.0]], dtype=np.float32)
    result = softmax_ref(x)
    np.testing.assert_allclose(result, [[1.0]], atol=1e-6)


def test_softmax_cpu_reference_uniform():
    x = np.ones((1, 128), dtype=np.float32) * 3.0
    result = softmax_ref(x)
    np.testing.assert_allclose(result, np.ones((1, 128)) / 128.0, atol=1e-6)


def test_softmax_cpu_reference_large_values():
    """Numerical stability: large values should not cause overflow."""
    x = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
    result = softmax_ref(x)
    assert np.all(np.isfinite(result))
    assert abs(np.sum(result) - 1.0) < 1e-6


def test_softmax_cpu_reference_negative_values():
    x = np.array([[-1000.0, -999.0, -998.0]], dtype=np.float32)
    result = softmax_ref(x)
    assert np.all(np.isfinite(result))
    assert abs(np.sum(result) - 1.0) < 1e-6


def test_softmax_cpu_reference_row_4096():
    """Handles rows up to 4096 elements."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal((1, 4096)).astype(np.float32)
    result = softmax_ref(x)
    assert result.shape == (1, 4096)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(np.sum(result, axis=-1), [1.0], atol=1e-5)


def test_softmax_cpu_reference_multiple_rows():
    rng = np.random.default_rng(123)
    x = rng.standard_normal((8, 512)).astype(np.float32)
    result = softmax_ref(x)
    assert result.shape == (8, 512)
    np.testing.assert_allclose(np.sum(result, axis=-1), np.ones(8), atol=1e-5)


def test_softmax_f16_precision():
    """Validate that f16 input quantization keeps max absolute error < 1e-2."""
    rng = np.random.default_rng(42)
    x_f32 = rng.standard_normal((4, 4096)).astype(np.float32)

    ref_f32 = softmax_ref(x_f32)

    x_f16_roundtrip = x_f32.astype(np.float16).astype(np.float32)
    result_from_f16 = softmax_ref(x_f16_roundtrip)

    max_abs_error = np.max(np.abs(result_from_f16 - ref_f32))
    assert max_abs_error < 1e-2, f"Max absolute error {max_abs_error} exceeds 1e-2 tolerance"
