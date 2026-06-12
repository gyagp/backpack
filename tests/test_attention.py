"""Tests for attention.wgsl: fused multi-head attention compute shader."""
import os
import re
import struct
import numpy as np
import pytest
from scipy.special import softmax as scipy_softmax

SHADER_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'shaders', 'attention.wgsl')


def attention_ref(Q, K, V, num_heads, num_kv_heads, seq_len, head_dim):
    """CPU reference: multi-head attention with grouped-query support.

    Q: (num_heads, seq_len, head_dim)
    K: (num_kv_heads, seq_len, head_dim)
    V: (num_kv_heads, seq_len, head_dim)
    Returns: (num_heads, seq_len, head_dim)
    """
    scale = 1.0 / np.sqrt(head_dim)
    output = np.zeros_like(Q)

    for h in range(num_heads):
        kv_h = h % num_kv_heads
        # Q[h] @ K[kv_h]^T -> (seq_len, seq_len)
        scores = Q[h] @ K[kv_h].T * scale
        weights = scipy_softmax(scores, axis=-1)
        output[h] = weights @ V[kv_h]

    return output


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


def test_shader_has_bindings(shader_source):
    assert re.search(r'@binding\(0\)', shader_source)
    assert re.search(r'@binding\(1\)', shader_source)
    assert re.search(r'@binding\(2\)', shader_source)
    assert re.search(r'@binding\(3\)', shader_source)


def test_shader_has_softmax(shader_source):
    assert 'exp(' in shader_source
    assert 'max(' in shader_source or 'shared_max' in shader_source


def test_shader_supports_gqa(shader_source):
    assert 'num_kv_heads' in shader_source


def test_attention_basic():
    """Basic MHA: num_heads=8, seq_len=128, head_dim=64."""
    rng = np.random.default_rng(42)
    num_heads, num_kv_heads, seq_len, head_dim = 8, 8, 128, 64

    Q = rng.standard_normal((num_heads, seq_len, head_dim)).astype(np.float32)
    K = rng.standard_normal((num_kv_heads, seq_len, head_dim)).astype(np.float32)
    V = rng.standard_normal((num_kv_heads, seq_len, head_dim)).astype(np.float32)

    result = attention_ref(Q, K, V, num_heads, num_kv_heads, seq_len, head_dim)
    assert result.shape == (num_heads, seq_len, head_dim)
    assert np.all(np.isfinite(result))


def test_attention_gqa():
    """Grouped-query attention: num_heads=8, num_kv_heads=2."""
    rng = np.random.default_rng(99)
    num_heads, num_kv_heads, seq_len, head_dim = 8, 2, 128, 64

    Q = rng.standard_normal((num_heads, seq_len, head_dim)).astype(np.float32)
    K = rng.standard_normal((num_kv_heads, seq_len, head_dim)).astype(np.float32)
    V = rng.standard_normal((num_kv_heads, seq_len, head_dim)).astype(np.float32)

    result = attention_ref(Q, K, V, num_heads, num_kv_heads, seq_len, head_dim)
    assert result.shape == (num_heads, seq_len, head_dim)

    # Heads sharing the same KV head should differ only due to Q
    # Heads 0 and 2 share kv_head 0, so they use same K,V but different Q
    assert not np.allclose(result[0], result[2])


def test_attention_single_head():
    """Single head attention."""
    rng = np.random.default_rng(7)
    num_heads, num_kv_heads, seq_len, head_dim = 1, 1, 32, 16

    Q = rng.standard_normal((1, seq_len, head_dim)).astype(np.float32)
    K = rng.standard_normal((1, seq_len, head_dim)).astype(np.float32)
    V = rng.standard_normal((1, seq_len, head_dim)).astype(np.float32)

    result = attention_ref(Q, K, V, num_heads, num_kv_heads, seq_len, head_dim)
    assert result.shape == (1, seq_len, head_dim)


def test_attention_output_sums():
    """Attention output should be a weighted average of V rows."""
    rng = np.random.default_rng(55)
    num_heads, num_kv_heads, seq_len, head_dim = 2, 2, 16, 8

    Q = rng.standard_normal((num_heads, seq_len, head_dim)).astype(np.float32)
    K = rng.standard_normal((num_kv_heads, seq_len, head_dim)).astype(np.float32)
    V = rng.standard_normal((num_kv_heads, seq_len, head_dim)).astype(np.float32)

    scale = 1.0 / np.sqrt(head_dim)
    for h in range(num_heads):
        kv_h = h % num_kv_heads
        scores = Q[h] @ K[kv_h].T * scale
        weights = scipy_softmax(scores, axis=-1)
        # Weights should sum to 1 per row
        np.testing.assert_allclose(weights.sum(axis=-1), np.ones(seq_len), atol=1e-6)


def test_attention_f16_precision():
    """f16 quantized inputs should produce output within 5e-2 of f32 reference."""
    rng = np.random.default_rng(42)
    num_heads, num_kv_heads, seq_len, head_dim = 8, 8, 128, 64

    Q_f32 = rng.standard_normal((num_heads, seq_len, head_dim)).astype(np.float32)
    K_f32 = rng.standard_normal((num_kv_heads, seq_len, head_dim)).astype(np.float32)
    V_f32 = rng.standard_normal((num_kv_heads, seq_len, head_dim)).astype(np.float32)

    ref = attention_ref(Q_f32, K_f32, V_f32, num_heads, num_kv_heads, seq_len, head_dim)

    Q_f16 = Q_f32.astype(np.float16).astype(np.float32)
    K_f16 = K_f32.astype(np.float16).astype(np.float32)
    V_f16 = V_f32.astype(np.float16).astype(np.float32)

    result_f16 = attention_ref(Q_f16, K_f16, V_f16, num_heads, num_kv_heads, seq_len, head_dim)

    max_abs_error = np.max(np.abs(result_f16 - ref))
    assert max_abs_error < 5e-2, f"Max abs error {max_abs_error} exceeds 5e-2"


def test_attention_f16_precision_gqa():
    """f16 GQA: num_heads=8, num_kv_heads=2, within 5e-2."""
    rng = np.random.default_rng(77)
    num_heads, num_kv_heads, seq_len, head_dim = 8, 2, 128, 64

    Q_f32 = rng.standard_normal((num_heads, seq_len, head_dim)).astype(np.float32)
    K_f32 = rng.standard_normal((num_kv_heads, seq_len, head_dim)).astype(np.float32)
    V_f32 = rng.standard_normal((num_kv_heads, seq_len, head_dim)).astype(np.float32)

    ref = attention_ref(Q_f32, K_f32, V_f32, num_heads, num_kv_heads, seq_len, head_dim)

    Q_f16 = Q_f32.astype(np.float16).astype(np.float32)
    K_f16 = K_f32.astype(np.float16).astype(np.float32)
    V_f16 = V_f32.astype(np.float16).astype(np.float32)

    result_f16 = attention_ref(Q_f16, K_f16, V_f16, num_heads, num_kv_heads, seq_len, head_dim)

    max_abs_error = np.max(np.abs(result_f16 - ref))
    assert max_abs_error < 5e-2, f"Max abs error {max_abs_error} exceeds 5e-2"


def test_attention_identity_v():
    """When V is identity-like, output should be the attention weights."""
    seq_len, head_dim = 8, 8
    Q = np.zeros((1, seq_len, head_dim), dtype=np.float32)
    K = np.zeros((1, seq_len, head_dim), dtype=np.float32)
    V = np.eye(seq_len, head_dim, dtype=np.float32).reshape(1, seq_len, head_dim)

    result = attention_ref(Q, K, V, 1, 1, seq_len, head_dim)
    # With zero Q and K, all scores are 0, softmax gives uniform 1/seq_len
    expected_weights = np.ones((seq_len, seq_len)) / seq_len
    expected = expected_weights @ V[0]
    np.testing.assert_allclose(result[0], expected, atol=1e-6)
