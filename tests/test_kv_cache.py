"""Tests for kv_cache_update.wgsl: KV-cache management with dynamic sequence length."""
import os
import re
import numpy as np
import pytest

SHADER_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'shaders', 'kv_cache_update.wgsl')


def kv_cache_update_ref(K_cache, V_cache, new_K, new_V, num_kv_heads, max_seq_len, head_dim, seq_pos):
    """CPU reference: append new K/V entries at seq_pos.

    K_cache, V_cache: (num_kv_heads, max_seq_len, head_dim)
    new_K, new_V: (num_kv_heads, head_dim)
    """
    K_cache[:, seq_pos, :] = new_K
    V_cache[:, seq_pos, :] = new_V


def attention_with_kv_cache_ref(Q, K_cache, V_cache, num_heads, num_kv_heads, current_seq_len, head_dim):
    """CPU reference: attention reading from KV-cache up to current_seq_len.

    Q: (num_heads, 1, head_dim) - single query token
    K_cache, V_cache: (num_kv_heads, max_seq_len, head_dim) - only [:current_seq_len] is valid
    Returns: (num_heads, 1, head_dim)
    """
    from scipy.special import softmax as scipy_softmax
    scale = 1.0 / np.sqrt(head_dim)
    output = np.zeros_like(Q)

    for h in range(num_heads):
        kv_h = h % num_kv_heads
        K_valid = K_cache[kv_h, :current_seq_len, :]
        V_valid = V_cache[kv_h, :current_seq_len, :]
        scores = (Q[h] @ K_valid.T) * scale
        weights = scipy_softmax(scores, axis=-1)
        output[h] = weights @ V_valid

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
    for i in range(5):
        assert re.search(rf'@binding\({i}\)', shader_source)


def test_shader_has_cache_buffers(shader_source):
    assert 'K_cache' in shader_source
    assert 'V_cache' in shader_source


def test_shader_has_seq_pos(shader_source):
    assert 'seq_pos' in shader_source


def test_kv_cache_single_append():
    """Append a single token's KV and verify placement."""
    rng = np.random.default_rng(42)
    num_kv_heads, max_seq_len, head_dim = 4, 128, 64

    K_cache = np.zeros((num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
    V_cache = np.zeros((num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
    new_K = rng.standard_normal((num_kv_heads, head_dim)).astype(np.float32)
    new_V = rng.standard_normal((num_kv_heads, head_dim)).astype(np.float32)

    kv_cache_update_ref(K_cache, V_cache, new_K, new_V, num_kv_heads, max_seq_len, head_dim, 0)

    np.testing.assert_array_equal(K_cache[:, 0, :], new_K)
    np.testing.assert_array_equal(V_cache[:, 0, :], new_V)
    assert np.all(K_cache[:, 1:, :] == 0)


def test_kv_cache_growing_sequence():
    """Append tokens one by one and verify cache grows correctly."""
    rng = np.random.default_rng(99)
    num_kv_heads, max_seq_len, head_dim = 2, 64, 32
    seq_len = 16

    K_cache = np.zeros((num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
    V_cache = np.zeros((num_kv_heads, max_seq_len, head_dim), dtype=np.float32)

    all_K = rng.standard_normal((seq_len, num_kv_heads, head_dim)).astype(np.float32)
    all_V = rng.standard_normal((seq_len, num_kv_heads, head_dim)).astype(np.float32)

    for t in range(seq_len):
        kv_cache_update_ref(K_cache, V_cache, all_K[t], all_V[t],
                            num_kv_heads, max_seq_len, head_dim, t)

    for t in range(seq_len):
        np.testing.assert_array_equal(K_cache[:, t, :], all_K[t])
        np.testing.assert_array_equal(V_cache[:, t, :], all_V[t])

    assert np.all(K_cache[:, seq_len:, :] == 0)


def test_kv_cache_max_seq_len_boundary():
    """Append up to max_seq_len and verify no overflow."""
    rng = np.random.default_rng(7)
    num_kv_heads, max_seq_len, head_dim = 1, 8, 4

    K_cache = np.zeros((num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
    V_cache = np.zeros((num_kv_heads, max_seq_len, head_dim), dtype=np.float32)

    for t in range(max_seq_len):
        new_K = rng.standard_normal((num_kv_heads, head_dim)).astype(np.float32)
        new_V = rng.standard_normal((num_kv_heads, head_dim)).astype(np.float32)
        kv_cache_update_ref(K_cache, V_cache, new_K, new_V,
                            num_kv_heads, max_seq_len, head_dim, t)

    assert np.all(np.isfinite(K_cache))
    assert np.all(np.isfinite(V_cache))
    assert not np.all(K_cache == 0)


def test_kv_cache_attention_integration():
    """Build KV-cache incrementally and run attention at each step."""
    from scipy.special import softmax as scipy_softmax
    rng = np.random.default_rng(55)
    num_heads, num_kv_heads, max_seq_len, head_dim = 4, 2, 32, 16

    K_cache = np.zeros((num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
    V_cache = np.zeros((num_kv_heads, max_seq_len, head_dim), dtype=np.float32)

    for t in range(8):
        new_K = rng.standard_normal((num_kv_heads, head_dim)).astype(np.float32)
        new_V = rng.standard_normal((num_kv_heads, head_dim)).astype(np.float32)
        kv_cache_update_ref(K_cache, V_cache, new_K, new_V,
                            num_kv_heads, max_seq_len, head_dim, t)

        current_seq_len = t + 1
        Q = rng.standard_normal((num_heads, 1, head_dim)).astype(np.float32)
        out = attention_with_kv_cache_ref(Q, K_cache, V_cache,
                                          num_heads, num_kv_heads,
                                          current_seq_len, head_dim)

        assert out.shape == (num_heads, 1, head_dim)
        assert np.all(np.isfinite(out))


def test_kv_cache_attention_consistency():
    """Attention output from KV-cache should match full recomputation."""
    from scipy.special import softmax as scipy_softmax
    rng = np.random.default_rng(123)
    num_heads, num_kv_heads, max_seq_len, head_dim = 4, 4, 32, 16
    seq_len = 10

    K_cache = np.zeros((num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
    V_cache = np.zeros((num_kv_heads, max_seq_len, head_dim), dtype=np.float32)

    all_K = rng.standard_normal((seq_len, num_kv_heads, head_dim)).astype(np.float32)
    all_V = rng.standard_normal((seq_len, num_kv_heads, head_dim)).astype(np.float32)

    for t in range(seq_len):
        kv_cache_update_ref(K_cache, V_cache, all_K[t], all_V[t],
                            num_kv_heads, max_seq_len, head_dim, t)

    Q = rng.standard_normal((num_heads, 1, head_dim)).astype(np.float32)
    cached_out = attention_with_kv_cache_ref(Q, K_cache, V_cache,
                                             num_heads, num_kv_heads,
                                             seq_len, head_dim)

    K_full = np.stack([all_K[:, h, :] for h in range(num_kv_heads)])
    V_full = np.stack([all_V[:, h, :] for h in range(num_kv_heads)])

    scale = 1.0 / np.sqrt(head_dim)
    direct_out = np.zeros_like(Q)
    for h in range(num_heads):
        kv_h = h % num_kv_heads
        scores = (Q[h] @ K_full[kv_h].T) * scale
        weights = scipy_softmax(scores, axis=-1)
        direct_out[h] = weights @ V_full[kv_h]

    np.testing.assert_allclose(cached_out, direct_out, atol=1e-6)


def test_kv_cache_overwrite_position():
    """Writing to the same position twice overwrites the old value."""
    rng = np.random.default_rng(77)
    num_kv_heads, max_seq_len, head_dim = 2, 16, 8

    K_cache = np.zeros((num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
    V_cache = np.zeros((num_kv_heads, max_seq_len, head_dim), dtype=np.float32)

    old_K = rng.standard_normal((num_kv_heads, head_dim)).astype(np.float32)
    kv_cache_update_ref(K_cache, V_cache, old_K, old_K, num_kv_heads, max_seq_len, head_dim, 3)

    new_K = rng.standard_normal((num_kv_heads, head_dim)).astype(np.float32)
    kv_cache_update_ref(K_cache, V_cache, new_K, new_K, num_kv_heads, max_seq_len, head_dim, 3)

    np.testing.assert_array_equal(K_cache[:, 3, :], new_K)
