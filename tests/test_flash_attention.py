"""Tests for flash_attention.wgsl: Flash-attention style tiled attention kernel."""
import os
import re
import struct
import numpy as np
import pytest

SHADER_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'shaders', 'flash_attention.wgsl')


def attention_ref(Q, K, V, scale):
    """CPU reference: standard scaled dot-product attention.
    Q: [num_heads, seq_len, head_dim]
    K, V: [num_kv_heads, seq_len, head_dim]
    """
    num_heads = Q.shape[0]
    num_kv_heads = K.shape[0]
    seq_len = Q.shape[1]
    head_dim = Q.shape[2]

    output = np.zeros_like(Q)
    for h in range(num_heads):
        kv_h = h % num_kv_heads
        scores = Q[h] @ K[kv_h].T * scale  # [seq_len, seq_len]
        # Numerically stable softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        output[h] = attn_weights @ V[kv_h]
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


def test_shader_has_qkv_bindings(shader_source):
    assert re.search(r'@binding\(0\)\s*var<storage,\s*read>', shader_source), "Missing Q binding"
    assert re.search(r'@binding\(1\)\s*var<storage,\s*read>', shader_source), "Missing K binding"
    assert re.search(r'@binding\(2\)\s*var<storage,\s*read>', shader_source), "Missing V binding"
    assert re.search(r'@binding\(3\)\s*var<storage,\s*read_write>', shader_source), "Missing output binding"


def test_shader_uses_online_softmax(shader_source):
    """Verify the shader uses online softmax (running max + running sum)."""
    assert 'running_max' in shader_source, "Should use running max for online softmax"
    assert 'running_sum' in shader_source, "Should use running sum for online softmax"


def test_shader_has_configurable_tile_size(shader_source):
    assert re.search(r'TILE_SIZE', shader_source), "Should have configurable TILE_SIZE"


def test_shader_no_hardcoded_seqlen_limit(shader_source):
    """Ensure no hardcoded attn_row array that limits sequence length."""
    assert 'attn_row' not in shader_source, "Should not use fixed attn_row array"


def test_cpu_reference_seq512():
    """Validate reference attention for seq_len=512, head_dim=64, 4 heads."""
    np.random.seed(42)
    num_heads, seq_len, head_dim = 4, 512, 64
    num_kv_heads = 4
    scale = 1.0 / np.sqrt(head_dim)

    Q = np.random.randn(num_heads, seq_len, head_dim).astype(np.float32)
    K = np.random.randn(num_kv_heads, seq_len, head_dim).astype(np.float32)
    V = np.random.randn(num_kv_heads, seq_len, head_dim).astype(np.float32)

    out = attention_ref(Q, K, V, scale)
    assert out.shape == (num_heads, seq_len, head_dim)

    # Verify softmax property: attention weights sum to 1
    for h in range(num_heads):
        kv_h = h % num_kv_heads
        scores = Q[h] @ K[kv_h].T * scale
        scores_max = scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores - scores_max)
        weights = weights / weights.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-5)


def test_cpu_reference_seq2048():
    """Validate reference attention for seq_len=2048 (exceeds naive kernel's 1024 limit)."""
    np.random.seed(123)
    num_heads, seq_len, head_dim = 2, 2048, 64
    num_kv_heads = 2
    scale = 1.0 / np.sqrt(head_dim)

    Q = np.random.randn(num_heads, seq_len, head_dim).astype(np.float32)
    K = np.random.randn(num_kv_heads, seq_len, head_dim).astype(np.float32)
    V = np.random.randn(num_kv_heads, seq_len, head_dim).astype(np.float32)

    out = attention_ref(Q, K, V, scale)
    assert out.shape == (num_heads, seq_len, head_dim)

    # Cross-validate: compute again with different method (explicit loop)
    for h in range(num_heads):
        kv_h = h % num_kv_heads
        for q_row in [0, 1023, 2047]:  # spot-check rows
            q_vec = Q[h, q_row]
            scores = (K[kv_h] @ q_vec) * scale
            scores -= scores.max()
            weights = np.exp(scores)
            weights /= weights.sum()
            expected = weights @ V[kv_h]
            np.testing.assert_allclose(out[h, q_row], expected, atol=1e-4)


def test_cpu_reference_gqa():
    """Validate grouped-query attention (num_kv_heads < num_heads)."""
    np.random.seed(99)
    num_heads, seq_len, head_dim = 8, 256, 64
    num_kv_heads = 2
    scale = 1.0 / np.sqrt(head_dim)

    Q = np.random.randn(num_heads, seq_len, head_dim).astype(np.float32)
    K = np.random.randn(num_kv_heads, seq_len, head_dim).astype(np.float32)
    V = np.random.randn(num_kv_heads, seq_len, head_dim).astype(np.float32)

    out = attention_ref(Q, K, V, scale)

    # Heads sharing the same KV head should produce different outputs (different Q)
    # but heads 0 and 4 share kv_head 0
    assert not np.allclose(out[0], out[4], atol=1e-6), "Different Q heads should produce different outputs"

    # Verify shapes
    assert out.shape == (num_heads, seq_len, head_dim)


def test_online_softmax_matches_standard():
    """Verify that online softmax (tiled) matches standard softmax exactly."""
    np.random.seed(77)
    seq_len = 512
    tile_size = 64
    scores = np.random.randn(seq_len).astype(np.float32) * 3.0

    # Standard softmax
    s_max = scores.max()
    exp_s = np.exp(scores - s_max)
    standard_weights = exp_s / exp_s.sum()

    # Online softmax (simulating the kernel's tile-by-tile approach)
    running_max = -np.finfo(np.float32).max
    running_sum = 0.0
    weighted_acc = np.zeros(seq_len, dtype=np.float32)

    num_tiles = (seq_len + tile_size - 1) // tile_size
    for t in range(num_tiles):
        start = t * tile_size
        end = min(start + tile_size, seq_len)
        tile = scores[start:end]

        tile_max = tile.max()
        tile_exp = np.exp(tile - tile_max)
        tile_sum = tile_exp.sum()

        new_max = max(running_max, tile_max)
        correction_old = np.exp(running_max - new_max) if running_sum > 0 else 0.0
        correction_new = np.exp(tile_max - new_max)

        weighted_acc[:] *= correction_old
        for k in range(end - start):
            weighted_acc[start + k] += tile_exp[k] * correction_new

        running_sum = running_sum * correction_old + tile_sum * correction_new
        running_max = new_max

    online_weights = weighted_acc / running_sum
    np.testing.assert_allclose(online_weights, standard_weights, atol=1e-6)


def test_online_softmax_seq2048():
    """Verify online softmax correctness at seq_len=2048."""
    np.random.seed(200)
    seq_len = 2048
    tile_size = 64
    scores = np.random.randn(seq_len).astype(np.float32) * 5.0

    s_max = scores.max()
    standard_weights = np.exp(scores - s_max)
    standard_weights /= standard_weights.sum()

    running_max = -np.finfo(np.float32).max
    running_sum = 0.0
    weighted_acc = np.zeros(seq_len, dtype=np.float32)

    num_tiles = (seq_len + tile_size - 1) // tile_size
    for t in range(num_tiles):
        start = t * tile_size
        end = min(start + tile_size, seq_len)
        tile = scores[start:end]

        tile_max = tile.max()
        tile_exp = np.exp(tile - tile_max)
        tile_sum = tile_exp.sum()

        new_max = max(running_max, tile_max)
        correction_old = np.exp(running_max - new_max) if running_sum > 0 else 0.0
        correction_new = np.exp(tile_max - new_max)

        weighted_acc[:] *= correction_old
        for k in range(end - start):
            weighted_acc[start + k] += tile_exp[k] * correction_new

        running_sum = running_sum * correction_old + tile_sum * correction_new
        running_max = new_max

    online_weights = weighted_acc / running_sum
    np.testing.assert_allclose(online_weights, standard_weights, atol=1e-5)
