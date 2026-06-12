"""Tests for embed_f16_to_f32.wgsl and embedding_lookup in inference.h."""
import os
import re
import struct
import pytest
import numpy as np

SHADER_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'shaders', 'embed_f16_to_f32.wgsl')
INFERENCE_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'inference.h')


@pytest.fixture
def shader_source():
    with open(SHADER_PATH) as f:
        return f.read()


@pytest.fixture
def inference_source():
    with open(INFERENCE_PATH) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Shader structural tests
# ---------------------------------------------------------------------------

def test_shader_exists():
    assert os.path.exists(SHADER_PATH)


def test_shader_has_input_binding(shader_source):
    assert re.search(r'@binding\(0\)\s+var<storage,\s*read>\s+input\s*:\s*array<u32>', shader_source)


def test_shader_has_output_binding(shader_source):
    assert re.search(r'@binding\(1\)\s+var<storage,\s*read_write>\s+output\s*:\s*array<f32>', shader_source)


def test_shader_has_params_binding(shader_source):
    assert re.search(r'@binding\(2\)\s+var<uniform>\s+params\s*:\s*Params', shader_source)


def test_shader_workgroup_size_64(shader_source):
    assert re.search(r'@workgroup_size\(64\)', shader_source)


def test_shader_has_bounds_check(shader_source):
    assert re.search(r'if\s*\(\s*i\s*>=\s*params\.count\s*\)', shader_source)


# ---------------------------------------------------------------------------
# embedding_lookup uses GPU-side operations (no CPU readback)
# ---------------------------------------------------------------------------

def test_f32_path_uses_copy_buffer_to_buffer(inference_source):
    """f32 embedding lookup uses CopyBufferToBuffer (GPU-side copy)."""
    f32_block = re.search(
        r'if\s*\(embed_table\.dtype\s*==\s*DType::f32\)\s*\{(.*?)\}',
        inference_source, re.DOTALL
    )
    assert f32_block, "f32 branch not found"
    body = f32_block.group(1)
    assert 'CopyBufferToBuffer' in body
    assert 'MapAsync' not in body
    assert 'GetMappedRange' not in body


def test_f16_path_uses_compute_shader(inference_source):
    """f16 embedding lookup uses a GPU compute shader dispatch."""
    f16_block = re.search(
        r'else if\s*\(embed_table\.dtype\s*==\s*DType::f16\)\s*\{(.*?)\}\s*else',
        inference_source, re.DOTALL
    )
    assert f16_block, "f16 branch not found"
    body = f16_block.group(1)
    assert 'embed_f16_to_f32.wgsl' in body
    assert 'DispatchWorkgroups' in body
    assert 'MapAsync' not in body
    assert 'GetMappedRange' not in body


def test_no_cpu_readback_in_embedding_lookup(inference_source):
    """The entire embedding_lookup function has no CPU readback calls."""
    func_match = re.search(
        r'inline wgpu::Buffer embedding_lookup\(.*?\n\}',
        inference_source, re.DOTALL
    )
    assert func_match, "embedding_lookup function not found"
    body = func_match.group(0)
    assert 'MapAsync' not in body, "CPU readback (MapAsync) found in embedding_lookup"
    assert 'GetMappedRange' not in body, "CPU readback (GetMappedRange) found in embedding_lookup"


# ---------------------------------------------------------------------------
# f16_to_f32 numerical correctness (reference implementation)
# ---------------------------------------------------------------------------

def f16_to_f32_reference(bits):
    """Python reference for the WGSL f16_to_f32 function."""
    sign = (bits >> 15) & 1
    exp = (bits >> 10) & 0x1F
    frac = bits & 0x3FF

    if exp == 0:
        if frac == 0:
            f32_bits = sign << 31
        else:
            e = 1
            f = frac
            while (f & 0x400) == 0:
                f = f << 1
                e = e - 1
            f = f & 0x3FF
            f32_bits = (sign << 31) | ((e + 112) << 23) | (f << 13)
    elif exp == 31:
        f32_bits = (sign << 31) | 0x7F800000 | (frac << 13)
    else:
        f32_bits = (sign << 31) | ((exp + 112) << 23) | (frac << 13)

    return struct.unpack('f', struct.pack('I', f32_bits))[0]


@pytest.mark.parametrize("f16_bits,expected_f32", [
    (0x0000, 0.0),
    (0x8000, -0.0),
    (0x3C00, 1.0),
    (0xBC00, -1.0),
    (0x4000, 2.0),
    (0x3800, 0.5),
    (0x4200, 3.0),
])
def test_f16_to_f32_known_values(f16_bits, expected_f32):
    result = f16_to_f32_reference(f16_bits)
    assert result == pytest.approx(expected_f32, abs=1e-7)


def test_f16_to_f32_matches_numpy():
    """Reference implementation matches numpy's f16 conversion for random values."""
    rng = np.random.default_rng(42)
    values = rng.uniform(-10.0, 10.0, size=100).astype(np.float16)
    for v in values:
        bits = int(np.frombuffer(v.tobytes(), dtype=np.uint16)[0])
        ref = f16_to_f32_reference(bits)
        expected = float(v)
        if np.isnan(expected):
            assert np.isnan(ref)
        else:
            assert ref == pytest.approx(expected, abs=1e-3)


def test_f16_to_f32_subnormals():
    """Subnormal f16 values are handled correctly."""
    result = f16_to_f32_reference(0x0001)
    assert result > 0.0
    assert result < 1e-6
