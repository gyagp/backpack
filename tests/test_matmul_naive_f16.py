"""Tests for matmul_naive_f16.wgsl acceptance criteria."""
import os
import re
import pytest

SHADER_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'shaders', 'matmul_naive_f16.wgsl')

@pytest.fixture
def shader_source():
    assert os.path.exists(SHADER_PATH), f"Shader file not found: {SHADER_PATH}"
    with open(SHADER_PATH) as f:
        return f.read()

# AC1: src/shaders/matmul_naive_f16.wgsl exists
def test_shader_file_exists():
    assert os.path.isfile(SHADER_PATH)

# AC2: Each thread computes one output element via dot product
def test_one_thread_per_output_element(shader_source):
    assert '@compute' in shader_source
    assert '@workgroup_size' in shader_source
    assert 'global_invocation_id' in shader_source
    # Thread computes row/col from global_invocation_id
    assert re.search(r'row\s*=\s*gid\.\w', shader_source)
    assert re.search(r'col\s*=\s*gid\.\w', shader_source)
    # Dot product accumulation loop over K
    assert re.search(r'for\s*\(.*k.*params\.K', shader_source)
    assert 'sum' in shader_source or 'acc' in shader_source

def test_single_output_write_per_thread(shader_source):
    # Output should be f32 array (no race condition from packed f16)
    assert 'array<f32>' in shader_source
    # Single store: C[row * N + col] = sum
    assert re.search(r'C\[row\s*\*\s*params\.N\s*\+\s*col\]', shader_source)

# AC3: Handles arbitrary M, N, K dimensions via uniform params
def test_uniform_params(shader_source):
    assert 'var<uniform>' in shader_source
    assert re.search(r'M\s*:\s*u32', shader_source)
    assert re.search(r'N\s*:\s*u32', shader_source)
    assert re.search(r'K\s*:\s*u32', shader_source)

def test_bounds_check(shader_source):
    # Must guard against out-of-bounds threads
    assert re.search(r'row\s*>=\s*params\.M', shader_source)
    assert re.search(r'col\s*>=\s*params\.N', shader_source)

# Verify no f16 store race condition
def test_no_packed_f16_output(shader_source):
    # Output buffer must NOT be array<u32> with manual f16 packing (race condition)
    # Binding 2 (output) should be array<f32>
    bindings = re.findall(r'@binding\(2\)\s*var<storage,\s*read_write>\s*\w+\s*:\s*(\w+<\w+>)', shader_source)
    assert len(bindings) == 1
    assert bindings[0] == 'array<f32>', f"Output buffer type is {bindings[0]}, expected array<f32>"

def test_f16_input_loading(shader_source):
    # Should load f16 from packed u32 input buffers
    assert 'load_f16' in shader_source or 'unpack2x16float' in shader_source
