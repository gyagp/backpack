import os
import sys

import numpy as np


FEATURE_ONLY_SHADER = """
enable subgroups;
enable chromium_experimental_subgroup_matrix;

@group(0) @binding(0) var<storage, read_write> Out: array<f32>;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    Out[lid.x] = 1.0;
}
"""


TYPE_DECL_SHADER = """
enable subgroups;
enable chromium_experimental_subgroup_matrix;

@group(0) @binding(0) var<storage, read_write> Out: array<f32>;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    var matA: subgroup_matrix_left<f32, 8, 8>;
    var matB: subgroup_matrix_right<f32, 8, 8>;
    var matC: subgroup_matrix_result<f32, 8, 8>;
    Out[lid.x] = 0.0;
}
"""


MMA_SHADER = """
enable subgroups;
enable chromium_experimental_subgroup_matrix;

@group(0) @binding(0) var<storage, read_write> Out: array<f32>;

var<workgroup> TileA: array<f32, 64>;
var<workgroup> TileB: array<f32, 64>;
var<workgroup> TileC: array<f32, 64>;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = lid.x;
    TileA[idx] = 1.0;
    TileA[idx + 32u] = 1.0;
    TileB[idx] = 1.0;
    TileB[idx + 32u] = 1.0;
    TileC[idx] = 0.0;
    TileC[idx + 32u] = 0.0;
    workgroupBarrier();

    let matA = subgroupMatrixLoad<subgroup_matrix_left<f32, 8, 8>>(
        &TileA, 0u, false, 8u);
    let matB = subgroupMatrixLoad<subgroup_matrix_right<f32, 8, 8>>(
        &TileB, 0u, true, 8u);
    var matC: subgroup_matrix_result<f32, 8, 8>;
    matC = subgroupMatrixMultiplyAccumulate(matA, matB, matC);
    subgroupMatrixStore(&TileC, 0u, matC, false, 8u);
    workgroupBarrier();

    Out[idx] = TileC[idx];
    Out[idx + 32u] = TileC[idx + 32u];
}
"""


def main():
    os.environ.setdefault('DAWN_BACKEND', 'vulkan')

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo_root)
    sys.path.insert(0, os.path.join(repo_root, 'third_party', 'triton-windows', 'python'))

    from triton.backends.webgpu.dawn_runner import DawnRunner, BufferBinding

    runner = DawnRunner()
    binding = [BufferBinding(name='Out', binding=0, access='read_write', elem_type='f32')]

    print('Adapter:', runner.adapter_info)
    print('ChromiumExperimentalSubgroupMatrix:', runner.has_subgroup_matrix)

    print('\n[1/3] Feature-only shader')
    pipeline, _ = runner.get_pipeline_info(FEATURE_ONLY_SHADER, binding, [])
    print('Compiled:', bool(pipeline))

    print('\n[2/3] Type declaration shader')
    try:
        pipeline, _ = runner.get_pipeline_info(TYPE_DECL_SHADER, binding, [])
        print('Unexpectedly compiled:', bool(pipeline))
    except Exception as exc:
        print('Observed failure type:', type(exc).__name__)
        print('Observed failure:')
        print(str(exc))

    print('\n[3/3] Load + MMA + Store shader')
    try:
        pipeline, _ = runner.get_pipeline_info(MMA_SHADER, binding, [])
        print('Unexpectedly compiled:', bool(pipeline))

        out = runner.run_kernel(
            wgsl_code=MMA_SHADER,
            buffer_bindings=binding,
            param_fields=[],
            workgroup_size=32,
            grid=(1,),
            buffers={'Out': np.zeros(64, dtype=np.float32)},
            scalars={},
        )
        print('Unexpectedly ran; first 8 outputs:', out['Out'][:8])
        return 1
    except Exception as exc:
        print('Observed failure type:', type(exc).__name__)
        print('Observed failure:')
        print(str(exc))
        return 0


if __name__ == '__main__':
    raise SystemExit(main())