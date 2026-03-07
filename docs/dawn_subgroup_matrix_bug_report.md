**Summary**
The original repro was using `f32` input subgroup-matrix types with an `8x8x8` shape on Vulkan. On the tested NVIDIA adapter, Dawn advertises subgroup-matrix support, but only for these input types and shapes:

- `f16` -> `f16` with `16x16x16`, `16x8x16`, `16x8x8`
- `f16` -> `f32` with `16x16x16`, `16x8x16`, `16x8x8`
- `u8` -> `u32` with `16x16x32`, `16x8x32`
- `i8` -> `i32` with `16x16x32`, `16x8x32`

There is no advertised `f32` input configuration. The failure `Unknown configuration is M(8), N(0), K(8), f32` is therefore a correctly rejected unsupported configuration, not a validated Dawn bug.

**Environment**
Windows
Vulkan backend through Dawn
NVIDIA GPU
Experimental Dawn feature exposed as `ChromiumExperimentalSubgroupMatrix`
Vendored Dawn commit: `62d81282211150ffc0baa8680841718f761be3b5`
Adapter from repro:
`{'vendor': 'nvidia', 'architecture': 'blackwell', 'device': 'NVIDIA GeForce RTX 5080', 'description': 'NVIDIA: 591.44 591.44.0.0', 'backend': 'Vulkan', 'vendorID': 4318, 'deviceID': 11266}`

**Repro**
Primary raw-Dawn repro script: [docs/dawn_subgroup_matrix_raw_repro.py](docs/dawn_subgroup_matrix_raw_repro.py)

Wrapper-based cross-check repro: [docs/dawn_subgroup_matrix_min_repro.py](docs/dawn_subgroup_matrix_min_repro.py)

C++ outline for Dawn-native debugging: [docs/dawn_subgroup_matrix_cpp_repro_outline.cc](docs/dawn_subgroup_matrix_cpp_repro_outline.cc)

Run:

```powershell
python docs/dawn_subgroup_matrix_raw_repro.py
```

The script does four things:
1. Queries and prints the adapter's advertised subgroup-matrix configs.
2. Compiles a baseline shader with `enable chromium_experimental_subgroup_matrix` but no subgroup-matrix types.
3. Compiles a shader that only declares `subgroup_matrix_left`, `subgroup_matrix_right`, and `subgroup_matrix_result` variables.
4. Compiles a minimal shader that uses fixed-size `subgroupMatrixLoad`, `subgroupMatrixMultiplyAccumulate`, and `subgroupMatrixStore` on `8x8` `f32` matrices in workgroup memory.

This repro does not use `DawnRunner` or any Triton runtime wrapper. It calls the Dawn C API directly via `ctypes`.

**Expected Result**
The baseline shader should compile. The `f32` subgroup-matrix shaders should fail unless the adapter explicitly advertises `f32` input configs for the requested shape.

**Actual Result**
Pipeline creation fails before execution with an error like:

```text
Failed to create compute pipeline (async): Subgroup matrix usage found which is not supported by the device.
Unknown configuration is M(8), N(0), K(8), f32
```

The zero dimension is expected Tint metadata for directional matrix operands, not proof of reflection corruption. For example, a `subgroup_matrix_left<T, K, M>` records `M` and `K`, leaving `N=0` because `N` is not meaningful for the left operand type.

**Unsupported WGSL Example**
This is the intentionally unsupported type-declaration shader from the repro:

```wgsl
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
```

And this is the failing load/MMA/store shader:

```wgsl
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
```

**Validation Result**
On the same adapter and backend, Dawn successfully compiles supported subgroup-matrix type declarations such as:

```wgsl
enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;

@group(0) @binding(0) var<storage, read_write> Out: array<f32>;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    var matA: subgroup_matrix_left<f16, 16, 16>;
    var matB: subgroup_matrix_right<f16, 16, 16>;
    var matC: subgroup_matrix_result<f32, 16, 16>;
    Out[lid.x] = 0.0;
}
```

That confirms Dawn validation is behaving correctly for supported configurations on this adapter.