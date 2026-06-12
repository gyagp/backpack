const TILE_SIZE: u32 = 16u;

struct Params {
    M: u32,
    N: u32,
    K: u32,
    batch_size: u32,
    stride_A: u32,
    stride_B: u32,
    stride_C: u32,
}

@group(0) @binding(0) var<storage, read> A: array<u32>;
@group(0) @binding(1) var<storage, read> B: array<u32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> tileA: array<f32, 256>;  // TILE_SIZE * TILE_SIZE
var<workgroup> tileB: array<f32, 256>;

fn load_f16(buf: ptr<storage, array<u32>, read>, index: u32) -> f32 {
    let word = (*buf)[index / 2u];
    let bits = (word >> ((index % 2u) * 16u)) & 0xFFFFu;
    return unpack2x16float(bits).x;
}

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let batch = wgid.z;
    if batch >= params.batch_size {
        return;
    }

    let row = gid.x;
    let col = gid.y;
    let lr = lid.x;
    let lc = lid.y;

    let a_offset = batch * params.stride_A;
    let b_offset = batch * params.stride_B;
    let c_offset = batch * params.stride_C;

    var sum: f32 = 0.0;
    let numTiles = (params.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        let aCol = t * TILE_SIZE + lc;
        if row < params.M && aCol < params.K {
            tileA[lr * TILE_SIZE + lc] = load_f16(&A, a_offset + row * params.K + aCol);
        } else {
            tileA[lr * TILE_SIZE + lc] = 0.0;
        }

        let bRow = t * TILE_SIZE + lr;
        if bRow < params.K && col < params.N {
            tileB[lr * TILE_SIZE + lc] = load_f16(&B, b_offset + bRow * params.N + col);
        } else {
            tileB[lr * TILE_SIZE + lc] = 0.0;
        }

        workgroupBarrier();

        for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tileA[lr * TILE_SIZE + k] * tileB[k * TILE_SIZE + lc];
        }

        workgroupBarrier();
    }

    if row < params.M && col < params.N {
        C[c_offset + row * params.N + col] = sum;
    }
}
