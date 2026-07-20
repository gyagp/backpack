// @meta bindings=4
enable subgroups;

// Normalize each projected PLE slice and combine it with its token signal.
// Grid: (M, n_layers, 1), WG=256.

@group(0) @binding(0) var<storage, read> Proj: array<f32>;
@group(0) @binding(1) var<storage, read> Norm: array<f32>;
@group(0) @binding(2) var<storage, read_write> Signal: array<f32>;
@group(0) @binding(3) var<storage, read> P: array<u32>;
var<workgroup> sums: array<f32, 8>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let M = P[0]; let D = P[1]; let L = P[2];
    let eps = bitcast<f32>(P[3]);
    let row = wid.x; let layer = wid.y; let tid = lid.x;
    if (row >= M || layer >= L) { return; }
    let base = (row * L + layer) * D;
    var ss = 0.0;
    for (var i = tid; i < D; i += 256u) { let v=Proj[base+i]; ss += v*v; }
    let lane=tid&31u; let warp=tid/32u; let ws=subgroupAdd(ss);
    if (lane==0u) { sums[warp]=ws; }
    workgroupBarrier();
    var total=0.0; for (var w=0u; w<8u; w++) { total += sums[w]; }
    let r=inverseSqrt(total/f32(D)+eps);
    for (var i=tid; i<D; i+=256u) {
        let p=base+i;
        Signal[p]=(Proj[p]*r*Norm[i]+Signal[p])*0.7071067811865476;
    }
}
