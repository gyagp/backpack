// @meta bindings=5
enable subgroups;

// Batched split and per-head L2 normalization. Grid: (3, max(nk,nv), T).
@group(0) @binding(0) var<storage, read> C: array<f32>;
@group(0) @binding(1) var<storage, read_write> Q: array<f32>;
@group(0) @binding(2) var<storage, read_write> K: array<f32>;
@group(0) @binding(3) var<storage, read_write> V: array<f32>;
@group(0) @binding(4) var<storage, read> P: array<u32>;

var<workgroup> sums: array<f32, 4>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let nk=P[0]; let nv=P[1]; let dk=P[2]; let dv=P[3];
    let eps=bitcast<f32>(P[4]); let T=P[5];
    let kind=wid.x; let h=wid.y; let t=wid.z; let d=lid.x;
    if (t >= T) { return; }
    let qsize=nk*dk; let channels=2u*qsize+nv*dv;
    if (kind == 2u) {
        if (h >= nv) { return; }
        if (d < dv) { V[(t*nv+h)*dv+d] = C[t*channels+2u*qsize+h*dv+d]; }
        return;
    }
    if (h >= nk) { return; }
    let src=t*channels+kind*qsize+h*dk;
    let x=select(0.0,C[src+d],d<dk);
    let ss=subgroupAdd(x*x);
    if ((d&31u)==0u) { sums[d/32u]=ss; }
    workgroupBarrier();
    let inv=1.0/max(sqrt(sums[0]+sums[1]+sums[2]+sums[3]),eps);
    if (d < dk) {
        let dst=(t*nk+h)*dk+d;
        if (kind==0u) { Q[dst]=x*inv; } else { K[dst]=x*inv; }
    }
}
