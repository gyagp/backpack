// @meta bindings=5
enable subgroups;

// Grid: (num_v_heads, T, 1).
@group(0) @binding(0) var<storage, read> Y: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> Z: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<storage, read> P: array<u32>;
var<workgroup> sums:array<f32,4>;
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid:vec3<u32>,
        @builtin(local_invocation_id) lid:vec3<u32>) {
    let nv=P[0];let dv=P[1];let eps=bitcast<f32>(P[2]);let T=P[3];
    let h=wid.x;let t=wid.y;let d=lid.x;if(h>=nv||t>=T){return;}
    let idx=(t*nv+h)*dv+d;let x=select(0.0,Y[idx],d<dv);
    let ss=subgroupAdd(x*x);if((d&31u)==0u){sums[d/32u]=ss;}workgroupBarrier();
    let scale=inverseSqrt((sums[0]+sums[1]+sums[2]+sums[3])/f32(dv)+eps);
    if(d<dv){let z=Z[idx];O[idx]=x*scale*W[d]*(z/(1.0+exp(-z)));}
}
