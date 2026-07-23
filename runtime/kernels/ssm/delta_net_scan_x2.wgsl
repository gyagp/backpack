// @meta bindings=8
enable subgroups;

// Qwen 3.5 DeltaNet prefill scan. A workgroup owns two V columns and advances
// them through all prompt rows, preserving the exact decode recurrence.
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read> Beta: array<f32>;
@group(0) @binding(4) var<storage, read> Gate: array<f32>;
@group(0) @binding(5) var<storage, read_write> State: array<f32>;
@group(0) @binding(6) var<storage, read_write> Y: array<f32>;
@group(0) @binding(7) var<storage, read> P: array<u32>;

var<workgroup> sums: array<f32, 8>;
fn reduce128(x:f32, lane128:u32, pair:u32)->f32 {
    let lane=lane128&31u; let warp=(lane128>>5u)+pair*4u;
    let s=subgroupAdd(x); if(lane==0u){sums[warp]=s;} workgroupBarrier();
    let b=pair*4u; let total=sums[b]+sums[b+1u]+sums[b+2u]+sums[b+3u];
    workgroupBarrier(); return total;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid:vec3<u32>,
        @builtin(local_invocation_id) lid:vec3<u32>) {
    let nv=P[0]; let nk=P[1]; let dk=P[2]; let dv=P[3]; let T=P[4];
    let head=wid.x; let pair=lid.x>>7u; let ki=lid.x&127u;
    let vi=wid.y*2u+pair; let valid=head<nv&&vi<dv;
    let kh=head/max(1u,nv/nk); let sb=head*dk*dv; let qs=inverseSqrt(f32(dk));
    for(var t=0u;t<T;t++) {
        let qbase=(t*nk+kh)*dk; let vbase=(t*nv+head)*dv;
        let bh=select(0.0,Beta[t*nv+head],valid);
        let gh=select(0.0,exp(Gate[t*nv+head]),valid);
        let kv=select(0.0,K[qbase+ki],valid&&ki<dk);
        let qv=select(0.0,Q[qbase+ki]*qs,valid&&ki<dk);
        let idx=sb+ki*dv+vi;
        let old=select(0.0,State[idx],valid&&ki<dk);
        let pred=reduce128(gh*old*kv,ki,pair);
        let delta=select(0.0,(V[vbase+vi]-pred)*bh,valid);
        let sn=gh*old+kv*delta;
        if(valid&&ki<dk){State[idx]=sn;}
        let out=reduce128(sn*qv,ki,pair);
        if(ki==0u&&valid){Y[vbase+vi]=out;}
    }
}
