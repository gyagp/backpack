// Four prompt rows share Q8 weight/scale loads. Each retains its original
// 256-lane K sequence and reduction tree. Products round before scale FMAs.
struct Params { M:u32, N:u32, K:u32, _pad:u32 };
@group(0) @binding(0) var<storage,read> X:array<f32>;
@group(0) @binding(1) var<storage,read> W:array<u32>;
@group(0) @binding(2) var<storage,read> scales:array<u32>;
@group(0) @binding(3) var<storage,read_write> Y:array<f32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> sums:array<vec4<f32>,1024>;
fn scale_at(index:u32)->f32 {
    let pair=unpack2x16float(scales[index>>1u]);
    return select(pair.x,pair.y,(index&1u)!=0u);
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_id) id:vec3<u32>) {
    let lane=id.x;let m0=group.y*4u;let n0=group.x*4u;let blocks=p.K/32u;
    var total0=vec4<f32>(0.0);var total1=vec4<f32>(0.0);
    var total2=vec4<f32>(0.0);var total3=vec4<f32>(0.0);
    for(var k=lane;k<p.K;k+=256u) {
        var x=vec4<f32>(0.0);
        for(var r=0u;r<4u;r++) { if(m0+r<p.M) { x[r]=X[(m0+r)*p.K+k]; } }
        let shift=(k&3u)*8u;let kw=k>>2u;let block=k/32u;
        if(n0<p.N) {
            let q=f32(i32((W[n0*(p.K/4u)+kw]>>shift)&255u)-128);
            let s=scale_at(n0*blocks+block);total0=fma(x*q,vec4<f32>(s),total0);
        }
        if(n0+1u<p.N) {
            let n=n0+1u;let q=f32(i32((W[n*(p.K/4u)+kw]>>shift)&255u)-128);
            let s=scale_at(n*blocks+block);total1=fma(x*q,vec4<f32>(s),total1);
        }
        if(n0+2u<p.N) {
            let n=n0+2u;let q=f32(i32((W[n*(p.K/4u)+kw]>>shift)&255u)-128);
            let s=scale_at(n*blocks+block);total2=fma(x*q,vec4<f32>(s),total2);
        }
        if(n0+3u<p.N) {
            let n=n0+3u;let q=f32(i32((W[n*(p.K/4u)+kw]>>shift)&255u)-128);
            let s=scale_at(n*blocks+block);total3=fma(x*q,vec4<f32>(s),total3);
        }
    }
    sums[lane]=total0;sums[256u+lane]=total1;sums[512u+lane]=total2;sums[768u+lane]=total3;
    workgroupBarrier();
    for(var stride=128u;stride>0u;stride>>=1u) {
        if(lane<stride) {
            sums[lane]+=sums[lane+stride];sums[256u+lane]+=sums[256u+lane+stride];
            sums[512u+lane]+=sums[512u+lane+stride];sums[768u+lane]+=sums[768u+lane+stride];
        }
        workgroupBarrier();
    }
    if(lane==0u) {
        for(var r=0u;r<4u;r++) {
            if(m0+r<p.M) {
                if(n0<p.N) { Y[(m0+r)*p.N+n0]=sums[0][r]; }
                if(n0+1u<p.N) { Y[(m0+r)*p.N+n0+1u]=sums[256][r]; }
                if(n0+2u<p.N) { Y[(m0+r)*p.N+n0+2u]=sums[512][r]; }
                if(n0+3u<p.N) { Y[(m0+r)*p.N+n0+3u]=sums[768][r]; }
            }
        }
    }
}
