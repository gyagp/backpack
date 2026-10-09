// P._pad must be all ones: preserve FP32 product rounding before scale FMA.
enable subgroups;

// Eight rows share two columns. High-half partial storage halves shared
// sums to 8 KiB while retaining the original per-output arithmetic sequence.
// Keep a dynamic bit identity on each product to prevent reassociation.
struct Params { M:u32, N:u32, K:u32, _pad:u32 };
@group(0) @binding(0) var<storage,read> X:array<f32>;
@group(0) @binding(1) var<storage,read> W:array<u32>;
@group(0) @binding(2) var<storage,read> scales:array<u32>;
@group(0) @binding(3) var<storage,read_write> Y:array<f32>;
@group(0) @binding(4) var<uniform> p:Params;
var<workgroup> sums:array<vec4<f32>,512>;
fn scale_at(index:u32)->f32 {
    let pair=unpack2x16float(scales[index>>1u]);
    return select(pair.x,pair.y,(index&1u)!=0u);
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_id) id:vec3<u32>) {
    let lane=id.x;let m0=group.y*8u;let n0=group.x*2u;let blocks=p.K/32u;
    var total0=vec4<f32>(0.0);var total1=vec4<f32>(0.0);
    var total2=vec4<f32>(0.0);var total3=vec4<f32>(0.0);
    for(var tile=0u;tile<p.K;tile+=256u) {
        let k=tile+lane;
        var low=vec4<f32>(0.0);var high=vec4<f32>(0.0);
        for(var r=0u;r<4u;r++) {
            if(m0+r<p.M && k<p.K) { low[r]=X[(m0+r)*p.K+k]; }
            if(m0+r+4u<p.M && k<p.K) { high[r]=X[(m0+r+4u)*p.K+k]; }
        }
        let shift=(k&3u)*8u;let kw=k>>2u;let block=k/32u;
        if(n0<p.N) {
            var q=0.0;var s=0.0;
            if(k<p.K) { q=f32(i32((W[n0*(p.K/4u)+kw]>>shift)&255u)-128);s=scale_at(n0*blocks+block); }
            let productLow=bitcast<vec4<f32>>(bitcast<vec4<u32>>(low*q) & vec4<u32>(p._pad));
            let productHigh=bitcast<vec4<f32>>(bitcast<vec4<u32>>(high*q) & vec4<u32>(p._pad));
            total0=fma(productLow,vec4<f32>(s),total0);
            total1=fma(productHigh,vec4<f32>(s),total1);
        }
        if(n0+1u<p.N) {
            let n=n0+1u;var q=0.0;var s=0.0;
            if(k<p.K) { q=f32(i32((W[n*(p.K/4u)+kw]>>shift)&255u)-128);s=scale_at(n*blocks+block); }
            let productLow=bitcast<vec4<f32>>(bitcast<vec4<u32>>(low*q) & vec4<u32>(p._pad));
            let productHigh=bitcast<vec4<f32>>(bitcast<vec4<u32>>(high*q) & vec4<u32>(p._pad));
            total2=fma(productLow,vec4<f32>(s),total2);
            total3=fma(productHigh,vec4<f32>(s),total3);
        }
    }
    if(lane>=128u) {
        let slot=lane-128u;
        sums[0u+slot]=total0;
        sums[128u+slot]=total1;
        sums[256u+slot]=total2;
        sums[384u+slot]=total3;
    }
    workgroupBarrier();
    if(lane<128u) {
        sums[0u+lane]=total0+sums[0u+lane];
        sums[128u+lane]=total1+sums[128u+lane];
        sums[256u+lane]=total2+sums[256u+lane];
        sums[384u+lane]=total3+sums[384u+lane];
    }
    workgroupBarrier();
    for(var stride=64u;stride>0u;stride>>=1u) {
        if(lane<stride) {
            sums[0u+lane]+=sums[0u+lane+stride];
            sums[128u+lane]+=sums[128u+lane+stride];
            sums[256u+lane]+=sums[256u+lane+stride];
            sums[384u+lane]+=sums[384u+lane+stride];
        }
        workgroupBarrier();
    }
    if(lane==0u) {
        for(var r=0u;r<4u;r++) {
            if(n0+0u<p.N) {
                if(m0+r<p.M) { Y[(m0+r)*p.N+n0+0u]=sums[0][r]; }
                if(m0+r+4u<p.M) { Y[(m0+r+4u)*p.N+n0+0u]=sums[128][r]; }
            }
            if(n0+1u<p.N) {
                if(m0+r<p.M) { Y[(m0+r)*p.N+n0+1u]=sums[256][r]; }
                if(m0+r+4u<p.M) { Y[(m0+r+4u)*p.N+n0+1u]=sums[384][r]; }
            }
        }
    }
}
