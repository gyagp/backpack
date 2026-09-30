// @meta noregistry=true
// Format-specialized native matmul/gather. All blocks retain the GGUF layout;
// only rows are padded. The codebook begins after N * rowStrideWords.
const TYPE: u32 = __TYPE__u;
const BLOCK_BYTES: u32 = __BLOCK_BYTES__u;
const BLOCK_ELEMENTS: u32 = __BLOCK_ELEMENTS__u;
const GATHER: bool = __GATHER__;
@group(0) @binding(0) var<storage, read> X: array<u32>;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> Y: array<f32>;
// K, N, blocksPerRow, rowStrideWords, outputOffset, outputStride (optional).
// Gather uses P[4] as the float bit pattern of embedding scale instead.
@group(0) @binding(4) var<storage, read> P: array<u32>;

fn u8(offset: u32) -> u32 { return (W[offset / 4u] >> (8u * (offset % 4u))) & 255u; }
fn u16(offset: u32) -> u32 { return u8(offset) | (u8(offset + 1u) << 8u); }
fn u32_at(offset: u32) -> u32 { return u16(offset) | (u16(offset + 2u) << 16u); }
fn half(offset: u32) -> f32 { return unpack2x16float(u16(offset)).x; }
fn signed_byte(value: u32) -> i32 { return i32(value ^ 128u) - 128; }
fn sign_mask(code: u32) -> u32 { return code | ((countOneBits(code) & 1u) << 7u); }
fn sign_value(mask: u32, lane: u32) -> f32 { return select(1.0, -1.0, ((mask >> lane) & 1u) != 0u); }
fn code_byte(index: u32, width: u32, lane: u32) -> u32 {
    return u8(4u * P[1] * P[3] + index * width + lane);
}
fn iq4_value(index: u32) -> f32 {
    let values = array<i32, 16>(-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113);
    return f32(values[index]);
}
fn k_scale(base: u32, group: u32) -> vec2<u32> {
    if (group < 4u) { return vec2<u32>(u8(base+4u+group)&63u, u8(base+8u+group)&63u); }
    let j = group - 4u;
    let low = u8(base+12u+j);
    return vec2<u32>((low&15u)|((u8(base+4u+j)>>2u)&48u),
                     (low>>4u)|((u8(base+8u+j)>>2u)&48u));
}
fn decode(row: u32, k: u32) -> f32 {
    let base = row * P[3] * 4u + (k / BLOCK_ELEMENTS) * BLOCK_BYTES;
    let e = k % BLOCK_ELEMENTS;
    let group = e / 32u;
    let lane = e % 32u;
    if (TYPE == 8u) { return half(base) * f32(signed_byte(u8(base+2u+e))); }
    if (TYPE == 10u) {
        let scale = u8(base + e / 16u);
        let q = (u8(base+16u+(e/128u)*32u+lane) >> (2u*((e%128u)/32u))) & 3u;
        return half(base+80u)*f32(scale&15u)*f32(q) - half(base+82u)*f32(scale>>4u);
    }
    if (TYPE == 11u) {
        let g = e / 16u;
        let lo = (u8(base+96u+(g%8u)) >> (4u*(g/8u))) & 15u;
        let hi = (u8(base+104u+(g%4u)) >> (2u*(g/4u))) & 3u;
        let scale = i32(lo | (hi<<4u)) - 32;
        let low = (u8(base+32u+(e/128u)*32u+lane) >> (2u*((e%128u)/32u))) & 3u;
        let high = (u8(base+lane) >> group) & 1u;
        return half(base+108u)*f32(scale)*f32(i32(low)-select(4,0,high!=0u));
    }
    if (TYPE == 12u || TYPE == 13u) {
        let sm = k_scale(base, group);
        let qoffset = select(16u,48u,TYPE==13u);
        var q = (u8(base+qoffset+(group/2u)*32u+lane) >> (4u*(group%2u))) & 15u;
        if (TYPE==13u) { q |= ((u8(base+16u+lane)>>group)&1u)<<4u; }
        return half(base)*f32(sm.x)*f32(q) - half(base+2u)*f32(sm.y);
    }
    if (TYPE == 14u) {
        let section = e / 128u;
        let part = (e % 128u) / 32u;
        let ql = (u8(base+section*64u+(part%2u)*32u+lane) >> (4u*(part/2u))) & 15u;
        let qh = (u8(base+128u+section*32u+lane) >> (2u*part)) & 3u;
        return half(base+208u)*f32(signed_byte(u8(base+192u+e/16u)))*f32(i32(ql|(qh<<4u))-32);
    }
    if (TYPE == 16u) {
        let high = u32_at(base+6u+group*8u);
        let index = u8(base+2u+group*8u+lane/8u);
        let scale = half(base)*(0.5+f32(high>>28u))*0.25;
        return scale*f32(code_byte(index,8u,lane%8u))*sign_value(sign_mask((high>>(7u*(lane/8u)))&127u),lane%8u);
    }
    if (TYPE == 17u) {
        let q = u16(base+2u+(e/8u)*2u);
        let s = (u8(base+66u+group) >> (4u*(lane/16u))) & 15u;
        return half(base)*(0.5+f32(s))*0.25*f32(code_byte(q&511u,8u,lane%8u))*sign_value(sign_mask(q>>9u),lane%8u);
    }
    if (TYPE == 18u) {
        let high = u32_at(base+66u+group*4u);
        let index = u8(base+2u+e/4u);
        let scale = half(base)*(0.5+f32(high>>28u))*0.5;
        return scale*f32(code_byte(index,4u,lane%4u))*sign_value(sign_mask((high>>(7u*(lane/8u)))&127u),lane%8u);
    }
    if (TYPE == 19u) {
        let high = u16(base+34u+group*2u);
        let index = u8(base+2u+e/8u) | (((high>>(3u*(lane/8u)))&7u)<<8u);
        let scale = half(base)*f32(2u*((high>>12u)&7u)+1u);
        let delta = select(0.125,-0.125,(high&32768u)!=0u);
        return scale*(f32(signed_byte(code_byte(index,8u,lane%8u)))+delta);
    }
    if (TYPE == 20u) {
        let q = (u8(base+2u+(e%16u)) >> (4u*(e/16u))) & 15u;
        return half(base)*iq4_value(q);
    }
    if (TYPE == 21u) {
        let index = u8(base+2u+e/4u) | (((u8(base+66u+group)>>(lane/4u))&1u)<<8u);
        let signs = u8(base+74u+e/8u);
        let scale = (u8(base+106u+group/2u)>>(4u*(group%2u)))&15u;
        return half(base)*f32(1u+2u*scale)*f32(code_byte(index,4u,lane%4u))*sign_value(signs,lane%8u);
    }
    if (TYPE == 22u) {
        let index = u8(base+2u+e/8u) | (((u8(base+66u+group)>>(2u*(lane/8u)))&3u)<<8u);
        let signs = u8(base+34u+e/8u);
        let scale = (u8(base+74u+group)>>(4u*(lane/16u)))&15u;
        return half(base)*(0.5+f32(scale))*0.25*f32(code_byte(index,8u,lane%8u))*sign_value(signs,lane%8u);
    }
    // IQ4_XS
    let lo = (u8(base+4u+group/2u)>>(4u*(group%2u)))&15u;
    let hi = (u16(base+2u)>>(2u*group))&3u;
    let q = (u8(base+8u+group*16u+(lane%16u))>>(4u*(lane/16u)))&15u;
    return half(base)*f32(i32(lo|(hi<<4u))-32)*iq4_value(q);
}

var<workgroup> sums: array<f32,256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let K=P[0]; let N=P[1]; let tid=lid.x;
    if (GATHER) {
        let k=wid.x*256u+tid; let token=X[wid.y];
        if (k<K && token<N) { Y[wid.y*K+k]=decode(token,k)*bitcast<f32>(P[4]); }
        return;
    }
    let column=wid.y*8u+tid/32u; let lane=tid%32u;
    var acc=0.0;
    if (column<N) {
        for(var k=lane;k<K;k+=32u) { acc+=bitcast<f32>(X[wid.x*K+k])*decode(column,k); }
    }
    sums[tid]=acc; workgroupBarrier();
    for(var offset=16u;offset>0u;offset/=2u) {
        if(lane<offset) { sums[tid]+=sums[tid+offset]; }
        workgroupBarrier();
    }
    var stride=N;
    // The runtime pads parameter blocks to 16-byte boundaries; a padded zero
    // is not an explicit stride. Keep contiguous rows for ordinary matmul.
    if (arrayLength(&P)>5u && P[5]!=0u) { stride=P[5]; }
    if(lane==0u && column<N) { Y[wid.x*stride+column+P[4]]=sums[tid]+Bias[column]; }
}
