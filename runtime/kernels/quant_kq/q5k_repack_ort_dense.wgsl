// Convert raw GGUF Q5_K blocks to one byte per logical weight plus exact
// affine scale/min pairs for the ORT-style 64x64 DP4A tile.
@group(0) @binding(0) var<storage, read> Raw: array<u32>;
@group(0) @binding(1) var<storage, read_write> Dense: array<u32>;
@group(0) @binding(2) var<storage, read_write> ScaleMin: array<f32>;
@group(0) @binding(3) var<storage, read> P: array<u32>;

fn pack4Bytes(low: u32, highBits: u32, subBlock: u32) -> u32 {
    var result = 0u;
    for (var i = 0u; i < 4u; i++) {
        let lo = (low >> (i * 8u)) & 15u;
        let hi = ((highBits >> (i * 8u + subBlock)) & 1u) << 4u;
        result |= (lo | hi) << (i * 8u);
    }
    return result;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let K=P[0]; let N=P[1]; let blocksPerRow=P[2]; let rowStrideWords=P[3];
    let groupsPerRow=K/32u; let group=gid.x;
    if(group>=N*groupsPerRow){return;}
    let row=group/groupsPerRow; let gr=group%groupsPerRow;
    let block=gr/8u; let sb=gr&7u; if(block>=blocksPerRow){return;}
    let base=row*rowStrideWords+block*44u; let qgroup=sb/2u;
    let highNibble=(sb&1u)!=0u; let qbase=base+12u+qgroup*8u;
    for(var part=0u;part<8u;part++){
        let packed=Raw[qbase+part];
        let low=select(packed&0x0F0F0F0Fu,(packed>>4u)&0x0F0F0F0Fu,highNibble);
        let highBits=Raw[base+4u+part];
        Dense[group*8u+part]=pack4Bytes(low,highBits,sb);
    }
    let dm=unpack2x16float(Raw[base]);let shift=(sb&3u)*8u;
    let dv=(Raw[base+1u]>>shift)&255u;let mv=(Raw[base+2u]>>shift)&255u;
    var scale:u32;var minValue:u32;
    if(sb<4u){scale=dv&63u;minValue=mv&63u;}
    else{let hi=(Raw[base+3u]>>shift)&255u;scale=(hi&15u)|((dv>>2u)&48u);minValue=(hi>>4u)|((mv>>2u)&48u);}
    ScaleMin[group*2u]=dm.x*f32(scale);ScaleMin[group*2u+1u]=dm.y*f32(minValue);
}
