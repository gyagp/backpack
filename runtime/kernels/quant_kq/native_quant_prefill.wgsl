// @meta noregistry=true
// Appended to the common native_quant decoder. P[6] is the prompt row count.
// Each weight is decoded once per 16x16 output tile and 32-wide K block.
// Keep 32 partial sums to preserve the scalar kernel's reduction order.
var<workgroup> tileA: array<f32, 528>;
// Pad the transposed weight tile: stride 16 maps a loading warp onto only
// two shared-memory banks; stride 17 distributes its writes across all 32.
var<workgroup> tileB: array<f32, 544>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let K=P[0]; let N=P[1]; let M=P[6]; let tid=lid.x;
    let lm=tid/16u; let ln=tid%16u;
    let row=wid.x*16u+lm; let col=wid.y*16u+ln;
    var acc: array<f32,32>;
    for(var k0=0u;k0<K;k0+=32u) {
        for(var i=tid;i<512u;i+=256u) {
            let r=i/32u; let k=i%32u;
            var a=0.0; var b=0.0;
            if(wid.x*16u+r<M && k0+k<K) { a=bitcast<f32>(X[(wid.x*16u+r)*K+k0+k]); }
            if(wid.y*16u+r<N && k0+k<K) { b=decode(wid.y*16u+r,k0+k); }
            tileA[r*33u+k]=a;
            tileB[k*17u+r]=b;
        }
        workgroupBarrier();
        acc[0]+=tileA[lm*33u+0u]*tileB[0u*17u+ln];
        acc[1]+=tileA[lm*33u+1u]*tileB[1u*17u+ln];
        acc[2]+=tileA[lm*33u+2u]*tileB[2u*17u+ln];
        acc[3]+=tileA[lm*33u+3u]*tileB[3u*17u+ln];
        acc[4]+=tileA[lm*33u+4u]*tileB[4u*17u+ln];
        acc[5]+=tileA[lm*33u+5u]*tileB[5u*17u+ln];
        acc[6]+=tileA[lm*33u+6u]*tileB[6u*17u+ln];
        acc[7]+=tileA[lm*33u+7u]*tileB[7u*17u+ln];
        acc[8]+=tileA[lm*33u+8u]*tileB[8u*17u+ln];
        acc[9]+=tileA[lm*33u+9u]*tileB[9u*17u+ln];
        acc[10]+=tileA[lm*33u+10u]*tileB[10u*17u+ln];
        acc[11]+=tileA[lm*33u+11u]*tileB[11u*17u+ln];
        acc[12]+=tileA[lm*33u+12u]*tileB[12u*17u+ln];
        acc[13]+=tileA[lm*33u+13u]*tileB[13u*17u+ln];
        acc[14]+=tileA[lm*33u+14u]*tileB[14u*17u+ln];
        acc[15]+=tileA[lm*33u+15u]*tileB[15u*17u+ln];
        acc[16]+=tileA[lm*33u+16u]*tileB[16u*17u+ln];
        acc[17]+=tileA[lm*33u+17u]*tileB[17u*17u+ln];
        acc[18]+=tileA[lm*33u+18u]*tileB[18u*17u+ln];
        acc[19]+=tileA[lm*33u+19u]*tileB[19u*17u+ln];
        acc[20]+=tileA[lm*33u+20u]*tileB[20u*17u+ln];
        acc[21]+=tileA[lm*33u+21u]*tileB[21u*17u+ln];
        acc[22]+=tileA[lm*33u+22u]*tileB[22u*17u+ln];
        acc[23]+=tileA[lm*33u+23u]*tileB[23u*17u+ln];
        acc[24]+=tileA[lm*33u+24u]*tileB[24u*17u+ln];
        acc[25]+=tileA[lm*33u+25u]*tileB[25u*17u+ln];
        acc[26]+=tileA[lm*33u+26u]*tileB[26u*17u+ln];
        acc[27]+=tileA[lm*33u+27u]*tileB[27u*17u+ln];
        acc[28]+=tileA[lm*33u+28u]*tileB[28u*17u+ln];
        acc[29]+=tileA[lm*33u+29u]*tileB[29u*17u+ln];
        acc[30]+=tileA[lm*33u+30u]*tileB[30u*17u+ln];
        acc[31]+=tileA[lm*33u+31u]*tileB[31u*17u+ln];
        workgroupBarrier();
    }
    acc[0]+=acc[16];
    acc[1]+=acc[17];
    acc[2]+=acc[18];
    acc[3]+=acc[19];
    acc[4]+=acc[20];
    acc[5]+=acc[21];
    acc[6]+=acc[22];
    acc[7]+=acc[23];
    acc[8]+=acc[24];
    acc[9]+=acc[25];
    acc[10]+=acc[26];
    acc[11]+=acc[27];
    acc[12]+=acc[28];
    acc[13]+=acc[29];
    acc[14]+=acc[30];
    acc[15]+=acc[31];
    acc[0]+=acc[8];
    acc[1]+=acc[9];
    acc[2]+=acc[10];
    acc[3]+=acc[11];
    acc[4]+=acc[12];
    acc[5]+=acc[13];
    acc[6]+=acc[14];
    acc[7]+=acc[15];
    acc[0]+=acc[4];
    acc[1]+=acc[5];
    acc[2]+=acc[6];
    acc[3]+=acc[7];
    acc[0]+=acc[2];
    acc[1]+=acc[3];
    acc[0]+=acc[1];
    var stride=N;
    if(P[5]!=0u) { stride=P[5]; }
    if(row<M && col<N) { Y[row*stride+col+P[4]]=acc[0]+Bias[col]; }
}
