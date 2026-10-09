#include "native_quant.h"
#include "quants_iq_tables.h"
#include "native_quant_source.h"
#include <stdexcept>

namespace {
struct Layout { uint32_t elements, bytes; const void* codebook; size_t codebookBytes; };
Layout layout(GGUFType type) {
    switch (type) {
        case GGUF_TYPE_Q2_K: return {256, 84, nullptr, 0};
        case GGUF_TYPE_Q3_K: return {256, 110, nullptr, 0};
        case GGUF_TYPE_Q4_K: return {256, 144, nullptr, 0};
        case GGUF_TYPE_Q5_K: return {256, 176, nullptr, 0};
        case GGUF_TYPE_Q6_K: return {256, 210, nullptr, 0};
        case GGUF_TYPE_Q8_0: return {32, 34, nullptr, 0};
        case GGUF_TYPE_IQ2_XXS: return {256, 66, iq2xxs_grid, sizeof(iq2xxs_grid)};
        case GGUF_TYPE_IQ2_XS: return {256, 74, iq2xs_grid, sizeof(iq2xs_grid)};
        case GGUF_TYPE_IQ3_XXS: return {256, 98, iq3xxs_grid, sizeof(iq3xxs_grid)};
        case GGUF_TYPE_IQ1_S: return {256, 50, iq1s_grid, sizeof(iq1s_grid)};
        case GGUF_TYPE_IQ4_NL: return {32, 18, nullptr, 0};
        case GGUF_TYPE_IQ3_S: return {256, 110, iq3s_grid, sizeof(iq3s_grid)};
        case GGUF_TYPE_IQ2_S: return {256, 82, iq2s_grid, sizeof(iq2s_grid)};
        case GGUF_TYPE_IQ4_XS: return {256, 136, nullptr, 0};
        default: return {};
    }
}
}

bool supportsNativeQuant(GGUFType type) { return layout(type).bytes != 0; }

KQuantPacked pack_native_quant(const void* raw, uint32_t rows, uint32_t cols, GGUFType type) {
    const auto spec = layout(type);
    if (!spec.bytes || !rows || !cols || cols % spec.elements)
        throw std::runtime_error("Unsupported native quantization shape/type");
    KQuantPacked out;
    out.N = rows; out.K = cols; out.nBlocks = cols / spec.elements;
    const size_t rowBytes = size_t(out.nBlocks) * spec.bytes;
    out.rowStrideWords = uint32_t((rowBytes + 3) / 4);
    const size_t weightWords = size_t(rows) * out.rowStrideWords;
    out.data.resize(weightWords + spec.codebookBytes / 4, 0);
    for (uint32_t row = 0; row < rows; ++row)
        memcpy(out.data.data() + size_t(row) * out.rowStrideWords,
               static_cast<const uint8_t*>(raw) + size_t(row) * rowBytes, rowBytes);
    if (spec.codebookBytes)
        memcpy(out.data.data() + weightWords, spec.codebook, spec.codebookBytes);
    return out;
}

std::string nativeQuantShader(GGUFType type, bool gather, bool prefill, uint32_t prefillRows, bool alignedU16, bool cacheBlockScale, bool packedIq4Lut) {
    const auto spec = layout(type);
    if (!spec.bytes) throw std::runtime_error("Unsupported native quantization shader");
    std::string source = NATIVE_QUANT_SOURCE;
    if (alignedU16) {
        const std::string old = "fn u16(offset: u32) -> u32 { return u8(offset) | (u8(offset + 1u) << 8u); }";
        const std::string replacement = R"WGSL(fn u16(offset: u32) -> u32 {
    if ((offset & 1u) == 0u) {
        return (W[offset / 4u] >> (8u * (offset & 2u))) & 65535u;
    }
    return u8(offset) | (u8(offset + 1u) << 8u);
})WGSL";
        const auto position = source.find(old);
        if (position == std::string::npos) throw std::runtime_error("Native u16 helper not found");
        source.replace(position, old.size(), replacement);
    }

    if (cacheBlockScale) {
        if (type != GGUF_TYPE_IQ3_S || gather || prefill)
            throw std::runtime_error("Block scale cache is only scalar IQ3_S");
        const std::string marker = "var<workgroup> sums:";
        const auto position = source.find(marker);
        if (position == std::string::npos) throw std::runtime_error("Native reduction marker missing");
        source.insert(position, R"WGSL(fn decode_iq3s_block(base:u32,e:u32,d:f32)->f32 {
    let group=e/32u;let lane=e%32u;
    let index = u8(base+2u+e/4u) | (((u8(base+66u+group)>>(lane/4u))&1u)<<8u);
    let signs = u8(base+74u+e/8u);
    let scale = (u8(base+106u+group/2u)>>(4u*(group%2u)))&15u;
    return d*f32(1u+2u*scale)*f32(code_byte(index,4u,lane%4u))*sign_value(signs,lane%8u);
}

)WGSL");
        const std::string oldLoop = "        for(var k=lane;k<K;k+=32u) { acc+=bitcast<f32>(X[wid.x*K+k])*decode(column,k); }";
        const auto loop = source.find(oldLoop);
        if (loop == std::string::npos) throw std::runtime_error("Native scalar loop missing");
        source.replace(loop,oldLoop.size(),R"WGSL(        let row_base=column*P[3]*4u;
        for(var k0=0u;k0<K;k0+=256u) {
            let block_base=row_base+(k0/256u)*110u;
            let block_scale=half(block_base);
            for(var group=0u;group<8u;group++) {
                let e=group*32u+lane;let k=k0+e;
                if(k<K) { acc+=bitcast<f32>(X[wid.x*K+k])*decode_iq3s_block(block_base,e,block_scale); }
            }
        })WGSL");
    }

    if (packedIq4Lut) {
        if(type!=GGUF_TYPE_IQ4_XS || gather || prefill || cacheBlockScale)
            throw std::runtime_error("Packed IQ4 lookup is only scalar IQ4_XS");
        const std::string old=R"WGSL(fn iq4_value(index: u32) -> f32 {
    let values = array<i32, 16>(-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113);
    return f32(values[index]);
})WGSL";
        const auto position=source.find(old);
        if(position==std::string::npos)throw std::runtime_error("IQ4 lookup marker missing");
        source.replace(position,old.size(),R"WGSL(fn iq4_value(index: u32) -> f32 {
    let low=select(0xbfad9881u,0xf6eaddcfu,(index&4u)!=0u);
    let high=select(0x26190d01u,0x71594535u,(index&4u)!=0u);
    let word=select(low,high,(index&8u)!=0u);
    return f32(signed_byte((word>>(8u*(index&3u)))&255u));
})WGSL");
        const auto reduction=source.find("var<workgroup> sums:");
        if(reduction==std::string::npos)throw std::runtime_error("Native reduction marker missing");
        source.insert(reduction,R"WGSL(fn iq4_dot_factors(row:u32,k:u32)->vec3<f32> {
    let base=row*P[3]*4u+(k/256u)*136u;
    let e=k%256u;let group=e/32u;let lane=e%32u;
    let lo=(u8(base+4u+group/2u)>>(4u*(group%2u)))&15u;
    let hi=(u16(base+2u)>>(2u*group))&3u;
    let q=(u8(base+8u+group*16u+(lane%16u))>>(4u*(lane/16u)))&15u;
    return vec3<f32>(half(base),iq4_value(q),f32(i32(lo|(hi<<4u))-32));
}

)WGSL");
        const std::string oldLoop="        for(var k=lane;k<K;k+=32u) { acc+=bitcast<f32>(X[wid.x*K+k])*decode(column,k); }";
        const auto loop=source.find(oldLoop);
        if(loop==std::string::npos)throw std::runtime_error("Native scalar loop missing");
        source.replace(loop,oldLoop.size(),R"WGSL(        let mask=P[7]; // All ones preserve the accepted compiled multiply boundaries.
        for(var k=lane;k<K;k+=32u) {
            let f=iq4_dot_factors(column,k);
            let xd=bitcast<f32>(bitcast<u32>(bitcast<f32>(X[wid.x*K+k])*f.x)&mask);
            let xdq=bitcast<f32>(bitcast<u32>(xd*f.y)&mask);
            acc=fma(xdq,f.z,acc);
        })WGSL");
    }

    if (prefill) {
        const auto main = source.find("\nvar<workgroup> sums:");
        if (gather || main == std::string::npos)
            throw std::runtime_error("Invalid native prefill shader assembly");
        source.erase(main);
        std::string tile = NATIVE_QUANT_PREFILL_SOURCE;
        if (prefillRows == 32) {
            auto replace = [&](const std::string& from, const std::string& to) {
                const auto p=tile.find(from);
                if(p==std::string::npos)throw std::runtime_error("Native prefill tile transform mismatch");
                tile.replace(p,from.size(),to);
            };
            replace("array<f32, 528>","array<f32, 1056>");
            replace("array<f32, 544>","array<f32, 288>");
            replace("let lm=tid/16u; let ln=tid%16u;","let lm=tid/8u; let ln=tid%8u;");
            replace("let row=wid.x*16u+lm; let col=wid.y*16u+ln;","let row=wid.x*32u+lm; let col=wid.y*8u+ln;");
            const auto begin=tile.find("        for(var i=tid;i<512u;i+=256u) {");
            const auto end=tile.find("        workgroupBarrier();",begin);
            if(begin==std::string::npos || end==std::string::npos)
                throw std::runtime_error("Native prefill tile loader not found");
            tile.replace(begin,end-begin,R"WGSL(        for(var i=tid;i<1024u;i+=256u) {
            let r=i/32u; let k=i%32u; var a=0.0;
            if(wid.x*32u+r<M && k0+k<K) { a=bitcast<f32>(X[(wid.x*32u+r)*K+k0+k]); }
            tileA[r*33u+k]=a;
        }
        let br=tid/32u; let bk=tid%32u; var b=0.0;
        if(wid.y*8u+br<N && k0+bk<K) { b=decode(wid.y*8u+br,k0+bk); }
        tileB[bk*9u+br]=b;
)WGSL");
            size_t p=0;
            while((p=tile.find("*17u",p))!=std::string::npos){tile.replace(p,4,"*9u");p+=3;}
            // Arithmetic below the loads is otherwise unchanged: all 32
            // independent accumulators and the original reduction tree stay.
        } else if (prefillRows != 16) {
            throw std::runtime_error("Unsupported native prefill row count");
        }
        source += tile;
    }
    for (const auto& entry : {std::pair<std::string, std::string>{"__TYPE__", std::to_string(type)},
                             {"__BLOCK_BYTES__", std::to_string(spec.bytes)},
                             {"__BLOCK_ELEMENTS__", std::to_string(spec.elements)},
                             {"__GATHER__", gather ? "true" : "false"}}) {
        size_t offset = 0;
        while ((offset = source.find(entry.first, offset)) != std::string::npos) {
            source.replace(offset, entry.first.size(), entry.second);
            offset += entry.second.size();
        }
    }
    return source;
}

std::string nativeQuantDecodeSliceShader(GGUFType type, bool alignedU16) {
    auto source=nativeQuantShader(type,false,true,32,alignedU16);
    const auto end=source.find("var<workgroup> tileA:");
    if(end==std::string::npos)throw std::runtime_error("Native decode slice splice failed");
    source.erase(end);
    source+=R"WGSL(
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let k=gid.x; let row=gid.y;
    if(k<P[0] && row<P[7]) { Y[row*P[0]+k]=decode(row+P[4],k); }
}
)WGSL";
    return source;
}

std::string nativeQuantDensePrefillShader(bool columnPair, bool alignedWeights, bool alignedActivations, bool transposedPairs) {
    if (transposedPairs) {
        if (!columnPair || !alignedWeights || !alignedActivations)
            throw std::runtime_error("Transposed dense prefill requires paired weights and aligned activations");
        return NATIVE_DENSE_PREFILL_TRANSPOSED_B_SOURCE;
    }
    if(columnPair) {
        if(!alignedWeights)return NATIVE_DENSE_PREFILL_PAIR_SOURCE;
        std::string source=NATIVE_DENSE_PREFILL_PAIR_VEC2_SOURCE;
        // Stride36 preserves the scalar load mapping and exact partial sums.
        // Explicit vector loaders were slower in the same real-weight probes.
        for(const auto& entry : {std::pair<std::string,std::string>{"__A_ELEMENTS__",alignedActivations?"1152":"1056"},
                                {"__A_STRIDE__",alignedActivations?"36":"33"}}) {
            size_t offset=0;
            while((offset=source.find(entry.first,offset))!=std::string::npos) {
                source.replace(offset,entry.first.size(),entry.second);offset+=entry.second.size();
            }
        }
        return source;
    }
    auto source=nativeQuantShader(GGUF_TYPE_IQ3_S,false,true,32);
    auto replace=[&](const std::string& from,const std::string& to) {
        const auto pos=source.find(from);
        if(pos==std::string::npos)throw std::runtime_error("Native dense prefill splice failed");
        source.replace(pos,from.size(),to);
    };
    replace("b=decode(wid.y*8u+br,k0+bk);","b=bitcast<f32>(W[(wid.y*8u+br)*K+k0+bk]);");
    replace("let N=P[1];","let N=P[7];");
    replace("Bias[col]","Bias[col+P[4]]");
    replace("Y[row*stride+col+P[4]]","Y[row*stride+col+P[8]]");
    return source;
}
