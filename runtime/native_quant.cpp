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

std::string nativeQuantShader(GGUFType type, bool gather, bool prefill, uint32_t prefillRows) {
    const auto spec = layout(type);
    if (!spec.bytes) throw std::runtime_error("Unsupported native quantization shader");
    std::string source = NATIVE_QUANT_SOURCE;
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

std::string nativeQuantDecodeSliceShader(GGUFType type) {
    auto source=nativeQuantShader(type,false,true,32);
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

std::string nativeQuantDensePrefillShader(bool columnPair) {
    if(columnPair)return NATIVE_DENSE_PREFILL_PAIR_SOURCE;
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
