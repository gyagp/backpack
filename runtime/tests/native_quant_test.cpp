#include <cstdlib>
#include <cstring>
#include "gpu_context.h"
#include "native_quant.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>

static void checkFields(GPUContext& gpu) {
    std::vector<uint8_t> bytes(1024);for(size_t i=0;i<bytes.size();++i)bytes[i]=uint8_t(i*73+(i>>3)*37);
    std::vector<uint32_t> poison(1023,0x7fc12345u);const uint32_t params[]={1023,0,0,0};
    auto w=gpu.createBuffer("field_bytes",bytes.size()),y=gpu.createBuffer("field_output",poison.size()*4),p=gpu.createBuffer("field_params",sizeof(params));
    gpu.writeBuffer(w,bytes.data(),bytes.size());gpu.writeBuffer(p,params,sizeof(params));
    for(bool aligned:{false,true}) {
        auto source=nativeQuantShader(GGUF_TYPE_IQ3_S,false,false,16,aligned);
        const auto end=source.find("\nvar<workgroup> sums:");if(end==std::string::npos)throw std::runtime_error("Field-contract splice failed");
        source.erase(end);source+=R"WGSL(
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {if(id.x<P[0]){Y[id.x]=f32(u16(id.x));}}
)WGSL";
        auto& pipeline=gpu.getOrCreatePipeline(aligned?"u16_aligned_fields":"u16_original_fields",source,5);
        auto group=gpu.createBindGroup(pipeline,{{1,w},{3,y},{4,p}});
        gpu.writeBuffer(y,poison.data(),poison.size()*4);
        auto result=gpu.submitAndReadback({{pipeline.pipeline,group,16,1,1,"u16_fields"}},y,y.size);
        auto* values=reinterpret_cast<const float*>(result.data());
        for(uint32_t i=0;i<1023;++i)if(values[i]!=float(uint32_t(bytes[i])|(uint32_t(bytes[i+1])<<8)))throw std::runtime_error("Aligned/odd/cross-word field extraction differs from CPU");
        wgpuBindGroupRelease(group);
    }
    for(auto b:{w,y,p})gpu.releaseBuffer(b);
    std::puts("2046 generic aligned/odd/cross-word u16 outputs exact");
}

static void checkLookup(GPUContext& gpu) {
    const float expected[]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    auto y=gpu.createBuffer("lookup_values",sizeof(expected));
    for(bool packed:{false,true}) {
        auto source=nativeQuantShader(GGUF_TYPE_IQ4_XS,false,false,16,false,false,packed);
        const auto end=source.find("var<workgroup> sums:");if(end==std::string::npos)throw std::runtime_error("Lookup splice failed");source.erase(end);
        source+=R"WGSL(@compute @workgroup_size(32) fn main(@builtin(global_invocation_id) id:vec3<u32>){if(id.x<16u){Y[id.x]=iq4_value(id.x);}})WGSL";
        auto& pl=gpu.getOrCreatePipeline(packed?"packed_lut_values":"original_lut_values",source,5);
        auto group=gpu.createBindGroup(pl,{{3,y}});auto result=gpu.submitAndReadback({{pl.pipeline,group,1,1,1,"lookup_values"}},y,sizeof(expected));
        if(std::memcmp(result.data(),expected,sizeof(expected)))throw std::runtime_error("IQ4 lookup entries differ");wgpuBindGroupRelease(group);
    }
    gpu.releaseBuffer(y);std::puts("32 original/packed lookup entries bit-exact against CPU");
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "Pass independent GGUF reference fixtures\n"); return 2; }
    const bool packedIq4 = std::string(argv[1]) == "--iq4-packed-lut";
    const bool cacheIQ3 = std::string(argv[1]) == "--iq3-block-scale";
    const bool alignedU16 = cacheIQ3 || std::string(argv[1]) == "--aligned-u16";
    const int modeArg = (alignedU16 || packedIq4) ? 2 : 1;
    if(argc <= modeArg){std::fprintf(stderr,"Pass a mode and/or GGUF fixtures after --aligned-u16\n");return 2;}
    const bool alignedActivations = std::string(argv[modeArg]) == "--staged-pair-vec2-a36";
    const bool alignedWeights = alignedActivations || std::string(argv[modeArg]) == "--staged-pair-vec2";
    const bool paired = alignedWeights || std::string(argv[modeArg]) == "--staged-pair";
    const bool staged = paired || std::string(argv[modeArg]) == "--staged";
    const bool tiled32 = staged || std::string(argv[modeArg]) == "--tiled32";
    const bool tiled = tiled32 || std::string(argv[modeArg]) == "--tiled";
    const uint32_t tileRows=tiled32?32u:16u, tileCols=tiled32?8u:16u;
    const int firstFixture = tiled ? modeArg + 1 : modeArg;
    GPUContext gpu;
    if (!gpu.init(WGPUBackendType_D3D12)) return 1;
    if((alignedU16 || packedIq4) && (gpu.adapterName!="NVIDIA GeForce RTX 5080" || gpu.supportsSubgroupMatrix))return 2;
    unsigned tests = 0,paddedGather = 0;
    try {
        if(packedIq4)checkLookup(gpu);
        if(alignedU16 && !tiled)checkFields(gpu);
        for (int arg = firstFixture; arg < argc; ++arg) {
            std::ifstream file(argv[arg], std::ios::binary);
            uint32_t header[4]{};
            file.read(reinterpret_cast<char*>(header), sizeof(header));
            if (!file || !supportsNativeQuant(GGUFType(header[0]))) throw std::runtime_error("Invalid fixture");
            std::vector<uint8_t> raw(header[3]);
            std::vector<float> values(size_t(header[1]) * header[2]);
            file.read(reinterpret_cast<char*>(raw.data()), raw.size());
            file.read(reinterpret_cast<char*>(values.data()), values.size() * 4);
            if (!file) throw std::runtime_error("Truncated fixture");
            const auto originalRaw = raw;
            const auto originalValues = values;
            while (values.size() < 19u * 17408u) {
                raw.insert(raw.end(), originalRaw.begin(), originalRaw.end());
                values.insert(values.end(), originalValues.begin(), originalValues.end());
            }
            const auto type = GGUFType(header[0]);
            for (uint32_t K : {256u, 768u, 5120u, 17408u}) {
             for (uint32_t M : tiled32 ? std::vector<uint32_t>{3,16,17,31,32,33,35,64} : tiled ? std::vector<uint32_t>{3,35} : (cacheIQ3 || packedIq4) ? std::vector<uint32_t>{1,2,3,7,8,9,17} : alignedU16 ? std::vector<uint32_t>{1,3} : std::vector<uint32_t>{3}) {
              for (bool strided : {false, true}) {
                const uint32_t N = tiled ? 19u : K == 256u ? 13u : 5u;
                if (size_t(N) * K > values.size()) continue;
                const auto packed = pack_native_quant(raw.data(), N, K, type);
                auto upload = [&](const char* label, const void* data, size_t bytes) {
                    auto b = gpu.createBuffer(label, bytes);
                    if (data) gpu.writeBuffer(b, data, bytes);
                    return b;
                };
                std::vector<float> x(M * K), bias(N), zeros(M * N * 2, 0);
                for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i) * 0.731f) * 0.25f;
                for (uint32_t i = 0; i < N; ++i) bias[i] = float(i) / 997;
                for(uint32_t m=0;m<M;++m)for(uint32_t n=0;n<N;++n)zeros[m*(strided?2*N:N)+(strided?N:0)+n]=std::numeric_limits<float>::quiet_NaN();
                const uint32_t params[] = {K, N, packed.nBlocks, packed.rowStrideWords,
                                           strided ? N : 0, strided ? 2*N : 0, M, packedIq4 && type==GGUF_TYPE_IQ4_XS && !tiled ? 0xffffffffu : 0u};
                auto bx=upload("x",x.data(),x.size()*4), bw=upload("w",packed.data.data(),packed.data.size()*4),
                     bb=upload("bias",bias.data(),bias.size()*4), by=upload("y",zeros.data(),zeros.size()*4),
                     bp=upload("params",params,sizeof(params));
                std::vector<uint8_t> result;
                if(staged) {
                    auto scratch=upload("staged",nullptr,uint64_t(K)*8*4);
                    auto& decode=gpu.getOrCreatePipeline("native_slice_"+std::to_string(type),nativeQuantDecodeSliceShader(type,alignedU16),5);
                    auto& dense=gpu.getOrCreatePipeline("native_dense",nativeQuantDensePrefillShader(paired,alignedWeights,alignedActivations),5);
                    std::vector<GPUBuffer> paramsBuffers;std::vector<WGPUBindGroup> groups;std::vector<Dispatch> dispatches;
                    for(uint32_t col=0;col<N;col+=8) {
                        const uint32_t count=std::min(8u,N-col),p[]={K,N,packed.nBlocks,packed.rowStrideWords,col,strided?2*N:N,M,count,(strided?N:0)+col};
                        auto parameter=upload("slice_params",p,sizeof(p));paramsBuffers.push_back(parameter);
                        auto dg=gpu.createBindGroup(decode,{{1,bw},{3,scratch},{4,parameter}});
                        auto mg=gpu.createBindGroup(dense,{{0,bx},{1,scratch},{2,bb},{3,by},{4,parameter}});
                        groups.push_back(dg);groups.push_back(mg);
                        dispatches.push_back({decode.pipeline,dg,(K+255)/256,count,1,"decode_slice"});
                        dispatches.push_back({dense.pipeline,mg,(M+31)/32,(count+7)/8,1,"dense_slice"});
                    }
                    result=gpu.submitAndReadback(dispatches,by,zeros.size()*4);
                    for(auto group:groups)wgpuBindGroupRelease(group);
                    for(auto buffer:paramsBuffers)gpu.releaseBuffer(buffer);
                    gpu.releaseBuffer(scratch);
                } else {
                    auto& pl=gpu.getOrCreatePipeline("native_quant_"+std::to_string(type),nativeQuantShader(type,false,tiled,tileRows,alignedU16,cacheIQ3 && type==GGUF_TYPE_IQ3_S && !tiled,packedIq4 && type==GGUF_TYPE_IQ4_XS && !tiled),5);
                    auto bg=gpu.createBindGroup(pl,{{0,bx},{1,bw},{2,bb},{3,by},{4,bp}});
                    result=gpu.submitAndReadback({{pl.pipeline,bg,tiled?(M+tileRows-1)/tileRows:M,tiled?(N+tileCols-1)/tileCols:(N+7)/8,1,"native_quant"}},by,zeros.size()*4);
                    wgpuBindGroupRelease(bg);
                }
                if((alignedU16 || packedIq4) && !tiled32){
                    auto& original=gpu.getOrCreatePipeline("native_original_"+std::to_string(type),nativeQuantShader(type,false,tiled,tileRows,cacheIQ3 && alignedU16,false),5);
                    auto originalGroup=gpu.createBindGroup(original,{{0,bx},{1,bw},{2,bb},{3,by},{4,bp}});
                    gpu.writeBuffer(by,zeros.data(),zeros.size()*4);
                    auto reference=gpu.submitAndReadback({{original.pipeline,originalGroup,tiled?(M+tileRows-1)/tileRows:M,tiled?(N+tileCols-1)/tileCols:(N+7)/8,1,"native_original"}},by,zeros.size()*4);
                    wgpuBindGroupRelease(originalGroup);if(result!=reference)throw std::runtime_error("Native aligned-u16 output bits differ");
                }
                if(tiled32){
                    auto& legacy=gpu.getOrCreatePipeline("native_quant_legacy_"+std::to_string(type),nativeQuantShader(type,false,true,16),5);
                    auto legacyBg=gpu.createBindGroup(legacy,{{0,bx},{1,bw},{2,bb},{3,by},{4,bp}});
                    gpu.writeBuffer(by,zeros.data(),zeros.size()*4);
                    auto reference=gpu.submitAndReadback({{legacy.pipeline,legacyBg,(M+15)/16,(N+15)/16,1,"native_quant_legacy"}},by,zeros.size()*4);
                    wgpuBindGroupRelease(legacyBg);
                    if(result!=reference){std::fprintf(stderr,"FAIL tile bit parity type=%u K=%u M=%u strided=%d\n",type,K,M,strided);return 1;}
                }
                const float* actual=reinterpret_cast<const float*>(result.data());
                double maxError=0;
                for(uint32_t m=0;m<M;++m) for(uint32_t n=0;n<N;++n) {
                    double expected=bias[n],sumAbs=std::abs(double(bias[n]));
                    for(uint32_t k=0;k<K;++k){const double product=double(x[m*K+k])*values[n*K+k];expected+=product;sumAbs+=std::abs(product);}
                    const uint32_t index = strided ? m*N*2+N+n : m*N+n;
                    const double error=std::abs(actual[index]-expected);
                    double tolerance=2e-4+2e-5*std::abs(expected);
                    if(K>5120){
                        // New long-dot coverage can exceed the old absolute
                        // tolerance even for the bit-identical 16-row result.
                        // Bound FP32 rounding along a lane's multiply/adds,
                        // five reduction stages, and the bias addition.
                        const double u=std::numeric_limits<float>::epsilon()/2.0;
                        const double depth=2.0*((K+31u)/32u)+6.0;
                        tolerance=std::max(tolerance,(depth*u/(1.0-depth*u))*sumAbs);
                    }
                    if(!std::isfinite(actual[index]) || error>tolerance || (strided && actual[m*N*2+n]!=0)) {
                        std::fprintf(stderr,"FAIL type=%u K=%u m=%u n=%u got=%.9g expected=%.9g\n",type,K,m,n,actual[index],expected);
                        return 1;
                    }
                    maxError=std::max(maxError,error);
                }
                std::vector<uint32_t> tokens(M);for(uint32_t i=0;i<M;++i)tokens[i]=(i*7+N-1)%N;
                uint32_t gatherParams[]={K,N,packed.nBlocks,packed.rowStrideWords,0x3f800000};
                auto bt=upload("tokens",tokens.data(),tokens.size()*4), gp=upload("gather_params",gatherParams,sizeof(gatherParams)),
                     gy=upload("gather",nullptr,size_t(M)*K*4);
                if(gy.size!=size_t(M)*K*4)++paddedGather;
                auto& gl=gpu.getOrCreatePipeline("native_quant_gather_"+std::to_string(type),nativeQuantShader(type,true,false,16,alignedU16),5);
                auto gb=gpu.createBindGroup(gl,{{0,bt},{1,bw},{2,bb},{3,gy},{4,gp}});
                std::vector<uint32_t> beforeGather(size_t(M)*K,0x7fc12345u);gpu.writeBuffer(gy,beforeGather.data(),beforeGather.size()*4);
                result=gpu.submitAndReadback({{gl.pipeline,gb,(K+255)/256,M,1,"native_gather"}},gy,size_t(M)*K*4);
                wgpuBindGroupRelease(gb);
                auto& originalGather=gpu.getOrCreatePipeline("gather_original_"+std::to_string(type),nativeQuantShader(type,true),5);
                auto originalGroup=gpu.createBindGroup(originalGather,{{0,bt},{1,bw},{2,bb},{3,gy},{4,gp}});
                std::vector<uint32_t> gatherPoison(size_t(M)*K,0x7fc12345u);gpu.writeBuffer(gy,gatherPoison.data(),gatherPoison.size()*4);
                auto reference=gpu.submitAndReadback({{originalGather.pipeline,originalGroup,(K+255)/256,M,1,"gather_original"}},gy,size_t(M)*K*4);
                wgpuBindGroupRelease(originalGroup);if(result!=reference)throw std::runtime_error("Aligned gather output bits differ");
                actual=reinterpret_cast<const float*>(result.data());
                for(uint32_t m=0;m<M;++m) for(uint32_t k=0;k<K;++k) {
                    if(actual[m*K+k]!=values[tokens[m]*K+k]) {
                        std::fprintf(stderr,"FAIL gather type=%u K=%u token=%u k=%u got=%.9g expected=%.9g\n",type,K,tokens[m],k,actual[m*K+k],values[tokens[m]*K+k]);
                        return 1;
                    }
                }
                std::printf("PASS type=%u K=%u N=%u M=%u strided=%d matmul_error=%.9g gather_exact\n",type,K,N,M,strided,maxError);
                for(auto b : {bx,bw,bb,by,bp,bt,gp,gy}) gpu.releaseBuffer(b);
                ++tests;
              }
             }
            }
        }
    } catch(const std::exception& error) { std::fprintf(stderr,"%s\n",error.what()); return 1; }
    std::printf("%u native quantization tests passed\n",tests);
    std::printf("%u padded gather allocations checked within logical extents\n",paddedGather);
    return tests ? 0 : 1;
}
