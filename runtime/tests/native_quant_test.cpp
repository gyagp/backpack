#include "gpu_context.h"
#include "native_quant.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "Pass independent GGUF reference fixtures\n"); return 2; }
    const bool alignedWeights = std::string(argv[1]) == "--staged-pair-vec2";
    const bool paired = alignedWeights || std::string(argv[1]) == "--staged-pair";
    const bool staged = paired || std::string(argv[1]) == "--staged";
    const bool tiled32 = staged || std::string(argv[1]) == "--tiled32";
    const bool tiled = tiled32 || std::string(argv[1]) == "--tiled";
    const uint32_t tileRows=tiled32?32u:16u, tileCols=tiled32?8u:16u;
    const int firstFixture = tiled ? 2 : 1;
    GPUContext gpu;
    if (!gpu.init(WGPUBackendType_D3D12)) return 1;
    unsigned tests = 0;
    try {
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
             for (uint32_t M : tiled32 ? std::vector<uint32_t>{3,16,17,31,32,33,35,64} : tiled ? std::vector<uint32_t>{3,35} : std::vector<uint32_t>{3}) {
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
                const uint32_t params[] = {K, N, packed.nBlocks, packed.rowStrideWords,
                                           strided ? N : 0, strided ? 2*N : 0, M, 0};
                auto bx=upload("x",x.data(),x.size()*4), bw=upload("w",packed.data.data(),packed.data.size()*4),
                     bb=upload("bias",bias.data(),bias.size()*4), by=upload("y",zeros.data(),zeros.size()*4),
                     bp=upload("params",params,sizeof(params));
                std::vector<uint8_t> result;
                if(staged) {
                    auto scratch=upload("staged",nullptr,uint64_t(K)*8*4);
                    auto& decode=gpu.getOrCreatePipeline("native_slice_"+std::to_string(type),nativeQuantDecodeSliceShader(type),5);
                    auto& dense=gpu.getOrCreatePipeline("native_dense",nativeQuantDensePrefillShader(paired,alignedWeights),5);
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
                    result=gpu.submitAndReadback(dispatches,by,by.size);
                    for(auto group:groups)wgpuBindGroupRelease(group);
                    for(auto buffer:paramsBuffers)gpu.releaseBuffer(buffer);
                    gpu.releaseBuffer(scratch);
                } else {
                    auto& pl=gpu.getOrCreatePipeline("native_quant_"+std::to_string(type),nativeQuantShader(type,false,tiled,tileRows),5);
                    auto bg=gpu.createBindGroup(pl,{{0,bx},{1,bw},{2,bb},{3,by},{4,bp}});
                    result=gpu.submitAndReadback({{pl.pipeline,bg,tiled?(M+tileRows-1)/tileRows:M,tiled?(N+tileCols-1)/tileCols:(N+7)/8,1,"native_quant"}},by,by.size);
                    wgpuBindGroupRelease(bg);
                }
                if(tiled32){
                    auto& legacy=gpu.getOrCreatePipeline("native_quant_legacy_"+std::to_string(type),nativeQuantShader(type,false,true,16),5);
                    auto legacyBg=gpu.createBindGroup(legacy,{{0,bx},{1,bw},{2,bb},{3,by},{4,bp}});
                    auto reference=gpu.submitAndReadback({{legacy.pipeline,legacyBg,(M+15)/16,(N+15)/16,1,"native_quant_legacy"}},by,by.size);
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
                auto& gl=gpu.getOrCreatePipeline("native_quant_gather_"+std::to_string(type),nativeQuantShader(type,true),5);
                auto gb=gpu.createBindGroup(gl,{{0,bt},{1,bw},{2,bb},{3,gy},{4,gp}});
                result=gpu.submitAndReadback({{gl.pipeline,gb,(K+255)/256,M,1,"native_gather"}},gy,gy.size);
                wgpuBindGroupRelease(gb);
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
    return tests ? 0 : 1;
}
