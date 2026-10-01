#include "gpu_context.h"
#include "native_quant.h"
#include "gguf_loader.h"
#include "json_parser.h"
#include "mapped_file.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) { if(!ok) throw std::runtime_error(message); }
void replace(std::string& text,const std::string& from,const std::string& to) {
    const auto pos=text.find(from);require(pos!=std::string::npos,"Shader splice not found");text.replace(pos,from.size(),to);
}
uint32_t blockBytes(GGUFType type) {
    switch(type) {
        case GGUF_TYPE_IQ3_S:return 110;
        case GGUF_TYPE_IQ3_XXS:return 98;
        case GGUF_TYPE_IQ4_XS:return 136;
        default:throw std::runtime_error("Probe type has not been mapped");
    }
}
std::vector<uint64_t> timestamps(GPUContext& gpu,GPUProfiler& profiler) {
    WGPUMapAsyncStatus status=WGPUMapAsyncStatus_Error;
    WGPUBufferMapCallbackInfo cb{};cb.mode=WGPUCallbackMode_WaitAnyOnly;cb.userdata1=&status;
    cb.callback=[](WGPUMapAsyncStatus s,WGPUStringView,void* data,void*) { *static_cast<WGPUMapAsyncStatus*>(data)=s; };
    const uint64_t bytes=uint64_t(profiler.nextIndex)*8;
    const auto future=wgpuBufferMapAsync(profiler.readbackBuf,WGPUMapMode_Read,0,bytes,cb);
    gpu.waitForMap(future);require(status==WGPUMapAsyncStatus_Success,"Timestamp map failed");
    const auto* data=static_cast<const uint64_t*>(wgpuBufferGetConstMappedRange(profiler.readbackBuf,0,bytes));
    std::vector<uint64_t> result(data,data+profiler.nextIndex);wgpuBufferUnmap(profiler.readbackBuf);return result;
}
}

int main(int argc,char** argv) {
    if(argc!=3) { std::cerr<<"Usage: probe request.json result.json\n";return 2; }
    GPUContext gpu;
    try {
        const char* host=std::getenv("COMPUTERNAME");require(host && _stricmp(host,"webgfx-104")==0,"Probe is scoped to webgfx-104");
        std::ifstream file(argv[1]);std::string text{std::istreambuf_iterator<char>(file),{}};const auto request=json_parse(text);
        GGUFFile model;const auto path=request["model"].as_string();require(model.open(path),"GGUF open failed");
        require(model.getString("general.name")=="Qwen3.8-27B" && model.getU32("general.file_type")==26,"Unexpected model identity");
        MappedFile mapped;require(mapped.open(path),"GGUF mapping failed");
        require(gpu.init(WGPUBackendType_D3D12),"GPU init failed");
        require(gpu.adapterName=="NVIDIA GeForce RTX 5080" && !gpu.supportsSubgroupMatrix && gpu.supportsTimestampQuery,"Unexpected adapter/features");
        gpu.bufferPoolEnabled=false;
        std::ofstream out(argv[2]);out<<std::setprecision(12)<<"{\"device\":\"webgfx-104 / NVIDIA GeForce RTX 5080\",\"cases\":[";
        bool firstCase=true;
        const uint32_t M=request["M"].as_uint();require(M==512,"Profile workload requires M512");
        for(const auto& c:request["cases"].as_array()) {
            const uint32_t K=c["K"].as_uint(),N=c["N"].as_uint();const auto type=GGUFType(c["type_id"].as_uint());
            const uint64_t rowBytes=uint64_t(K/256)*blockBytes(type),scratchBytes=uint64_t(K)*N*4;
            require(K%256==0 && N<=65535 && scratchBytes<=1024ull*1048576,"Probe shape/scratch bound exceeded");
            require(c["scratch_bytes"].as_number()==double(scratchBytes),"Scratch metadata mismatch");
            std::vector<uint8_t> raw;raw.reserve(size_t(rowBytes*N));uint64_t rows=0;
            for(const auto& name:c["tensors"].as_array()) {
                const auto found=model.tensor_index.find(name.as_string());require(found!=model.tensor_index.end(),"Tensor not found");
                const auto& tensor=model.tensors[found->second];
                require(tensor.shape.size()==2 && tensor.shape[0]==K && tensor.type==type,"Tensor metadata mismatch");
                const uint64_t bytes=rowBytes*tensor.shape[1],offset=model.data_offset+tensor.offset;
                require(offset<=mapped.size && bytes<=mapped.size-offset,"Tensor exceeds artifact");
                raw.insert(raw.end(),mapped.data+offset,mapped.data+offset+bytes);rows+=tensor.shape[1];
            }
            require(rows==N,"Combined gate/up row count mismatch");
            auto packed=pack_native_quant(raw.data(),N,K,type);raw.clear();raw.shrink_to_fit();
            std::vector<float> x(size_t(M)*K),bias(N);
            for(size_t i=0;i<x.size();++i)x[i]=std::sin(float(i%65521)*0.0131f)*0.25f;
            for(uint32_t i=0;i<N;++i)bias[i]=float(int(i%31)-15)/997.0f;
            const uint32_t params[]={K,N,packed.nBlocks,packed.rowStrideWords,0,N,M,0};
            auto upload=[&](const char* label,const void* data,uint64_t bytes) {auto b=gpu.createBuffer(label,bytes);require(b.handle!=nullptr,"Buffer allocation failed");if(data)gpu.writeBuffer(b,data,bytes);return b;};
            auto bx=upload("input",x.data(),x.size()*4),bw=upload("packed",packed.data.data(),packed.data.size()*4),
                 bb=upload("bias",bias.data(),bias.size()*4),bp=upload("params",params,sizeof(params)),
                 by=upload("output",nullptr,uint64_t(M)*N*4),bs=upload("scratch",nullptr,scratchBytes);
            packed.data.clear();packed.data.shrink_to_fit();
            auto source=nativeQuantShader(type,false,true,32),dense=source,decode=source;
            replace(dense,"b=decode(wid.y*8u+br,k0+bk);","b=bitcast<f32>(W[(wid.y*8u+br)*K+k0+bk]);");
            const auto end=decode.find("var<workgroup> tileA:");require(end!=std::string::npos,"Decode splice not found");decode.erase(end);
            decode+=R"WGSL(
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    let k=gid.x; let row=gid.y;
    if(k<P[0] && row<P[1]) { Y[row*P[0]+k]=decode(row,k); }
}
)WGSL";
            const auto suffix=std::to_string(type);
            auto& fusedPipeline=gpu.getOrCreatePipeline("probe_fused_"+suffix,source,5);
            auto& densePipeline=gpu.getOrCreatePipeline("probe_dense_"+suffix,dense,5);
            auto& decodePipeline=gpu.getOrCreatePipeline("probe_decode_"+suffix,decode,5);
            auto fusedBg=gpu.createBindGroup(fusedPipeline,{{0,bx},{1,bw},{2,bb},{3,by},{4,bp}});
            auto denseBg=gpu.createBindGroup(densePipeline,{{0,bx},{1,bs},{2,bb},{3,by},{4,bp}});
            auto decodeBg=gpu.createBindGroup(decodePipeline,{{1,bw},{3,bs},{4,bp}});
            const Dispatch fused{fusedPipeline.pipeline,fusedBg,(M+31)/32,(N+7)/8,1,"fused"};
            const Dispatch denseD{densePipeline.pipeline,denseBg,(M+31)/32,(N+7)/8,1,"dense"};
            const Dispatch decodeD{decodePipeline.pipeline,decodeBg,(K+255)/256,N,1,"decode_weights"};
            const std::vector<std::vector<Dispatch>> variants={{fused},{decodeD},{denseD},{decodeD,denseD}};
            const char* names[]={"fused","decode_weights","dense","combined"};
            // All pipeline creation and complete output parity precede timing.
            const auto expected=gpu.submitAndReadback({fused},by,by.size);
            for(int reset=0;reset<2;++reset) {
                const auto actual=gpu.submitAndReadback(variants[3],by,by.size);
                if(actual!=expected) {
                    const auto* a=reinterpret_cast<const float*>(actual.data());const auto* b=reinterpret_cast<const float*>(expected.data());
                    size_t differences=0,worst=0;double error=0;
                    for(size_t i=0;i<actual.size()/4;++i)if(std::memcmp(a+i,b+i,4)){++differences;double e=std::abs(double(a[i])-b[i]);if(e>error){error=e;worst=i;}}
                    std::cerr<<"FAIL parity "<<c["example"].as_string()<<" differences="<<differences<<" index="<<worst<<" max_abs="<<error<<"\n";
                    throw std::runtime_error("Staged output is not bit-exact");
                }
            }
            require(!gpu.executionError && !gpu.deviceLost,"GPU validation/execution failure");
            if(!firstCase)out<<',';firstCase=false;
            out<<"{\"name\":"<<std::quoted(c["example"].as_string())<<",\"type\":"<<std::quoted(c["type"].as_string())
               <<",\"K\":"<<K<<",\"N\":"<<N<<",\"M\":"<<M<<",\"scratch_bytes\":"<<scratchBytes
               <<",\"bit_exact\":true,\"output_elements\":"<<uint64_t(M)*N<<",\"repeated_parity_checks\":2,\"samples\":[";
            out.flush();std::cout<<"PASS parity "<<c["example"].as_string()<<" scratch_MiB="<<scratchBytes/1048576.0<<std::endl;
            GPUProfiler profiler;require(profiler.init(gpu.device,gpu.instance,gpu.queue),"Profiler init failed");bool firstSample=true;
            const int warmups=request["warmups"].as_int(),repetitions=request["repetitions"].as_int();
            for(int repeat=-warmups;repeat<repetitions;++repeat)for(int slot=0;slot<4;++slot) {
                const int mode=(repeat%2==0)?slot:3-slot;profiler.nextIndex=0;profiler.entries.clear();
                const auto start=std::chrono::steady_clock::now();
                gpu.submitAndReadbackProfiled(variants[mode],by,4,profiler);
                const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                auto ticks=timestamps(gpu,profiler);require(!gpu.executionError && !gpu.deviceLost,"Profile execution failed");
                if(repeat<0)continue;
                if(!firstSample)out<<',';firstSample=false;
                out<<"{\"repetition\":"<<repeat<<",\"mode\":\""<<names[mode]<<"\",\"wall_ms\":"<<wall<<",\"regions\":[";
                for(size_t i=0;i<profiler.entries.size();++i) {
                    const auto& entry=profiler.entries[i];const auto begin=ticks.at(entry.beginIdx),endTick=ticks.at(entry.endIdx);
                    require(begin && endTick>begin,"Invalid GPU timestamp pair");
                    out<<(i?",":"")<<"{\"name\":"<<std::quoted(entry.name)<<",\"begin_ns\":\""<<begin<<"\",\"end_ns\":\""<<endTick<<"\",\"gpu_ms\":"<<double(endTick-begin)/1e6<<'}';
                }
                out<<"]}";out.flush();
            }
            out<<"]}";out.flush();profiler.destroy();
            for(auto group:{fusedBg,denseBg,decodeBg})wgpuBindGroupRelease(group);
            for(auto buffer:{bx,bw,bb,bp,by,bs})gpu.releaseBuffer(buffer);
            std::cout<<"PROFILE complete "<<c["example"].as_string()<<std::endl;
        }
        out<<"]}\n";gpu.destroy();return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n';gpu.destroy();return 1; }
}
