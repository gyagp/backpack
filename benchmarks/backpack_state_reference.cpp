#include "backpack.h"
#include "lm_session.h"
#include "json_parser.h"
#include "../apps/common/app_common.h"
#include <filesystem>
#include <fstream>
#include <iterator>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <unordered_map>
#ifdef BP_NATIVE_MEMORY_DIAGNOSTICS
#include <dawn/native/DawnNative.h>
#endif
#ifdef _WIN32
#include <dxgi1_4.h>
#pragma comment(lib, "dxgi.lib")
#endif
struct MemorySample { uint64_t local=0, nonlocal=0, budget=0; bool available=false; };
#ifdef BP_NATIVE_MEMORY_DIAGNOSTICS
struct NativeMemoryDump : dawn::native::MemoryDump {
    struct Entry { std::string label, usage; uint64_t bytes=0; };
    std::unordered_map<std::string,Entry> entries;
    void AddScalar(const char* name,const char* key,const char*,uint64_t value) override {
        if(std::string(key)=="size")entries[name].bytes=value;
    }
    void AddString(const char* name,const char* key,const std::string& value) override {
        if(std::string(key)=="label")entries[name].label=value;
        else if(std::string(key)=="usage")entries[name].usage=value;
    }
    void write(const std::filesystem::path& path) const {
        std::ofstream file(path);
        for(const auto& [name,e]:entries)
            file<<"{\"object\":"<<std::quoted(name)<<",\"label\":"<<std::quoted(e.label)
                <<",\"usage\":"<<std::quoted(e.usage)<<",\"bytes\":"<<e.bytes<<"}\n";
    }
};
#endif
static MemorySample memorySample(const std::string& name) {
    MemorySample result;
#ifdef _WIN32
    IDXGIFactory1* factory=nullptr;
    if(FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))return result;
    for(UINT index=0;;++index){
        IDXGIAdapter1* adapter=nullptr;if(factory->EnumAdapters1(index,&adapter)!=S_OK)break;
        DXGI_ADAPTER_DESC1 desc{};adapter->GetDesc1(&desc);char utf8[512]={};
        WideCharToMultiByte(CP_UTF8,0,desc.Description,-1,utf8,sizeof(utf8),nullptr,nullptr);std::string candidate(utf8);
        if(candidate==name){IDXGIAdapter3* budget=nullptr;
            if(SUCCEEDED(adapter->QueryInterface(IID_PPV_ARGS(&budget)))){
                DXGI_QUERY_VIDEO_MEMORY_INFO local{},shared{};
                if(SUCCEEDED(budget->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&local)) &&
                   SUCCEEDED(budget->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,&shared)))
                    result={local.CurrentUsage,shared.CurrentUsage,local.Budget,true};
                budget->Release();
            }
        }
        adapter->Release();if(result.available)break;
    }
    factory->Release();
#endif
    return result;
}
int main(int argc,char** argv) {
    if(argc!=4)return 2;
    try {
        std::ifstream input(argv[2]);std::string text{std::istreambuf_iterator<char>(input),{}};
        auto request=json_parse(text);auto device=app::createDevice("d3d12");
        bp::LmOptions options;options.maxSeqLen=request.has("max_seq_len")?request["max_seq_len"].as_int():1024;
        options.fastDecode=request.has("fast_decode")?request["fast_decode"].as_bool():false;
        options.prefillChunkSize=request.has("prefill_chunk")?request["prefill_chunk"].as_int():0;
        auto session=bp::LmSession::Create(device,argv[1],options);if(!session.IsValid())return 1;
        auto* gpu=static_cast<GPUContext*>(device.GetGPUContext());
        MemorySample peak;
        auto sample=[&]{auto current=memorySample(gpu->adapterName);peak.local=std::max(peak.local,current.local);
            peak.nonlocal=std::max(peak.nonlocal,current.nonlocal);peak.budget=current.budget;peak.available=current.available;};
        const bool nativeMemory=request.has("native_memory") && request["native_memory"].as_bool();
#ifndef BP_NATIVE_MEMORY_DIAGNOSTICS
        if(nativeMemory)throw std::runtime_error("Build the reference helper with BP_NATIVE_MEMORY_DIAGNOSTICS and C++20 for native memory diagnostics");
#endif
        struct Snapshot { std::string stage; int cycle; uint64_t allocated; MemorySample dxgi; uint64_t pooled;
            uint64_t dawnUsed=0,dawnAllocated=0,dawnBuffers=0,dawnObjects=0; };
        std::vector<Snapshot> snapshots;
        auto snapshot=[&](const char* stage,int cycle){
            sample();
            snapshots.push_back({stage,cycle,gpu->totalAllocatedBytes,memorySample(gpu->adapterName),gpu->pooledBufferBytes()});
            if(nativeMemory){
#ifdef BP_NATIVE_MEMORY_DIAGNOSTICS
                // Native allocator totals include heap placement and internal
                // allocations that Backpack's logical live-byte counter omits.
                // Neither these totals nor DXGI usage establish full residency.
                const auto allocation=dawn::native::GetAllocatorMemoryInfo(gpu->device);
                const auto estimated=dawn::native::ComputeEstimatedMemoryUsageInfo(gpu->device);
                NativeMemoryDump dump;dawn::native::DumpMemoryStatistics(gpu->device,&dump);
                auto& s=snapshots.back();s.dawnUsed=allocation.totalUsedMemory;
                s.dawnAllocated=allocation.totalAllocatedMemory;s.dawnBuffers=estimated.buffersUsage;
                s.dawnObjects=dump.entries.size();
                dump.write(std::filesystem::path(argv[3]).parent_path()/(std::string("buffers-")+stage+"-"+std::to_string(cycle)+".jsonl"));
                fprintf(stderr,"[native-memory] %s %d used=%llu allocated=%llu buffers=%llu objects=%llu\n",
                    stage,cycle,(unsigned long long)s.dawnUsed,(unsigned long long)s.dawnAllocated,
                    (unsigned long long)s.dawnBuffers,(unsigned long long)s.dawnObjects);
#endif
            }
            if(std::getenv("BP_ALLOC_TRACE"))
                fprintf(stderr,"[memory-snapshot] %s %d %llu\n",stage,cycle,
                    (unsigned long long)gpu->totalAllocatedBytes);
        };
        snapshot("loaded",0);
        const int resetCycles=request.has("reset_cycles")?request["reset_cycles"].as_int():0;
        for(int cycle=1;cycle<=resetCycles;++cycle){session.Reset();snapshot("reset_only",cycle);}
        const int repetitions=request.has("repetitions")?request["repetitions"].as_int():1;
        std::vector<std::vector<int32_t>> allGenerated;
        std::vector<double> prefillMs,decodeMs;
        using Clock=std::chrono::steady_clock;
        const int profileDecode=request.has("profile_decode_step")?request["profile_decode_step"].as_int():-1;
        const int profilePrefill=request.has("profile_prefill_batch")?request["profile_prefill_batch"].as_int():-1;
        const int profileRepetition=request.has("profile_repetition")?request["profile_repetition"].as_int():0;
        if(repetitions<=0 || profileRepetition<0 || profileRepetition>=repetitions)
            throw std::runtime_error("Invalid repetition or profile repetition count");
        const bool timestamps=!request.has("gpu_timestamps") || request["gpu_timestamps"].as_bool();
        const int trimStep=request.has("trim_pool_before_decode")?request["trim_pool_before_decode"].as_int():-1;
        const auto profileDir=std::filesystem::path(argv[3]).parent_path();
        std::ofstream diagnostics;
        if(profileDecode>=0 || profilePrefill>=0 || trimStep>=0) diagnostics.open(profileDir/"diagnostics.jsonl");
        double measuredMs=0;
        auto measure=[&](const char* phase,int index,int tokens,bool enabled,auto&& run){
            const auto position=session.GetPosition();
            const auto pooledBytes=gpu->pooledBufferBytes();
            if(enabled && timestamps)session.EnableProfiling();
            if(enabled){gpu->diagnostics={};gpu->diagnosticsEnabled=true;}
            const auto begin=Clock::now();const auto result=run();
            const double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
            measuredMs=elapsed;
            if(!enabled)return result;
            gpu->diagnosticsEnabled=false;const auto d=gpu->diagnostics;
            diagnostics<<"{\"phase\":\""<<phase<<"\",\"step\":"<<index<<",\"position\":"<<position
                <<",\"tokens\":"<<tokens<<",\"next\":"<<result<<",\"pooled_bytes_before\":"<<pooledBytes
                <<",\"pooled_bytes_after\":"<<gpu->pooledBufferBytes()<<",\"gpu_timestamps\":"<<(timestamps?"true":"false")
                <<",\"wall_ms\":"<<elapsed<<",\"dispatches\":"<<d.dispatches<<",\"submits\":"<<d.submits
                <<",\"flushes\":"<<d.flushes<<",\"writes\":"<<d.writes<<",\"write_bytes\":"<<d.writeBytes
                <<",\"queue_waits\":"<<d.queueWaits<<",\"map_waits\":"<<d.mapWaits
                <<",\"encode_ms\":"<<d.encodeNs/1e6<<",\"submit_ms\":"<<d.submitNs/1e6
                <<",\"write_ms\":"<<d.writeNs/1e6<<",\"queue_wait_ms\":"<<d.queueWaitNs/1e6
                <<",\"map_wait_ms\":"<<d.mapWaitNs/1e6<<"}\n";diagnostics.flush();
            if(timestamps)session.FinishProfiling((profileDir/(std::string(phase)+"-"+std::to_string(index)+".html")).string(),tokens,elapsed,std::string(phase)=="prefill");
            if(session.GetPosition()!=position+tokens)throw std::runtime_error("Profiling changed session position");
            return result;
        };
        for(int repetition=0;repetition<repetitions;++repetition){
            if(repetition){session.Reset();snapshot("reset",repetition);}
            int32_t next=-1;
            int batchIndex=0;
            for(const auto& batch:request["batches"].as_array()) {
                std::vector<int32_t> tokens;for(const auto& t:batch.as_array())tokens.push_back(t.as_int());
                next=measure("prefill",batchIndex,(int)tokens.size(),repetition==profileRepetition && batchIndex==profilePrefill,
                    [&]{return session.Prefill(tokens.data(),(uint32_t)tokens.size());});++batchIndex;
                prefillMs.push_back(measuredMs);sample();
            }
            if(nativeMemory)snapshot("prefilled",repetition);
            std::vector<int32_t> generated;
            for(int i=0;i<request["max_new_tokens"].as_int();++i){
                generated.push_back(next);bool stop=false;
                for(const auto& id:request["stop_ids"].as_array())if(id.as_int()==next)stop=true;
                if(stop || i+1==request["max_new_tokens"].as_int())break;
                if(i+1==trimStep){
                    snapshot("before_trim",repetition);
                    const auto begin=Clock::now();const auto freed=session.TrimMemory();
                    const double elapsed=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
                    diagnostics<<"{\"phase\":\"trim\",\"step\":"<<i+1<<",\"freed_pool_bytes\":"<<freed<<",\"wall_ms\":"<<elapsed<<"}\n";
                    diagnostics.flush();snapshot("after_trim",repetition);
                }
                next=measure("decode",i+1,1,repetition==profileRepetition && i+1==profileDecode,[&]{return session.Decode();});
                decodeMs.push_back(measuredMs);sample();
                if(nativeMemory && (i+1==2 || i+1==8))snapshot(i+1==2?"decode2":"decode8",repetition);
            }
            allGenerated.push_back(std::move(generated));snapshot("generated",repetition);
        }
        if(gpu->executionError.load() || gpu->deviceLost)
            throw std::runtime_error("WebGPU execution failed");
        std::ofstream out(argv[3]);out<<"{\"gpu_allocated_bytes\":"<<gpu->totalAllocatedBytes
            <<",\"peak_gpu_allocated_bytes\":"<<gpu->peakAllocatedBytes
            <<",\"dxgi_available\":"<<(peak.available?"true":"false")<<",\"dxgi_local_bytes\":"<<peak.local
            <<",\"dxgi_nonlocal_bytes\":"<<peak.nonlocal<<",\"dxgi_budget_bytes\":"<<peak.budget<<",\"tokens\":[";
        const auto& generated=allGenerated.at(0);
        for(size_t i=0;i<generated.size();++i)out<<(i?",":"")<<generated[i];out<<"],\"repetition_tokens\":[";
        for(size_t i=0;i<allGenerated.size();++i){out<<(i?",[":"[");
            for(size_t j=0;j<allGenerated[i].size();++j)out<<(j?",":"")<<allGenerated[i][j];out<<"]";}
        out<<"],\"prefill_ms\":[";
        for(size_t i=0;i<prefillMs.size();++i)out<<(i?",":"")<<prefillMs[i];out<<"],\"decode_ms\":[";
        for(size_t i=0;i<decodeMs.size();++i)out<<(i?",":"")<<decodeMs[i];out<<"],\"memory_snapshots\":[";
        for(size_t i=0;i<snapshots.size();++i){const auto& s=snapshots[i];out<<(i?",":"")
            <<"{\"stage\":\""<<s.stage<<"\",\"cycle\":"<<s.cycle<<",\"allocated_bytes\":"<<s.allocated
            <<",\"pooled_bytes\":"<<s.pooled
            <<",\"dawn_used_bytes\":"<<s.dawnUsed<<",\"dawn_allocated_bytes\":"<<s.dawnAllocated
            <<",\"dawn_buffer_bytes\":"<<s.dawnBuffers<<",\"dawn_buffer_count\":"<<s.dawnObjects
            <<",\"dxgi_available\":"<<(s.dxgi.available?"true":"false")<<",\"dxgi_local_bytes\":"<<s.dxgi.local
            <<",\"dxgi_nonlocal_bytes\":"<<s.dxgi.nonlocal<<",\"dxgi_budget_bytes\":"<<s.dxgi.budget<<"}";}
        out<<"]}\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
