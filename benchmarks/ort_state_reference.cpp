// Native ORT reference runner for stateful text ONNX graphs. Requests contain
// token batches, so tokenization and chat formatting cannot hide runtime drift.
#include <onnxruntime_cxx_api.h>
#include "json_parser.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>
#include <memory>
#include <unordered_map>

namespace fs=std::filesystem;
struct TensorData {
    ONNXTensorElementDataType type;
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;
};
static size_t elementSize(ONNXTensorElementDataType type) {
    switch(type){case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:return 4;
        default:throw std::runtime_error("Unsupported reference tensor type");}
}
static float halfValue(uint16_t h) {
    const float sign=h&0x8000?-1.0f:1.0f;
    int exponent=(h>>10)&31, mantissa=h&1023;
    if(exponent==31)return mantissa?NAN:sign*INFINITY;
    return sign*std::ldexp(float(exponent?1024+mantissa:mantissa),exponent?exponent-25:-24);
}
static TensorData integers(std::vector<int64_t> shape,const std::vector<int64_t>& values) {
    TensorData t{ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,std::move(shape),std::vector<uint8_t>(values.size()*8)};
    if(!values.empty())memcpy(t.bytes.data(),values.data(),t.bytes.size());return t;
}
int main(int argc,char** argv) {
    if(argc!=5){std::cerr<<"Usage: ort_state_reference model.onnx request.json output-dir cpu|webgpu\n";return 2;}
    try {
        fs::path model=fs::absolute(argv[1]),out=fs::absolute(argv[3]);fs::create_directories(out);
        std::ifstream input(argv[2]);std::string data{std::istreambuf_iterator<char>(input),{}};auto request=json_parse(data);
        Ort::Env env(ORT_LOGGING_LEVEL_INFO,"backpack-reference");Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(8);opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        const bool webgpu=std::string(argv[4])=="webgpu";
        if(webgpu){const char* keys[]={"dawnBackendType","powerPreference","enableGraphCapture","validationMode","adapterIndex"};
            const char* values[]={"D3D12","high-performance","0","basic","0"};
            Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider(opts,"WebGPU",keys,values,5));}
        opts.EnableProfiling((out/"profile").c_str());
        Ort::Session session(env,model.c_str(),opts);Ort::AllocatorWithDefaultOptions allocator;
        std::unique_ptr<Ort::Session> embedding;
        std::vector<std::string> embeddingNames;
        std::vector<TensorData> embeddingInputs;
        if (request.has("embedding_model")) {
            const bool cpuEmbedding = request.has("embedding_provider") && request["embedding_provider"].as_string() == "cpu";
            auto embeddingOptions = cpuEmbedding ? Ort::SessionOptions{} : opts.Clone();
            if (cpuEmbedding) {
                embeddingOptions.SetIntraOpNumThreads(8);
                embeddingOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            }
            embeddingOptions.EnableProfiling((out/"embedding-profile").c_str());
            const auto embeddingPath = fs::absolute(request["embedding_model"].as_string());
            embedding = std::make_unique<Ort::Session>(env, embeddingPath.c_str(), embeddingOptions);
            for (size_t i = 0; i < embedding->GetInputCount(); ++i) {
                embeddingNames.emplace_back(embedding->GetInputNameAllocated(i,allocator).get());
                auto type = embedding->GetInputTypeInfo(i);
                auto info = type.GetTensorTypeAndShapeInfo();
                embeddingInputs.push_back({info.GetElementType(), info.GetShape(), {}});
            }
        }
        Ort::MemoryInfo memory=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        std::vector<std::string> names,outputNames;std::vector<const char*> rawNames,rawOutputs;
        std::vector<TensorData> initial;
        for(size_t i=0;i<session.GetInputCount();++i){
            names.emplace_back(session.GetInputNameAllocated(i,allocator).get());
            auto typeInfo=session.GetInputTypeInfo(i);
            auto info=typeInfo.GetTensorTypeAndShapeInfo();
            initial.push_back({info.GetElementType(),info.GetShape(),{}});
        }
        for(size_t i=0;i<session.GetOutputCount();++i)outputNames.emplace_back(session.GetOutputNameAllocated(i,allocator).get());
        for(auto& name:names)rawNames.push_back(name.c_str());for(auto& name:outputNames)rawOutputs.push_back(name.c_str());
        std::unordered_map<std::string,TensorData> states;
        size_t position=0,step=0;std::ofstream trace(out/"steps.jsonl");
        auto run=[&](const std::vector<int64_t>& tokens){
            const int64_t count=(int64_t)tokens.size();
            TensorData embedded{};
            if (embedding) {
                std::vector<TensorData> tensors;
                for (size_t i = 0; i < embeddingNames.size(); ++i) {
                    const auto& name = embeddingNames[i];
                    if (name == "input_ids") tensors.push_back(integers({1,count},tokens));
                    else if (name == "image_features" || name == "audio_features") {
                        // Text-only requests pass zero media tokens through the
                        // artifact's original embedding graph.
                        auto tensor = embeddingInputs[i];
                        if (tensor.shape.size() != 2 || tensor.shape[1] <= 0)
                            throw std::runtime_error("Unsupported empty media feature shape");
                        tensor.shape[0] = 0;
                        tensors.push_back(std::move(tensor));
                    } else throw std::runtime_error("Unsupported embedding input: " + name);
                }
                std::vector<Ort::Value> values;
                std::vector<const char*> inputNames;
                for (size_t i = 0; i < tensors.size(); ++i) {
                    auto& t = tensors[i];
                    values.push_back(Ort::Value::CreateTensor(memory,t.bytes.data(),t.bytes.size(),
                                                             t.shape.data(),t.shape.size(),t.type));
                    inputNames.push_back(embeddingNames[i].c_str());
                }
                const char* outputName = "inputs_embeds";
                auto output = embedding->Run(Ort::RunOptions{nullptr},inputNames.data(),values.data(),
                                             values.size(),&outputName,1);
                const auto info = output[0].GetTensorTypeAndShapeInfo();
                embedded.type = info.GetElementType();
                embedded.shape = info.GetShape();
                const auto* data = static_cast<const uint8_t*>(output[0].GetTensorRawData());
                embedded.bytes.assign(data,data+info.GetElementCount()*elementSize(embedded.type));
            }
            std::vector<TensorData> storage;storage.reserve(names.size());
            for(size_t i=0;i<names.size();++i){const auto& name=names[i];
                if(name=="input_ids")storage.push_back(integers({1,count},tokens));
                else if(name=="inputs_embeds" && embedding) storage.push_back(embedded);
                else if(name=="attention_mask")storage.push_back(integers({1,(int64_t)position+count},std::vector<int64_t>(position+count,1)));
                else if(name=="position_ids"){
                    const int64_t axes=initial[i].shape.size()==3?initial[i].shape[0]:1;
                    std::vector<int64_t> values(axes*count);for(int64_t a=0;a<axes;++a)for(int64_t t=0;t<count;++t)values[a*count+t]=position+t;
                    storage.push_back(integers(axes==1?std::vector<int64_t>{1,count}:std::vector<int64_t>{axes,1,count},values));
                }else if(name.rfind("past",0)==0){
                    auto found=states.find(name);
                    if(found!=states.end())storage.push_back(found->second);
                    else{auto tensor=initial[i];
                        if(request.has("state_shapes") && request["state_shapes"].has(name)) {
                            tensor.shape.clear();for(const auto& d:request["state_shapes"][name].as_array())tensor.shape.push_back(d.as_int());
                        }
                        const bool kv=name.find(".key")!=std::string::npos||name.find(".value")!=std::string::npos;
                        size_t elements=1;for(size_t d=0;d<tensor.shape.size();++d){if(tensor.shape[d]<0)tensor.shape[d]=(kv&&d==2)?0:1;elements*=tensor.shape[d];}
                        tensor.bytes.resize(elements*elementSize(tensor.type),0);storage.push_back(std::move(tensor));}
                }else throw std::runtime_error("Unsupported model input: "+name);
            }
            std::vector<Ort::Value> inputs;inputs.reserve(storage.size());
            for(auto& t:storage)inputs.push_back(Ort::Value::CreateTensor(memory,t.bytes.data(),t.bytes.size(),t.shape.data(),t.shape.size(),t.type));
            auto values=session.Run(Ort::RunOptions{nullptr},rawNames.data(),inputs.data(),inputs.size(),rawOutputs.data(),rawOutputs.size());
            std::ofstream debugMeta;
            const bool dumpAll = request.has("dump_all_outputs") && request["dump_all_outputs"].as_bool();
            if (dumpAll) debugMeta.open(out/("outputs-"+std::to_string(step)+".jsonl"));
            int32_t next=-1;float best=-INFINITY;
            for(size_t i=0;i<values.size();++i){auto info=values[i].GetTensorTypeAndShapeInfo();auto type=info.GetElementType();
                auto shape=info.GetShape();size_t elements=info.GetElementCount();const auto* bytes=(const uint8_t*)values[i].GetTensorRawData();
                if (dumpAll) {
                    const auto fileName = "tensor-"+std::to_string(i)+"-"+std::to_string(step)+".bin";
                    std::ofstream file(out/fileName,std::ios::binary);
                    file.write(reinterpret_cast<const char*>(bytes),elements*elementSize(type));
                    debugMeta << "{\"name\":" << std::quoted(outputNames[i]) << ",\"file\":" << std::quoted(fileName)
                              << ",\"type\":" << type << ",\"count\":" << elements << "}\n";
                }
                if(outputNames[i]=="logits"){
                    size_t vocab=(size_t)shape.back(),offset=elements-vocab;const auto* row=bytes+offset*elementSize(type);
                    for(size_t j=0;j<vocab;++j){float v=type==ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT?((const float*)row)[j]:halfValue(((const uint16_t*)row)[j]);
                        if(std::isnan(v))throw std::runtime_error("NaN reference logits");if(v>best){best=v;next=(int32_t)j;}}
                    std::ofstream dump(out/("logits-"+std::to_string(step)+".bin"),std::ios::binary);dump.write((const char*)row,vocab*elementSize(type));
                    trace<<"{\"step\":"<<step<<",\"position\":"<<position<<",\"input_tokens\":"<<count<<",\"next\":"<<next<<",\"logits_type\":"<<type<<",\"vocab\":"<<vocab<<"}\n";
                }else if(outputNames[i].rfind("present.",0)==0){
                    const auto suffix=outputNames[i].substr(8);std::string past="past."+suffix;
                    if(std::find(names.begin(),names.end(),past)==names.end())past="past_key_values."+suffix;
                    if(std::find(names.begin(),names.end(),past)==names.end())throw std::runtime_error("Unmapped state output: "+outputNames[i]);
                    states[past]={type,shape,std::vector<uint8_t>(bytes,bytes+elements*elementSize(type))};
                    if(outputNames[i].rfind("present.0.",0)==0){std::ofstream dump(out/(outputNames[i]+"-"+std::to_string(step)+".bin"),std::ios::binary);dump.write((const char*)bytes,elements*elementSize(type));}
                }
            }
            position+=tokens.size();++step;trace.flush();return next;
        };
        int32_t next=-1;for(const auto& batch:request["batches"].as_array()){std::vector<int64_t> tokens;for(auto& t:batch.as_array())tokens.push_back(t.as_int());next=run(tokens);}
        std::vector<int32_t> generated;int limit=request["max_new_tokens"].as_int();
        for(int i=0;i<limit;++i){generated.push_back(next);bool stop=false;for(auto& id:request["stop_ids"].as_array())if(id.as_int()==next)stop=true;
            if(stop)break;next=run({next});}
        std::ofstream result(out/"result.json");result<<"{\"ort_version\":\""<<Ort::GetVersionString()<<"\",\"tokens\":[";
        for(size_t i=0;i<generated.size();++i)result<<(i?",":"")<<generated[i];result<<"]}\n";
        if (embedding) embedding->EndProfilingAllocated(allocator);
        auto profile=session.EndProfilingAllocated(allocator);std::cout<<"profile="<<profile.get()<<"\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
}
