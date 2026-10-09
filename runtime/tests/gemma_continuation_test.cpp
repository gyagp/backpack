// Real-model continuation regression. Run reference and queued modes separately
// to keep only one model resident, then compare the JSON and binary logits.
#include "backpack.h"
#include "lm_session.h"
#include "gpu_context.h"
#include "../../apps/common/app_common.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: backpack_continuation_test MODEL [application-]reference|queued OUTPUT.json\n";
        return 2;
    }
    try {
        const char* host=std::getenv("COMPUTERNAME");if(!host || _stricmp(host,"webgfx-104")!=0)return 2;
        std::string mode = argv[2];
        const bool boundary = mode.rfind("boundary-", 0) == 0;
        if(boundary)mode.erase(0,9);
        const bool application = mode.rfind("application-", 0) == 0;
        if (application) mode.erase(0, 12);
        if (mode != "reference" && mode != "queued") throw std::runtime_error("Unknown test mode");
        const bool reference = mode == "reference";
        auto device = app::createDevice("d3d12");
        if (device.GetName() != "NVIDIA GeForce RTX 5080")
            throw std::runtime_error("This regression targets RTX 5080 only");
        bp::LmOptions options;
        options.maxSeqLen = boundary ? 128 : 640;
        options.fastDecode = !reference;
        options.prefillChunkSize = 32;
        auto session = bp::LmSession::Create(device, argv[1], options);
        if (!session.IsValid()) throw std::runtime_error("Cannot load model");
        const auto cfg = session.GetConfig();
        if(cfg.arch!="gemma4" || cfg.vocabSize!=262144)throw std::runtime_error("Expected cared Gemma E2B");
        const bool standard = cfg.format != "onnx_generic";
        const auto folder = std::filesystem::path(argv[3]).parent_path();
        std::ofstream out(argv[3]);
        if (!out) throw std::runtime_error("Cannot open result file");
        if(boundary) {
            out<<"{\"cases\":[";bool firstCase=true;
            const auto one=session.TokenizeRaw(" A");if(one.size()!=1)throw std::runtime_error("Expected one token");
            for(int count:{119,120})for(int round=0;round<2;++round) {
                session.Reset();std::vector<int32_t> input(count,one[0]),stream;
                int32_t next=session.Prefill(input.data(),uint32_t(input.size()));stream.push_back(next);
                auto step=[&](){next=reference?session.Prefill(&next,1):session.Decode();stream.push_back(next);};
                for(int i=0;i<5;++i)step();const auto position=session.GetPosition();
                const int32_t extra[]={one[0],one[0],one[0],one[0],one[0]};
                if(session.Prefill(extra,5)!=-1 || session.GetPosition()!=position)throw std::runtime_error("Invalid append changed position");
                const auto label=std::to_string(count)+"-"+std::to_string(round);
                const auto dump=(folder/("prefill-"+label+".f32")).string();_putenv_s("BP_DUMP_PREFILL_LOGITS",dump.c_str());
                next=session.Prefill(extra,2);_putenv_s("BP_DUMP_PREFILL_LOGITS","");stream.push_back(next);
                while(session.GetPosition()<127)step();
                auto logits=session.DecodeLogits();if(logits.size()!=size_t(cfg.vocabSize))throw std::runtime_error("Missing final logits");
                if(!std::all_of(logits.begin(),logits.end(),[](float x){return std::isfinite(x);}))throw std::runtime_error("Nonfinite logits");
                std::ofstream raw(folder/("decode-"+label+".f32"),std::ios::binary);raw.write(reinterpret_cast<const char*>(logits.data()),logits.size()*4);
                stream.push_back(int32_t(std::max_element(logits.begin(),logits.end())-logits.begin()));
                if(session.GetPosition()!=128 || session.Decode()!=-1 || !session.DecodeLogits().empty())throw std::runtime_error("Capacity boundary failed");
                if(!firstCase)out<<',';firstCase=false;
                out<<"{\"count\":"<<count<<",\"round\":"<<round<<",\"position\":"<<session.GetPosition()<<",\"tokens\":[";
                for(size_t i=0;i<stream.size();++i)out<<(i?",":"")<<stream[i];out<<"]}";out.flush();
                auto* gpu=static_cast<GPUContext*>(device.GetGPUContext());if(gpu->deviceLost||gpu->executionError)throw std::runtime_error("GPU failure");
            }
            out<<"]}";session.Reset();return 0;
        }
        if (application) {
            const auto prompt = app::applyChatTemplate("Explain in one sentence why the sky is blue.", cfg.arch);
            const auto tokens = session.Tokenize(prompt);
            const auto follow = cfg.arch == "gemma4"
                ? app::applyGemma4UserTemplate("Write a short sentence about a red balloon.")
                : app::applyQwenUserTemplate("Write a short sentence about a red balloon.");
            out << "{\"cases\":[";
            bool firstCase = true;
            std::ofstream memory(folder / "memory.json");
            memory << '[';
            for (int stop : {1, 5, 8}) {
                for (int round = 0; round < 2; ++round) {
                    session.Reset();
                    std::string initial;
                    if (reference) {
                        int32_t next = session.Prefill(tokens.data(), uint32_t(tokens.size()));
                        int emitted = 0;
                        for (int steps = 0; emitted < stop && steps < 32; ++steps) {
                            if (next < 0 || next == session.GetEosTokenId()) break;
                            const auto piece = session.Detokenize(next);
                            if (!(piece.size() >= 2 && piece.front() == '<' && piece.back() == '>')) {
                                initial += piece;
                                if (++emitted == stop) break;
                            }
                            next = standard ? session.Prefill(&next, 1) : session.Decode();
                        }
                        if (emitted != stop) throw std::runtime_error("Reference ended before cancellation boundary");
                    } else {
                        int calls = 0;
                        initial = session.Generate(prompt, 32, {}, [&](const std::string&) { return ++calls < stop; });
                        if (calls != stop) throw std::runtime_error("Callback ended before cancellation boundary");
                    }
                    const auto boundary = session.GetPosition();
                    bp::SamplingParams sampling;
                    sampling.temperature = 0.7f;
                    sampling.topK = 5;
                    sampling.seed = 1234;
                    const auto sampled = session.Generate(follow, 8, sampling, nullptr, false);
                    if (sampled.empty()) throw std::runtime_error("Empty sampled continuation");
                    if (!firstCase) out << ',';
                    out << "{\"stop\":" << stop << ",\"round\":" << round << ",\"boundary\":" << boundary
                        << ",\"position\":" << session.GetPosition() << ",\"initial\":\"" << app::jsonEscape(initial)
                        << "\",\"sampled\":\"" << app::jsonEscape(sampled) << "\"}";
                    session.Reset();
                    session.TrimMemory();
                    auto* gpu = static_cast<GPUContext*>(device.GetGPUContext());
                    if (!firstCase) memory << ',';
                    memory << "{\"stop\":" << stop << ",\"round\":" << round
                           << ",\"resident_bytes\":" << gpu->totalAllocatedBytes << '}';
                    firstCase = false;
                    out.flush();
                    memory.flush();
                    if (gpu->deviceLost || gpu->executionError) throw std::runtime_error("GPU failure");
                    std::cout << "APPLICATION stop=" << stop << " round=" << round << " boundary=" << boundary << std::endl;
                }
            }
            out << "]}";
            memory << ']';
            return 0;
        }
        out << "{\"cases\":[";
        bool first = true;
        const int initialSteps[] = {1, 5, 6, 8, 8};
        for (int scenario = 0; scenario < 5; ++scenario) {
            std::string prompt = "Explain in one sentence why the sky is blue.";
            if (scenario == 4)
                for (int i = 0; i < 16; ++i) prompt = "A library keeps books by title and author. " + prompt;
            const auto tokens = session.Tokenize(app::applyChatTemplate(prompt, cfg.arch));
            if (scenario == 4 && tokens.size() <= 128)
                throw std::runtime_error("Long-prompt scenario must exercise staged prefill");
            for (int round = 0; round < 2; ++round) {
                session.Reset();
                std::vector<int32_t> stream;
                int32_t next;
                if (scenario == 3) {
                    (void)session.Prefill(tokens.data(), 7);
                    next = session.Prefill(tokens.data() + 7, uint32_t(tokens.size() - 7));
                } else next = session.Prefill(tokens.data(), uint32_t(tokens.size()));
                stream.push_back(next);
                auto step = [&]() {
                    // Known one-token input consumes exactly the committed
                    // token without speculative future state, using the same
                    // pooled arithmetic as queued Standard decoding.
                    next = reference && standard ? session.Prefill(&next, 1) : session.Decode();
                    if (next < 0) throw std::runtime_error("Unexpected context exhaustion");
                    stream.push_back(next);
                };
                for (int i = 0; i < initialSteps[scenario]; ++i) step();
                const auto boundary = session.GetPosition();
                uint32_t expectedPosition = boundary;
                for (int turn = 0; turn < 3; ++turn) {
                    const std::string question = turn == 1
                        ? "Answer with one word. Name the capital of France."
                        : "Answer with only the number. What is 2 + 2?";
                    const auto follow = session.TokenizeRaw(cfg.arch == "gemma4"
                        ? app::applyGemma4UserTemplate(question) : app::applyQwenUserTemplate(question));
                    const auto label = std::to_string(scenario) + "-" + std::to_string(round) + "-" + std::to_string(turn);
                    const auto dump = (folder / ("prefill-" + label + ".f32")).string();
                    if (standard) _putenv_s("BP_DUMP_PREFILL_LOGITS", dump.c_str());
                    next = session.Prefill(follow.data(), uint32_t(follow.size()));
                    _putenv_s("BP_DUMP_PREFILL_LOGITS", "");
                    stream.push_back(next);
                    const int steps = turn == 0 ? 5 : (turn == 1 ? 6 : 1);
                    for (int i = 0; i < steps; ++i) step();
                    expectedPosition += uint32_t(follow.size()) + steps;
                    if (session.GetPosition() != expectedPosition)
                        throw std::runtime_error("Appended-input position mismatch");
                }
                // Switch directly from queued argmax to raw logits. The API
                // consumes lastToken but leaves token selection to the caller.
                const auto logits = session.DecodeLogits();
                if (logits.size() != size_t(cfg.vocabSize))
                    throw std::runtime_error("Wrong vocabulary size");
                if (!std::all_of(logits.begin(), logits.end(), [](float value) { return std::isfinite(value); }))
                    throw std::runtime_error("Non-finite continuation logits");
                std::ofstream raw(folder / ("decode-" + std::to_string(scenario) + "-" + std::to_string(round) + ".f32"), std::ios::binary);
                raw.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
                stream.push_back(int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin()));
                if (session.GetPosition() != expectedPosition + 1)
                    throw std::runtime_error("Raw-logit transition position mismatch");
                if (!first) out << ',';
                first = false;
                out << "{\"scenario\":" << scenario << ",\"round\":" << round
                    << ",\"input\":" << tokens.size() << ",\"boundary\":" << boundary
                    << ",\"position\":" << session.GetPosition() << ",\"tokens\":[";
                for (size_t i = 0; i < stream.size(); ++i) out << (i ? "," : "") << stream[i];
                out << "]}";
                out.flush();
                std::cout << "CASE " << scenario << " round=" << round << " input=" << tokens.size()
                          << " final=" << session.GetPosition() << std::endl;
                auto* gpu = static_cast<GPUContext*>(device.GetGPUContext());
                if (gpu->deviceLost || gpu->executionError) throw std::runtime_error("GPU failure");
            }
        }
        out << "]}";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
