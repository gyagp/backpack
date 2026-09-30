// Exact-token and chat conformance runner for a pinned native ORT GenAI DLL.
// Usage: genai_state_reference.exe model-directory request.json result.json
#include "ort_genai.h"
#include "json_parser.h"
#include <algorithm>
#include <chrono>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <sstream>

static JsonValue readJson(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open " + path.string());
    return json_parse(std::string{std::istreambuf_iterator<char>(file), {}});
}
static std::string quoteJson(const std::string& value) {
    std::string result = "\"";
    for (unsigned char ch : value) {
        if (ch == '"' || ch == '\\') { result += '\\'; result += ch; }
        else if (ch < 32) { char escaped[7]; std::snprintf(escaped, sizeof(escaped), "\\u%04x", static_cast<unsigned>(ch)); result += escaped; }
        else result += ch;
    }
    return result + '"';
}
int main(int argc, char** argv) {
    if (argc != 4) return 2;
    // GenAI device initialization can throw from a noexcept boundary. Keep the
    // underlying message visible instead of reporting only Windows 0xC0000409.
    std::set_terminate([] {
        if (auto error = std::current_exception()) try { std::rethrow_exception(error); }
        catch (const std::exception& e) { std::cerr << "GenAI initialization terminated: " << e.what() << "\n"; }
        catch (...) { std::cerr << "GenAI initialization terminated with an unknown exception\n"; }
        std::_Exit(127);
    });
    OgaHandle lifetime;
    try {
        const auto request = readJson(argv[2]);
        const bool capture = request.has("graph_capture") ? request["graph_capture"].as_bool() : true;
        const bool verbose = request.has("verbose") && request["verbose"].as_bool();
        const int maxTokens = request.has("max_new_tokens") ? request["max_new_tokens"].as_int() : 128;
        const int maxLength = request.has("max_seq_len") ? request["max_seq_len"].as_int() : 1024;
        const int repetitions = request.has("repetitions") ? request["repetitions"].as_int() : 1;
        const int warmups = request.has("warmup_runs") ? request["warmup_runs"].as_int() : 0;
        const bool benchmark = request.has("benchmark") && request["benchmark"].as_bool();
        const std::string profilePrefix = request.has("profile_prefix") ? request["profile_prefix"].as_string() : "";
        if (repetitions < 1 || warmups < 0) throw std::runtime_error("Invalid repetition counts");
        if (maxTokens < 1 || maxLength < maxTokens) throw std::runtime_error("Invalid token limits");
        auto config = OgaConfig::Create(argv[1]);
        config->SetProviderOption("webgpu", "dawnBackendType", "D3D12");
        config->SetProviderOption("webgpu", "powerPreference", "high-performance");
        // GenAI's initialization session does not forward adapterIndex. Use the
        // same high-performance automatic selector for initialization and model
        // sessions, avoiding a conflicting selector on ORT's shared context.
        config->SetProviderOption("webgpu", "enableGraphCapture", capture ? "1" : "0");
        if (verbose) config->Overlay("{\"model\":{\"decoder\":{\"session_options\":{\"log_severity_level\":0}}}}");
        if (!profilePrefix.empty()) {
            const auto overlay = "{\"model\":{\"decoder\":{\"session_options\":{\"enable_profiling\":" + quoteJson(profilePrefix) + "}}}}";
            config->Overlay(overlay.c_str());
        }
        const auto manifest = readJson(std::filesystem::path(argv[1]) / "genai_config.json");
        const int prefillChunk = manifest.has("search") && manifest["search"].has("chunk_size")
            ? manifest["search"]["chunk_size"].as_int() : 0;
        auto model = OgaModel::Create(*config);
        auto tokenizer = OgaTokenizer::Create(*model);
        auto params = OgaGeneratorParams::Create(*model);
        params->SetSearchOption("max_length", maxLength);
        params->SetSearchOptionBool("do_sample", false);
        std::vector<std::vector<int32_t>> batches;
        if (request.has("batches")) {
            for (const auto& batch : request["batches"].as_array()) {
                std::vector<int32_t> tokens;
                for (const auto& token : batch.as_array()) tokens.push_back(token.as_int());
                batches.push_back(std::move(tokens));
            }
        } else {
            const auto type = manifest["model"]["type"].as_string();
            const auto prompt = request["prompt"].as_string();
            std::vector<std::string> texts;
            if (type == "gemma4") {
                texts = {"<|turn>system\nYou are a helpful AI assistant.<turn|>\n",
                         "<|turn>user\n" + prompt + "<turn|>\n<|turn>model\n"};
            } else if (type.find("qwen3_5") != std::string::npos) {
                texts = {"You are a helpful AI assistant.",
                         "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"};
            } else throw std::runtime_error("Supply exact token batches for this model type: " + type);
            for (const auto& text : texts) {
                auto sequence = OgaSequences::Create();
                tokenizer->Encode(text.c_str(), *sequence);
                batches.emplace_back(sequence->SequenceData(0), sequence->SequenceData(0) + sequence->SequenceCount(0));
            }
        }
        size_t inputTokens = 0;
        for (const auto& batch : batches) {
            if (batch.empty()) throw std::runtime_error("Empty input batch");
            inputTokens += batch.size();
        }
        if (batches.empty() || inputTokens + maxTokens > static_cast<size_t>(maxLength))
            throw std::runtime_error("Request exceeds context limit");
        auto stopIds = tokenizer->GetEosTokenIds();
        // Chat-end may be declared by tokenizer_config rather than the GenAI
        // search configuration. Preserve that model-defined stop boundary.
        const auto tokenizerConfig = readJson(std::filesystem::path(argv[1]) / "tokenizer_config.json");
        if (tokenizerConfig.has("eos_token")) {
            const auto& value = tokenizerConfig["eos_token"];
            const auto token = value.is_string() ? value.as_string() : value["content"].as_string();
            auto sequence = OgaSequences::Create();
            tokenizer->Encode(token.c_str(), *sequence);
            if (sequence->SequenceCount(0) == 1) stopIds.push_back(sequence->SequenceData(0)[0]);
        }
        if (request.has("stop_ids")) {
            stopIds.clear();
            for (const auto& id : request["stop_ids"].as_array()) stopIds.push_back(id.as_int());
        }
        if (benchmark) {
            // Keep the exact requested output count, including the first
            // prediction, even if the model predicts its normal EOS token.
            params->SetSearchOption("min_length", inputTokens + maxTokens);
            stopIds.clear();
        }
        auto generator = OgaGenerator::Create(*model, *params);
        using Clock = std::chrono::steady_clock;
        struct RunResult {
            std::vector<int32_t> tokens;
            double resetMs = 0, prefillMs = 0, decodeMs = 0;
            size_t finalSequenceLength = 0;
            bool stopped = false;
        };
        std::vector<RunResult> measured;
        std::vector<int32_t> expected;
        for (int repetition = 0; repetition < warmups + repetitions; ++repetition) {
            RunResult result;
            if (repetition) {
                const auto resetStart = Clock::now();
                generator->RewindTo(0);
                result.resetMs = std::chrono::duration<double, std::milli>(Clock::now() - resetStart).count();
            }
            result.tokens.reserve(maxTokens);
            bool done = false;
            auto generate = [&]() {
                generator->GenerateNextToken();
                const auto tokens = generator->GetNextTokens();
                if (tokens.size() != 1) throw std::runtime_error("Expected one next token");
                result.tokens.push_back(tokens[0]);
                result.stopped = std::find(stopIds.begin(), stopIds.end(), tokens[0]) != stopIds.end();
                done = generator->IsDone();
            };
            const auto start = Clock::now();
            for (const auto& batch : batches) generator->AppendTokens(batch.data(), batch.size());
            generate();  // Prefill includes the first prediction.
            const auto firstToken = Clock::now();
            for (int i = 1; i < maxTokens && !done && !result.stopped; ++i) generate();
            const auto end = Clock::now();
            result.prefillMs = std::chrono::duration<double, std::milli>(firstToken - start).count();
            result.decodeMs = std::chrono::duration<double, std::milli>(end - firstToken).count();
            result.finalSequenceLength = generator->TokenCount();
            // GenAI may omit a terminal EOS from its stored sequence even
            // though GetNextTokens returned it. Fixed-count benchmarks disable
            // EOS stopping and must still account for every requested token.
            if (benchmark && (result.finalSequenceLength != inputTokens + result.tokens.size() ||
                result.tokens.size() != static_cast<size_t>(maxTokens)))
                throw std::runtime_error("Native reference did not execute the exact workload");
            if (repetition && result.tokens != expected) {
                std::ofstream failure(std::string(argv[3]) + ".failure.json");
                failure << "{\"error\":\"continuation changed after reset\",\"repetition\":" << repetition
                        << ",\"input_tokens\":" << inputTokens << ",\"final_sequence_length\":" << result.finalSequenceLength
                        << ",\"expected\":[";
                for (size_t i = 0; i < expected.size(); ++i) failure << (i ? "," : "") << expected[i];
                failure << "],\"actual\":[";
                for (size_t i = 0; i < result.tokens.size(); ++i) failure << (i ? "," : "") << result.tokens[i];
                failure << "]}\n";
                throw std::runtime_error("Native reference continuation changed after warmup/reset");
            }
            expected = result.tokens;
            std::cerr << "[reference-run] " << (repetition < warmups ? "warmup" : "measured")
                      << " repetition=" << repetition << " tokens=" << result.tokens.size()
                      << " reset_ms=" << result.resetMs << " prefill_ms=" << result.prefillMs << " decode_ms=" << result.decodeMs << "\n";
            if (repetition >= warmups) measured.push_back(std::move(result));
        }
        const auto& generated = measured.front().tokens;
        const bool stopped = measured.front().stopped;
        const auto text = tokenizer->Decode(generated.data(), generated.size() - (stopped ? 1 : 0));
        std::ofstream output(argv[3]);
        output << std::setprecision(12) << "{\"graph_capture_requested\":" << (capture ? "true" : "false")
               << ",\"profiling\":" << (profilePrefix.empty() ? "false" : "true")
               << ",\"max_seq_len\":" << maxLength << ",\"prefill_chunk\":" << prefillChunk
               << ",\"prefill_includes_first_token\":true"
               << ",\"input_tokens\":" << inputTokens << ",\"text\":" << quoteJson(static_cast<const char*>(text))
               << ",\"tokens\":[";
        for (size_t i = 0; i < generated.size(); ++i) output << (i ? "," : "") << generated[i];
        output << "],\"warmup_runs\":" << warmups << ",\"reuse_generator\":true,\"runs\":[";
        for (size_t i = 0; i < measured.size(); ++i) {
            const auto& result = measured[i];
            const size_t decodeCalls = result.tokens.size() - 1;
            output << (i ? "," : "") << "{\"reset_ms\":" << result.resetMs << ",\"prefill_ms\":" << result.prefillMs
                   << ",\"decode_ms\":" << result.decodeMs
                   << ",\"prefill_tok_s\":" << inputTokens * 1000.0 / result.prefillMs
                   << ",\"decode_tok_s\":" << (decodeCalls ? decodeCalls * 1000.0 / result.decodeMs : 0)
                   << ",\"decode_calls\":" << decodeCalls
                   << ",\"generated_tokens\":" << result.tokens.size()
                   << ",\"final_position\":" << result.finalSequenceLength - 1
                   << ",\"final_sequence_length\":" << result.finalSequenceLength << ",\"tokens\":[";
            for (size_t j = 0; j < result.tokens.size(); ++j) output << (j ? "," : "") << result.tokens[j];
            output << "]}";
        }
        output << "]}\n";
        if (!output) throw std::runtime_error("Cannot write reference result");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
