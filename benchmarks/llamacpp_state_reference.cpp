// Resident, fixed-text reference for the pinned llama.cpp b11295 public C ABI.
#define NOMINMAX
#include <windows.h>
#include <llama.h>
#include "json_parser.h"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
static constexpr const char* kCommit = "3b3d022b823abaa62a467b26a44e10659e080ee7";
static bool verboseLogging = false;
static void logMessage(ggml_log_level level, const char* text, void*) {
    if (verboseLogging || level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) std::fputs(text, stderr);
}

static JsonValue readJson(const fs::path& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot read " + path.string());
    return json_parse(std::string{std::istreambuf_iterator<char>(file), {}});
}
static std::string quote(const std::string& value) {
    std::string result = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { result += '\\'; result += c; }
        else if (c < 32) { char text[7]; std::snprintf(text, sizeof(text), "\\u%04x", c); result += text; }
        else result += c;
    }
    return result + '"';
}

struct Api {
    std::vector<HMODULE> modules;
    explicit Api(const fs::path& directory) {
        for (const auto* name : {L"ggml-base.dll", L"ggml.dll", L"llama.dll"}) {
            auto module = LoadLibraryExW((directory / name).c_str(), nullptr,
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            if (!module) throw std::runtime_error("Cannot load pinned llama.cpp library");
            modules.push_back(module);
        }
    }
    template<class T> T get(const char* name) const {
        for (auto module : modules) if (auto symbol = GetProcAddress(module, name)) return reinterpret_cast<T>(symbol);
        throw std::runtime_error(std::string("Missing pinned C API symbol: ") + name);
    }
};
#define LOAD(name) const auto p_##name = api.get<decltype(&name)>(#name)

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: llamacpp_state_reference runtime-directory model.gguf request.json result.json\n";
        return 2;
    }
    try {
        const fs::path runtime = fs::absolute(argv[1]);
        const auto manifest = readJson(runtime / "build-manifest.json");
        if (manifest["release"].as_string() != "b11295" || manifest["source_commit"].as_string() != kCommit)
            throw std::runtime_error("Runtime differs from this helper's pinned public C ABI");
        const auto request = readJson(argv[3]);
        verboseLogging = request.has("verbose") && request["verbose"].as_bool();
        const int maxTokens = request.has("max_new_tokens") ? request["max_new_tokens"].as_int() : 128;
        const int maxLength = request.has("max_seq_len") ? request["max_seq_len"].as_int() : 640;
        const int warmups = request.has("warmup_runs") ? request["warmup_runs"].as_int() : 1;
        const int repetitions = request.has("repetitions") ? request["repetitions"].as_int() : 5;
        const bool backendSampling = !request.has("backend_sampling") || request["backend_sampling"].as_bool();
        const bool addSpecialTokens = request.has("add_special_tokens") && request["add_special_tokens"].as_bool();
        if (maxTokens <= 0 || maxLength <= 0 || warmups < 0 || repetitions <= 0)
            throw std::runtime_error("Invalid request limits");
        Api api(runtime);
        LOAD(ggml_backend_load_all_from_path);
        LOAD(ggml_backend_dev_by_name);
        LOAD(ggml_backend_dev_description);
        LOAD(ggml_backend_dev_type);
        LOAD(ggml_log_set);
        LOAD(llama_log_set);
        LOAD(llama_backend_init);
        LOAD(llama_model_default_params);
        LOAD(llama_context_default_params);
        LOAD(llama_model_load_from_file);
        LOAD(llama_model_free);
        LOAD(llama_model_get_vocab);
        LOAD(llama_vocab_n_tokens);
        LOAD(llama_init_from_model);
        LOAD(llama_free);
        LOAD(llama_n_ctx_seq);
        LOAD(llama_get_memory);
        LOAD(llama_memory_clear);
        LOAD(llama_memory_seq_pos_max);
        LOAD(llama_tokenize);
        LOAD(llama_detokenize);
        LOAD(llama_decode);
        LOAD(llama_sampler_chain_default_params);
        LOAD(llama_sampler_chain_init);
        LOAD(llama_sampler_chain_add);
        LOAD(llama_sampler_init_greedy);
        LOAD(llama_sampler_free);
        LOAD(llama_sampler_reset);
        LOAD(llama_sampler_accept);
        LOAD(llama_sampler_sample);
        LOAD(llama_get_sampled_token_ith);
        p_ggml_log_set(logMessage, nullptr);
        p_llama_log_set(logMessage, nullptr);
        p_ggml_backend_load_all_from_path(runtime.string().c_str());
        p_llama_backend_init();
        auto device = p_ggml_backend_dev_by_name("Vulkan0");
        if (!device || p_ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU ||
            std::string(p_ggml_backend_dev_description(device)) != "NVIDIA GeForce RTX 5080")
            throw std::runtime_error("This reference requires Vulkan0 / NVIDIA GeForce RTX 5080");
        ggml_backend_dev_t devices[]{device, nullptr};
        auto modelParams = p_llama_model_default_params();
        modelParams.devices = devices;
        modelParams.n_gpu_layers = 99;
        modelParams.split_mode = LLAMA_SPLIT_MODE_NONE;
        std::unique_ptr<llama_model, decltype(p_llama_model_free)> model(
            p_llama_model_load_from_file(argv[2], modelParams), p_llama_model_free);
        if (!model) throw std::runtime_error("Cannot load reference model");
        const auto* vocab = p_llama_model_get_vocab(model.get());
        std::vector<llama_token> prompt;
        const std::string promptText = request.has("prompt") ? request["prompt"].as_string() : "";
        if (!promptText.empty()) {
            if (promptText.size() > size_t(std::numeric_limits<int32_t>::max())) throw std::runtime_error("Prompt is too long");
            int count = p_llama_tokenize(vocab, promptText.data(), int(promptText.size()), nullptr, 0, addSpecialTokens, true);
            if (count == std::numeric_limits<int32_t>::min()) throw std::runtime_error("Tokenizer overflow");
            prompt.resize(size_t(count < 0 ? -count : count));
            count = p_llama_tokenize(vocab, promptText.data(), int(promptText.size()), prompt.data(), int(prompt.size()), addSpecialTokens, true);
            if (count < 0) throw std::runtime_error("Tokenization failed");
            prompt.resize(size_t(count));
        }
        if (request.has("input_tokens")) {
            std::vector<llama_token> expected;
            for (const auto& token : request["input_tokens"].as_array()) expected.push_back(token.as_int());
            if (!promptText.empty() && prompt != expected) throw std::runtime_error("Reference tokenizer differs from supplied input tokens");
            prompt = std::move(expected);
        }
        if (prompt.empty() || prompt.size() + size_t(maxTokens) > size_t(maxLength))
            throw std::runtime_error("Request exceeds context limit");
        for (auto token : prompt) if (token < 0 || token >= p_llama_vocab_n_tokens(vocab)) throw std::runtime_error("Invalid input token");
        auto samplerParams = p_llama_sampler_chain_default_params();
        samplerParams.no_perf = true;
        std::unique_ptr<llama_sampler, decltype(p_llama_sampler_free)> sampler(
            p_llama_sampler_chain_init(samplerParams), p_llama_sampler_free);
        p_llama_sampler_chain_add(sampler.get(), p_llama_sampler_init_greedy());
        llama_sampler_seq_config sampling{0, sampler.get()};
        auto contextParams = p_llama_context_default_params();
        contextParams.n_ctx = uint32_t(maxLength);
        contextParams.n_batch = uint32_t(prompt.size());
        contextParams.n_ubatch = uint32_t(prompt.size());
        contextParams.n_seq_max = 1;
        contextParams.n_outputs_max = 1;
        contextParams.n_outputs_max_per_seq = 1;
        contextParams.n_threads = 24;
        contextParams.n_threads_batch = 24;
        contextParams.no_perf = true;
        contextParams.type_k = contextParams.type_v = GGML_TYPE_F16;
        contextParams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        contextParams.samplers = backendSampling ? &sampling : nullptr;
        contextParams.n_samplers = backendSampling ? 1 : 0;
        std::unique_ptr<llama_context, decltype(p_llama_free)> context(
            p_llama_init_from_model(model.get(), contextParams), p_llama_free);
        if (!context) throw std::runtime_error("Cannot create reference context");
        const uint32_t effectiveCapacity = p_llama_n_ctx_seq(context.get());
        // llama.cpp pads its cache allocation; preserve the requested workload
        // bound and expose the allocation capacity instead of hiding it.
        if (effectiveCapacity < uint32_t(maxLength) || effectiveCapacity - uint32_t(maxLength) > 255)
            throw std::runtime_error("Unexpected effective context capacity");
        auto memory = p_llama_get_memory(context.get());
        if (!memory) throw std::runtime_error("Reference model has no resettable memory");
        std::vector<llama_pos> positions(prompt.size());
        std::vector<int8_t> outputs(prompt.size(), 0); outputs.back() = 1;
        for (size_t i = 0; i < positions.size(); ++i) positions[i] = llama_pos(i);
        struct Run { double prefillMs, decodeMs; int finalPosition; std::vector<llama_token> tokens; };
        std::vector<Run> runs;
        std::vector<llama_token> expected;
        auto sample = [&]() {
            if (!backendSampling) return p_llama_sampler_sample(sampler.get(), context.get(), -1);
            auto token = p_llama_get_sampled_token_ith(context.get(), -1);
            if (token == LLAMA_TOKEN_NULL) throw std::runtime_error("Backend did not produce a sampled token");
            p_llama_sampler_accept(sampler.get(), token);
            return token;
        };
        using Clock = std::chrono::steady_clock;
        for (int repetition = 0; repetition < warmups + repetitions; ++repetition) {
            p_llama_memory_clear(memory, true);
            p_llama_sampler_reset(sampler.get());
            Run run{}; run.tokens.reserve(size_t(maxTokens));
            llama_batch batch{};
            batch.n_tokens = int32_t(prompt.size()); batch.token = prompt.data();
            batch.pos = positions.data(); batch.logits = outputs.data();
            const auto begin = Clock::now();
            if (p_llama_decode(context.get(), batch) != 0) throw std::runtime_error("Reference prefill failed");
            auto next = sample(); run.tokens.push_back(next);
            const auto first = Clock::now();
            for (int i = 1; i < maxTokens; ++i) {
                llama_pos position = llama_pos(prompt.size() + size_t(i) - 1);
                int8_t output = 1;
                llama_batch step{}; step.n_tokens = 1; step.token = &next; step.pos = &position; step.logits = &output;
                if (p_llama_decode(context.get(), step) != 0) throw std::runtime_error("Reference decode failed");
                next = sample(); run.tokens.push_back(next);
            }
            const auto end = Clock::now();
            run.prefillMs = std::chrono::duration<double, std::milli>(first - begin).count();
            run.decodeMs = std::chrono::duration<double, std::milli>(end - first).count();
            run.finalPosition = p_llama_memory_seq_pos_max(memory, 0) + 1;
            if (run.finalPosition != int(prompt.size()) + maxTokens - 1) throw std::runtime_error("Unexpected final sequence position");
            if (expected.empty()) expected = run.tokens;
            else if (expected != run.tokens) throw std::runtime_error("Continuation changed across identical resets");
            if (repetition >= warmups) runs.push_back(std::move(run));
        }
        std::vector<char> decoded(expected.size() * 32 + 1024);
        int length = p_llama_detokenize(vocab, expected.data(), int(expected.size()), decoded.data(), int(decoded.size()), true, false);
        if (length < 0) {
            decoded.resize(size_t(-length));
            length = p_llama_detokenize(vocab, expected.data(), int(expected.size()), decoded.data(), int(decoded.size()), true, false);
        }
        if (length < 0) throw std::runtime_error("Detokenization failed");
        std::ofstream out(argv[4]);
        out << std::setprecision(12) << "{\"runtime_commit\":" << quote(kCommit)
            << ",\"backend\":\"vulkan\",\"adapter\":\"NVIDIA GeForce RTX 5080\",\"device\":\"Vulkan0\""
            << ",\"benchmark_protocol\":\"llm-fixed-text-reuse-first-token-v2\",\"warmup_runs\":" << warmups
            << ",\"reuse_generator\":true,\"prefill_includes_first_token\":true,\"max_seq_len\":" << maxLength
            << ",\"effective_context_capacity\":" << effectiveCapacity << ",\"diagnostics\":" << (verboseLogging ? "true" : "false")
            << ",\"sampling\":" << quote(backendSampling ? "backend-greedy" : "cpu-greedy")
            << ",\"add_special_tokens\":" << (addSpecialTokens ? "true" : "false")
            << ",\"text\":" << quote(std::string(decoded.data(), size_t(length))) << ",\"prompt_token_ids\":[";
        for (size_t i = 0; i < prompt.size(); ++i) out << (i ? "," : "") << prompt[i];
        out << "],\"runs\":[";
        for (size_t i = 0; i < runs.size(); ++i) {
            const auto& run = runs[i];
            out << (i ? "," : "") << "{\"input_tokens\":" << prompt.size() << ",\"generated_tokens\":" << maxTokens
                << ",\"decode_sample_tokens\":" << maxTokens - 1 << ",\"final_position\":" << run.finalPosition
                << ",\"prefill_ms\":" << run.prefillMs << ",\"decode_ms\":" << run.decodeMs
                << ",\"prefill_tok_s\":" << prompt.size() * 1000.0 / run.prefillMs
                << ",\"decode_tok_s\":" << (maxTokens > 1 ? (maxTokens - 1) * 1000.0 / run.decodeMs : 0)
                << ",\"generated_token_ids\":[";
            for (size_t j = 0; j < run.tokens.size(); ++j) out << (j ? "," : "") << run.tokens[j];
            out << "]}";
        }
        out << "]}\n";
        if (!out) throw std::runtime_error("Cannot write reference result");
        std::cout << "PASS " << runs.size() << " identical reference runs on Vulkan0 / RTX 5080\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
