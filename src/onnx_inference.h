#pragma once

#include <chrono>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "generate_result.h"
#include "gpu_context.h"
#include "onnx_loader.h"
#include "onnx_runtime.h"
#include "sampling.h"

struct OnnxGenerateParams {
    uint32_t max_tokens = 128;
    float temperature = 0.0f;
    uint32_t top_k = 40;
    float top_p = 0.9f;
    std::string shader_dir = "src/shaders";
};

inline GenerateResult generate_onnx(const std::string& onnx_path,
                                    const std::vector<uint32_t>& input_ids,
                                    const OnnxGenerateParams& params = {}) {
    auto ctx = create_gpu_context();
    auto model = onnx::load_onnx(onnx_path);

    onnx_runtime::OnnxRuntime runtime(ctx, params.shader_dir);

    std::string input_name;
    for (auto& vi : model.graph.inputs) {
        bool is_initializer = false;
        for (auto& init : model.graph.initializers) {
            if (init.name == vi.name) { is_initializer = true; break; }
        }
        if (!is_initializer) {
            input_name = vi.name;
            break;
        }
    }
    if (input_name.empty())
        throw std::runtime_error("generate_onnx: no graph input found");

    std::string output_name;
    if (!model.graph.outputs.empty())
        output_name = model.graph.outputs[0].name;
    if (output_name.empty())
        throw std::runtime_error("generate_onnx: no graph output found");

    std::mt19937 rng(42);
    GenerateResult result{};

    std::vector<uint32_t> token_sequence(input_ids.begin(), input_ids.end());

    auto prefill_start = std::chrono::high_resolution_clock::now();

    std::vector<float> input_floats(token_sequence.begin(), token_sequence.end());
    std::unordered_map<std::string, std::vector<float>> feed;
    feed[input_name] = input_floats;

    auto outputs = runtime.run(model.graph, feed);
    auto& logits = outputs.at(output_name);

    uint32_t seq_len = static_cast<uint32_t>(token_sequence.size());
    if (logits.size() % seq_len != 0)
        throw std::runtime_error("generate_onnx: logits size not divisible by seq_len");
    uint32_t vocab_size = static_cast<uint32_t>(logits.size()) / seq_len;
    const float* last_logits = logits.data() + (seq_len - 1) * vocab_size;

    uint32_t first_token;
    if (params.temperature <= 0.0f) {
        first_token = sample_greedy(last_logits, vocab_size);
    } else {
        first_token = sample_topk(last_logits, vocab_size,
                                  params.top_k, params.temperature, rng);
    }
    result.tokens.push_back(first_token);

    auto prefill_end = std::chrono::high_resolution_clock::now();
    double prefill_secs = std::chrono::duration<double>(prefill_end - prefill_start).count();
    result.prefill_tok_per_sec = static_cast<double>(input_ids.size()) / prefill_secs;

    auto decode_start = std::chrono::high_resolution_clock::now();
    uint32_t decode_count = 0;

    for (uint32_t step = 1; step < params.max_tokens; step++) {
        token_sequence.push_back(result.tokens.back());

        std::vector<float> seq_floats(token_sequence.begin(), token_sequence.end());
        feed[input_name] = seq_floats;

        outputs = runtime.run(model.graph, feed);
        auto& step_logits = outputs.at(output_name);

        uint32_t step_seq = static_cast<uint32_t>(token_sequence.size());
        if (step_logits.size() % step_seq != 0)
            throw std::runtime_error("generate_onnx: logits size not divisible by seq_len");
        uint32_t step_vocab = static_cast<uint32_t>(step_logits.size()) / step_seq;
        const float* step_last = step_logits.data() + (step_seq - 1) * step_vocab;

        uint32_t next_token;
        if (params.temperature <= 0.0f) {
            next_token = sample_greedy(step_last, step_vocab);
        } else {
            next_token = sample_topk(step_last, step_vocab,
                                     params.top_k, params.temperature, rng);
        }

        result.tokens.push_back(next_token);
        decode_count++;

        if (next_token == 0) break;
    }

    auto decode_end = std::chrono::high_resolution_clock::now();
    double decode_secs = std::chrono::duration<double>(decode_end - decode_start).count();
    result.decode_tok_per_sec = decode_count > 0
        ? static_cast<double>(decode_count) / decode_secs
        : 0.0;

    return result;
}
