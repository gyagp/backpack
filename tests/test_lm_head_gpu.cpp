#include "../src/gpu_context.h"
#include "../src/buffer_utils.h"
#include "../src/lm_head.h"
#include "../src/reference_matmul.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void cpu_rmsnorm(const float* input, float* output, uint32_t row_length, float epsilon) {
    float sum_sq = 0.0f;
    for (uint32_t i = 0; i < row_length; i++)
        sum_sq += input[i] * input[i];
    float inv_rms = 1.0f / std::sqrt(sum_sq / (float)row_length + epsilon);
    for (uint32_t i = 0; i < row_length; i++)
        output[i] = input[i] * inv_rms;
}

static void cpu_lm_head(const float* hidden, const float* norm_weight,
                         const float16_t* output_weight,
                         float* logits,
                         uint32_t hidden_dim, uint32_t vocab_size, float epsilon) {
    // 1. RMSNorm
    std::vector<float> normed(hidden_dim);
    cpu_rmsnorm(hidden, normed.data(), hidden_dim, epsilon);

    // 2. Multiply by norm weight
    for (uint32_t i = 0; i < hidden_dim; i++)
        normed[i] *= norm_weight[i];

    // 3. Matmul: [1 x hidden_dim] @ [hidden_dim x vocab_size] -> [1 x vocab_size]
    // Convert normed to f16 for reference matmul
    std::vector<float16_t> normed_f16(hidden_dim);
    for (uint32_t i = 0; i < hidden_dim; i++)
        normed_f16[i] = f32_to_f16(normed[i]);

    std::vector<float16_t> logits_f16(vocab_size);
    cpu_matmul_f16(normed_f16.data(), output_weight, logits_f16.data(), 1, vocab_size, hidden_dim);

    for (uint32_t i = 0; i < vocab_size; i++)
        logits[i] = f16_to_f32(logits_f16[i]);
}

int main() {
    std::printf("Running lm_head GPU test...\n");
    auto ctx = create_gpu_context();

    uint32_t hidden_dim = 64;
    uint32_t vocab_size = 32;
    float epsilon = 1e-5f;

    // Generate deterministic test data
    std::vector<float> hidden(hidden_dim);
    for (uint32_t i = 0; i < hidden_dim; i++)
        hidden[i] = ((float)((i * 7 + 3) % 19) - 9.0f) * 0.1f;

    std::vector<float> norm_weight(hidden_dim);
    for (uint32_t i = 0; i < hidden_dim; i++)
        norm_weight[i] = 0.5f + ((float)(i % 5)) * 0.1f;

    std::vector<float16_t> output_weight(hidden_dim * vocab_size);
    for (size_t i = 0; i < output_weight.size(); i++)
        output_weight[i] = f32_to_f16(((float)((i * 11 + 2) % 23) - 11.0f) * 0.02f);

    // CPU reference
    std::vector<float> ref_logits(vocab_size);
    cpu_lm_head(hidden.data(), norm_weight.data(), output_weight.data(),
                ref_logits.data(), hidden_dim, vocab_size, epsilon);

    // GPU: upload buffers
    auto buf_hidden = create_storage_buffer(ctx.device, hidden_dim * sizeof(float), hidden.data());
    auto buf_norm = create_storage_buffer(ctx.device, hidden_dim * sizeof(float), norm_weight.data());

    // Pack output_weight f16 -> u32 array
    uint32_t weight_elems = hidden_dim * vocab_size;
    uint32_t weight_u32 = (weight_elems + 1) / 2;
    std::vector<uint32_t> weight_packed(weight_u32);
    std::memcpy(weight_packed.data(), output_weight.data(), weight_elems * sizeof(float16_t));
    auto buf_weight = create_storage_buffer(ctx.device, weight_u32 * sizeof(uint32_t), weight_packed.data());

    // Run GPU lm_head
    PipelineCache pipeline_cache(ctx.device);
    auto buf_logits = lm_head_forward(ctx.device, ctx.queue, pipeline_cache,
                                      buf_hidden, buf_norm, buf_weight,
                                      1, hidden_dim, vocab_size, epsilon);

    // Readback
    auto gpu_logits = read_buffer_as<float>(ctx.device, ctx.queue, buf_logits, vocab_size);

    // Verify shape (implicitly correct if we read vocab_size floats) and values
    int fail = 0;
    float max_err = 0.0f;
    for (uint32_t i = 0; i < vocab_size; i++) {
        float err = std::fabs(gpu_logits[i] - ref_logits[i]);
        if (err > max_err) max_err = err;
        if (err > 1e-2f) {
            if (fail < 5)
                std::printf("  MISMATCH logits[%u]: got=%f expected=%f err=%f\n",
                            i, gpu_logits[i], ref_logits[i], err);
            fail++;
        }
    }

    if (fail) {
        std::printf("FAIL lm_head: %d/%u mismatches (max_err=%.6f)\n", fail, vocab_size, max_err);
        return 1;
    }

    std::printf("PASS lm_head (hidden_dim=%u, vocab_size=%u, max_err=%.6f)\n",
                hidden_dim, vocab_size, max_err);
    return 0;
}
