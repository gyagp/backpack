#include "../src/gpu_context.h"
#include "../src/buffer_utils.h"
#include "../src/reference_matmul.h"
#include "../src/transformer_layer.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    std::printf("Running transformer layer GPU test...\n");
    auto ctx = create_gpu_context();
    PipelineCache pipeline_cache(ctx.device);
    auto pipelines = load_transformer_pipelines(pipeline_cache);

    uint32_t D = 64;
    uint32_t n_heads = 4;
    uint32_t n_kv_heads = 2;
    uint32_t head_dim = D / n_heads;  // 16
    uint32_t intermediate_dim = 128;
    uint32_t max_seq_len = 32;

    TransformerLayerConfig cfg{};
    cfg.hidden_dim = D;
    cfg.intermediate_dim = intermediate_dim;
    cfg.n_heads = n_heads;
    cfg.n_kv_heads = n_kv_heads;
    cfg.head_dim = head_dim;
    cfg.max_seq_len = max_seq_len;
    cfg.rope_theta = 10000.0f;
    cfg.norm_epsilon = 1e-5f;

    // Generate deterministic pseudo-random weights
    auto make_f32 = [](size_t n, uint32_t seed) {
        std::vector<float> v(n);
        for (size_t i = 0; i < n; i++)
            v[i] = ((float)((i * seed + 5) % 17) - 8.0f) * 0.05f;
        return v;
    };
    auto make_f16 = [](size_t n, uint32_t seed) {
        std::vector<float16_t> v(n);
        for (size_t i = 0; i < n; i++)
            v[i] = f32_to_f16(((float)((i * seed + 5) % 17) - 8.0f) * 0.05f);
        return v;
    };

    auto attn_norm_data = make_f32(D, 3);
    auto ffn_norm_data = make_f32(D, 7);

    uint32_t q_dim = n_heads * head_dim;
    uint32_t kv_dim = n_kv_heads * head_dim;
    auto wq_data = make_f16(D * q_dim, 11);
    auto wk_data = make_f16(D * kv_dim, 13);
    auto wv_data = make_f16(D * kv_dim, 17);
    auto wo_data = make_f16(q_dim * D, 19);
    auto wg_data = make_f16(D * intermediate_dim, 23);
    auto wu_data = make_f16(D * intermediate_dim, 29);
    auto wd_data = make_f16(intermediate_dim * D, 31);

    TransformerLayer layer{};
    layer.attn_norm = create_storage_buffer(ctx.device, D * sizeof(float), attn_norm_data.data());
    layer.ffn_norm = create_storage_buffer(ctx.device, D * sizeof(float), ffn_norm_data.data());

    auto upload_f16 = [&](const std::vector<float16_t>& data) {
        size_t u32_count = (data.size() + 1) / 2;
        return create_storage_buffer(ctx.device, u32_count * sizeof(uint32_t), data.data());
    };
    layer.w_q = upload_f16(wq_data);
    layer.w_k = upload_f16(wk_data);
    layer.w_v = upload_f16(wv_data);
    layer.w_o = upload_f16(wo_data);
    layer.w_gate = upload_f16(wg_data);
    layer.w_up = upload_f16(wu_data);
    layer.w_down = upload_f16(wd_data);

    // Input hidden state
    auto hidden_data = make_f32(D, 37);
    auto hidden_buf = create_storage_buffer(ctx.device, D * sizeof(float), hidden_data.data());

    // KV caches
    size_t cache_size = n_kv_heads * max_seq_len * (head_dim / 2) * sizeof(uint32_t);
    auto k_cache = create_storage_buffer(ctx.device, cache_size, nullptr);
    auto v_cache = create_storage_buffer(ctx.device, cache_size, nullptr);

    // Run one forward pass at seq_pos=0
    auto output = transformer_layer_forward(
        ctx.device, ctx.queue, pipelines, pipeline_cache, layer, cfg,
        hidden_buf, k_cache, v_cache, 0);

    // Read back and verify
    auto result = read_buffer_as<float>(ctx.device, ctx.queue, output, D);

    bool has_nan = false;
    bool has_inf = false;
    for (uint32_t i = 0; i < D; i++) {
        if (std::isnan(result[i])) has_nan = true;
        if (std::isinf(result[i])) has_inf = true;
    }

    int fail = 0;
    if (has_nan) {
        std::printf("FAIL: output contains NaN\n");
        fail++;
    }
    if (has_inf) {
        std::printf("FAIL: output contains Inf\n");
        fail++;
    }

    // Verify output shape (implicitly D elements read successfully)
    std::printf("Output shape: [%u] (hidden_dim)\n", D);
    std::printf("Sample values: [0]=%f [1]=%f [%u]=%f\n",
                result[0], result[1], D - 1, result[D - 1]);

    if (fail) {
        std::printf("\nFAILED\n");
        return 1;
    }
    std::printf("\nPASS: transformer layer forward produced valid output (no NaN/Inf)\n");
    return 0;
}
