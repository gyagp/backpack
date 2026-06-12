#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../src/reference_matmul.h"

static bool approx(float a, float b, float tol = 1e-2f) {
    return std::fabs(a - b) < tol + std::fabs(b) * 0.02f;
}

static void pack_q4_k_m_block(const float* vals, uint8_t* block) {
    std::memset(block, 0, Q4_K_BYTES_PER_BLOCK);

    float vmin = vals[0], vmax = vals[0];
    for (int i = 1; i < Q4_K_BLOCK_SIZE; i++) {
        if (vals[i] < vmin) vmin = vals[i];
        if (vals[i] > vmax) vmax = vals[i];
    }

    float range = vmax - vmin;
    float d_val = range > 0 ? range / 15.0f : 1.0f;
    float dmin_val = vmin < 0 ? -vmin : 0.0f;

    uint16_t d_f16 = f32_to_f16(d_val);
    uint16_t dmin_f16 = f32_to_f16(dmin_val);
    std::memcpy(block, &d_f16, 2);
    std::memcpy(block + 2, &dmin_f16, 2);

    uint8_t* scales = block + 4;
    for (int i = 0; i < 4; i++) {
        scales[i] = 1;
        scales[i + 4] = 1;
    }
    for (int i = 0; i < 4; i++) {
        scales[8 + i] = 1 | (1 << 4);
    }

    float d_f = f16_to_f32(d_f16);
    float dmin_f = f16_to_f32(dmin_f16);
    uint8_t* quants = block + 16;

    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 16; i++) {
            float v_lo = vals[j * 32 + i];
            float v_hi = vals[j * 32 + i + 16];
            int q_lo = (int)roundf((v_lo + dmin_f) / d_f);
            int q_hi = (int)roundf((v_hi + dmin_f) / d_f);
            if (q_lo < 0) q_lo = 0; if (q_lo > 15) q_lo = 15;
            if (q_hi < 0) q_hi = 0; if (q_hi > 15) q_hi = 15;
            quants[j * 16 + i] = (uint8_t)(q_lo | (q_hi << 4));
        }
    }
}

// --- Batched f16 matmul tests ---

static int test_batched_f16(size_t batch, size_t M, size_t N, size_t K, const char* label) {
    std::vector<float16_t> A(batch * M * K);
    std::vector<float16_t> B(batch * K * N);
    std::vector<float16_t> C(batch * M * N);

    for (size_t i = 0; i < A.size(); i++)
        A[i] = f32_to_f16((float)((i * 3 + 7) % 11) * 0.2f - 1.0f);
    for (size_t i = 0; i < B.size(); i++)
        B[i] = f32_to_f16((float)((i * 5 + 3) % 13) * 0.15f - 0.9f);

    cpu_batched_matmul_f16(A.data(), B.data(), C.data(), batch, M, N, K);

    // Verify each batch against individual matmul
    size_t a_stride = M * K, b_stride = K * N, c_stride = M * N;
    for (size_t bi = 0; bi < batch; bi++) {
        std::vector<float16_t> C_ref(M * N);
        cpu_matmul_f16(A.data() + bi * a_stride, B.data() + bi * b_stride, C_ref.data(), M, N, K);
        for (size_t i = 0; i < M * N; i++) {
            float got = f16_to_f32(C[bi * c_stride + i]);
            float exp = f16_to_f32(C_ref[i]);
            if (!approx(got, exp)) {
                printf("FAIL %s batch=%zu idx=%zu: expected %f got %f\n", label, bi, i, exp, got);
                return 1;
            }
        }
    }
    printf("PASS %s (batch=%zu M=%zu N=%zu K=%zu)\n", label, batch, M, N, K);
    return 0;
}

// --- Batched Q4_K_M matmul tests ---

static int test_batched_q4km(size_t batch, size_t M, size_t N, size_t K, const char* label) {
    size_t blocks_per_col = K / Q4_K_BLOCK_SIZE;

    // Generate A
    std::vector<float16_t> A(batch * M * K);
    for (size_t i = 0; i < A.size(); i++)
        A[i] = f32_to_f16((float)((i * 3 + 5) % 9) * 0.2f - 0.8f);

    // Generate and quantize B per batch
    std::vector<uint8_t> B_q4(batch * N * blocks_per_col * Q4_K_BYTES_PER_BLOCK);
    size_t b_stride = N * blocks_per_col * Q4_K_BYTES_PER_BLOCK;

    for (size_t bi = 0; bi < batch; bi++) {
        for (size_t n = 0; n < N; n++) {
            std::vector<float> col(K);
            for (size_t k = 0; k < K; k++)
                col[k] = (float)(((bi * N + n) * K + k) * 7 + 13) / 100.0f - 1.0f;
            for (size_t blk = 0; blk < blocks_per_col; blk++) {
                pack_q4_k_m_block(col.data() + blk * Q4_K_BLOCK_SIZE,
                    B_q4.data() + bi * b_stride + (n * blocks_per_col + blk) * Q4_K_BYTES_PER_BLOCK);
            }
        }
    }

    // Run batched
    std::vector<float16_t> C(batch * M * N);
    cpu_batched_matmul_q4km_f16(A.data(), B_q4.data(), C.data(), batch, M, N, K);

    // Verify each batch against individual matmul
    size_t a_stride = M * K;
    size_t c_stride = M * N;
    for (size_t bi = 0; bi < batch; bi++) {
        std::vector<float16_t> C_ref(M * N);
        cpu_matmul_q4km_f16(A.data() + bi * a_stride, B_q4.data() + bi * b_stride,
                            C_ref.data(), M, N, K);
        for (size_t i = 0; i < M * N; i++) {
            float got = f16_to_f32(C[bi * c_stride + i]);
            float exp = f16_to_f32(C_ref[i]);
            if (!approx(got, exp, 0.5f)) {
                printf("FAIL %s batch=%zu idx=%zu: expected %f got %f\n", label, bi, i, exp, got);
                return 1;
            }
        }
    }
    printf("PASS %s (batch=%zu M=%zu N=%zu K=%zu)\n", label, batch, M, N, K);
    return 0;
}

int main() {
    int fail = 0;

    // Batched f16 matmul: batch sizes 1, 4, 8
    fail += test_batched_f16(1, 4, 4, 8, "f16_batch1");
    fail += test_batched_f16(4, 4, 4, 8, "f16_batch4");
    fail += test_batched_f16(8, 4, 4, 8, "f16_batch8");
    fail += test_batched_f16(4, 3, 5, 16, "f16_batch4_rect");

    // Batched Q4_K_M matmul
    fail += test_batched_q4km(1, 2, 2, 256, "q4km_batch1");
    fail += test_batched_q4km(4, 2, 2, 256, "q4km_batch4");
    fail += test_batched_q4km(8, 2, 2, 256, "q4km_batch8");
    fail += test_batched_q4km(4, 4, 8, 256, "q4km_batch4_rect");

    if (fail) printf("\n%d test(s) FAILED\n", fail);
    else printf("\nAll tests PASSED\n");
    return fail;
}
