#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../src/reference_matmul.h"

static bool approx(float a, float b, float tol = 0.5f) {
    return std::fabs(a - b) < tol + std::fabs(b) * 0.05f;
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

    // sc=1, mn=1 for all 8 sub-blocks
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

static int run_q4km_matmul_test(size_t M, size_t N, size_t K, const char* label) {
    size_t blocks_per_col = K / Q4_K_BLOCK_SIZE;

    // Generate deterministic test data
    std::vector<float> B_float(K * N);
    for (size_t i = 0; i < K * N; i++)
        B_float[i] = (float)((i * 7 + 13) % 11) * 0.3f - 1.5f;

    // Quantize B column-major
    std::vector<uint8_t> B_q4(N * blocks_per_col * Q4_K_BYTES_PER_BLOCK);
    for (size_t n = 0; n < N; n++) {
        std::vector<float> col(K);
        for (size_t k = 0; k < K; k++) col[k] = B_float[k * N + n];
        for (size_t blk = 0; blk < blocks_per_col; blk++) {
            pack_q4_k_m_block(col.data() + blk * Q4_K_BLOCK_SIZE,
                              B_q4.data() + (n * blocks_per_col + blk) * Q4_K_BYTES_PER_BLOCK);
        }
    }

    // Create A
    std::vector<float16_t> A(M * K);
    for (size_t i = 0; i < M * K; i++)
        A[i] = f32_to_f16((float)((i * 3 + 5) % 9) * 0.2f - 0.8f);

    // Compute expected: dequantize B, then multiply in f32
    std::vector<float> B_deq(K * N);
    for (size_t n = 0; n < N; n++) {
        for (size_t blk = 0; blk < blocks_per_col; blk++) {
            std::vector<float> deq(Q4_K_BLOCK_SIZE);
            cpu_dequant_q4_k_m(B_q4.data() + (n * blocks_per_col + blk) * Q4_K_BYTES_PER_BLOCK, deq.data());
            for (size_t i = 0; i < Q4_K_BLOCK_SIZE; i++)
                B_deq[(blk * Q4_K_BLOCK_SIZE + i) * N + n] = deq[i];
        }
    }

    std::vector<float> expected(M * N, 0.0f);
    for (size_t m = 0; m < M; m++)
        for (size_t n = 0; n < N; n++)
            for (size_t k = 0; k < K; k++)
                expected[m * N + n] += f16_to_f32(A[m * K + k]) * B_deq[k * N + n];

    // Run function under test
    std::vector<float16_t> C(M * N);
    cpu_matmul_q4km_f16(A.data(), B_q4.data(), C.data(), M, N, K);

    // Compare
    for (size_t i = 0; i < M * N; i++) {
        float got = f16_to_f32(C[i]);
        float exp = expected[i];
        if (!approx(got, exp)) {
            printf("FAIL %s[%zu]: expected %f got %f\n", label, i, exp, got);
            return 1;
        }
    }
    printf("PASS %s (M=%zu N=%zu K=%zu)\n", label, M, N, K);
    return 0;
}

int main() {
    int fail = 0;
    fail += run_q4km_matmul_test(2, 2, 256, "q4km_small_square");
    fail += run_q4km_matmul_test(1, 4, 256, "q4km_single_row");
    fail += run_q4km_matmul_test(4, 1, 256, "q4km_single_col");
    fail += run_q4km_matmul_test(8, 16, 256, "q4km_rectangular");
    fail += run_q4km_matmul_test(4, 8, 512, "q4km_multi_block");
    fail += run_q4km_matmul_test(3, 5, 768, "q4km_odd_dims_3block");

    if (fail) printf("\n%d test(s) FAILED\n", fail);
    else printf("\nAll tests PASSED\n");
    return fail;
}
