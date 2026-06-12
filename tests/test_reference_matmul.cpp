#include <cstdio>
#include <cmath>
#include <cstdlib>
#include "../src/reference_matmul.h"

static bool approx(float a, float b, float tol = 1e-2f) {
    return std::fabs(a - b) < tol;
}

static int test_f16_roundtrip() {
    float vals[] = {0.0f, 1.0f, -1.0f, 0.5f, 65504.0f, -65504.0f, 1.0f/1024.0f};
    for (float v : vals) {
        float rt = f16_to_f32(f32_to_f16(v));
        if (!approx(rt, v, std::fabs(v) * 0.01f + 1e-5f)) {
            printf("FAIL roundtrip: %f -> %f\n", v, rt);
            return 1;
        }
    }
    printf("PASS f16_roundtrip\n");
    return 0;
}

static int test_subnormal() {
    float small = 5.96e-8f; // smallest f16 subnormal
    float16_t h = f32_to_f16(small);
    float back = f16_to_f32(h);
    // subnormals may lose precision but should be in the right ballpark or zero
    if (back != 0.0f && !approx(back, small, small * 2.0f)) {
        printf("FAIL subnormal: %e -> %e\n", small, back);
        return 1;
    }
    printf("PASS subnormal\n");
    return 0;
}

static int test_matmul_identity() {
    // 2x2 identity * [1,2;3,4] = [1,2;3,4]
    float A_f[] = {1,0, 0,1};
    float B_f[] = {1,2, 3,4};
    float16_t A[4], B[4], C[4];
    for (int i = 0; i < 4; i++) { A[i] = f32_to_f16(A_f[i]); B[i] = f32_to_f16(B_f[i]); }

    cpu_matmul_f16(A, B, C, 2, 2, 2);

    for (int i = 0; i < 4; i++) {
        float got = f16_to_f32(C[i]);
        if (!approx(got, B_f[i])) {
            printf("FAIL identity[%d]: expected %f got %f\n", i, B_f[i], got);
            return 1;
        }
    }
    printf("PASS matmul_identity\n");
    return 0;
}

static int test_matmul_3x2_2x4() {
    // A[3x2] * B[2x4] = C[3x4]
    float Af[] = {1,2, 3,4, 5,6};
    float Bf[] = {1,2,3,4, 5,6,7,8};
    float expected[] = {11,14,17,20, 23,30,37,44, 35,46,57,68};

    float16_t A[6], B[8], C[12];
    for (int i = 0; i < 6; i++) A[i] = f32_to_f16(Af[i]);
    for (int i = 0; i < 8; i++) B[i] = f32_to_f16(Bf[i]);

    cpu_matmul_f16(A, B, C, 3, 4, 2);

    for (int i = 0; i < 12; i++) {
        float got = f16_to_f32(C[i]);
        if (!approx(got, expected[i])) {
            printf("FAIL 3x2*2x4[%d]: expected %f got %f\n", i, expected[i], got);
            return 1;
        }
    }
    printf("PASS matmul_3x2_2x4\n");
    return 0;
}

static int test_batched() {
    float Af[] = {1,0, 0,1,  2,0, 0,2};
    float Bf[] = {5,6, 7,8,  5,6, 7,8};
    float exp[] = {5,6,7,8, 10,12,14,16};

    float16_t A[8], B[8], C[8];
    for (int i = 0; i < 8; i++) { A[i] = f32_to_f16(Af[i]); B[i] = f32_to_f16(Bf[i]); }

    cpu_batched_matmul_f16(A, B, C, 2, 2, 2, 2);

    for (int i = 0; i < 8; i++) {
        float got = f16_to_f32(C[i]);
        if (!approx(got, exp[i])) {
            printf("FAIL batched[%d]: expected %f got %f\n", i, exp[i], got);
            return 1;
        }
    }
    printf("PASS batched\n");
    return 0;
}

static void pack_q4_k_m_block(const float* vals, uint8_t* block) {
    // Find range
    float vmin = vals[0], vmax = vals[0];
    for (int i = 1; i < Q4_K_BLOCK_SIZE; i++) {
        if (vals[i] < vmin) vmin = vals[i];
        if (vals[i] > vmax) vmax = vals[i];
    }

    // Use uniform scale across all sub-blocks for simplicity
    float range = vmax - vmin;
    float d_val = range / 15.0f;
    float dmin_val = -vmin;
    if (d_val == 0.0f) d_val = 1.0f;

    // Write super-block d and dmin as f16
    uint16_t d_f16 = f32_to_f16(d_val);
    uint16_t dmin_f16 = f32_to_f16(dmin_val / 1.0f); // dmin such that actual_min = dmin * mn
    std::memcpy(block, &d_f16, 2);

    // We'll use sc=1, mn=1 for all sub-blocks, so actual scale = d*1 = d, actual min = dmin*1
    // Thus val ≈ d * q - dmin, so q = round((val + dmin) / d)
    float dmin_actual = dmin_val; // we set mn=1, so min = dmin * 1
    uint16_t dmin_h = f32_to_f16(dmin_actual);
    std::memcpy(block + 2, &dmin_h, 2);

    // Pack scales: sc[i]=1, mn[i]=1 for all 8 sub-blocks
    uint8_t* scales = block + 4;
    std::memset(scales, 0, 12);
    for (int i = 0; i < 4; i++) {
        scales[i] = 1;       // sc[0..3] in low 6 bits
        scales[i + 4] = 1;   // mn[0..3] in low 6 bits
    }
    for (int i = 0; i < 4; i++) {
        scales[8 + i] = 1 | (1 << 4); // sc[4..7] low4 | mn[4..7] low4
        // high 2 bits of sc[4..7] come from scales[0..3] bits 6-7 = 0
        // high 2 bits of mn[4..7] come from scales[4..7] bits 6-7 = 0
    }

    // Quantize and pack 4-bit values
    uint8_t* quants = block + 16;
    float d_f = f16_to_f32(d_f16);
    float dmin_f = f16_to_f32(dmin_h);

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

static int test_q4km_matmul() {
    // A[2x256] * B_q4[256x2] = C[2x2]
    // K=256 (one Q4_K block per column)
    const size_t M = 2, N = 2, K = 256;

    // Create known weight values (B in float, then quantize)
    std::vector<float> B_float(K * N);
    for (size_t i = 0; i < K * N; i++) {
        B_float[i] = (float)(i % 7) * 0.5f; // values 0..3
    }

    // Quantize B column-major: column n -> block n
    std::vector<uint8_t> B_q4(N * Q4_K_BYTES_PER_BLOCK);
    for (size_t n = 0; n < N; n++) {
        // Extract column n
        std::vector<float> col(K);
        for (size_t k = 0; k < K; k++) col[k] = B_float[k * N + n];
        pack_q4_k_m_block(col.data(), B_q4.data() + n * Q4_K_BYTES_PER_BLOCK);
    }

    // Create A
    std::vector<float16_t> A(M * K);
    std::vector<float> Af(M * K);
    for (size_t i = 0; i < M * K; i++) {
        Af[i] = (float)((i % 5) + 1) * 0.1f;
        A[i] = f32_to_f16(Af[i]);
    }

    // Compute expected: dequantize B, then multiply
    std::vector<float> B_deq(K * N);
    for (size_t n = 0; n < N; n++) {
        std::vector<float> col_deq(K);
        cpu_dequant_q4_k_m(B_q4.data() + n * Q4_K_BYTES_PER_BLOCK, col_deq.data());
        for (size_t k = 0; k < K; k++) B_deq[k * N + n] = col_deq[k];
    }

    std::vector<float> expected(M * N, 0.0f);
    for (size_t m = 0; m < M; m++)
        for (size_t n = 0; n < N; n++)
            for (size_t k = 0; k < K; k++)
                expected[m * N + n] += f16_to_f32(A[m * K + k]) * B_deq[k * N + n];

    // Run the function under test
    std::vector<float16_t> C(M * N);
    cpu_matmul_q4km_f16(A.data(), B_q4.data(), C.data(), M, N, K);

    // Compare
    for (size_t i = 0; i < M * N; i++) {
        float got = f16_to_f32(C[i]);
        float exp = expected[i];
        if (!approx(got, exp, std::fabs(exp) * 0.05f + 0.5f)) {
            printf("FAIL q4km_matmul[%zu]: expected %f got %f\n", i, exp, got);
            return 1;
        }
    }
    printf("PASS q4km_matmul\n");
    return 0;
}

int main() {
    int fail = 0;
    fail += test_f16_roundtrip();
    fail += test_subnormal();
    fail += test_matmul_identity();
    fail += test_matmul_3x2_2x4();
    fail += test_batched();
    fail += test_q4km_matmul();
    if (fail) printf("\n%d test(s) FAILED\n", fail);
    else printf("\nAll tests PASSED\n");
    return fail;
}
