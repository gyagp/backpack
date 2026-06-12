#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include "../src/reference_dequant.h"

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT_NEAR(a, b, eps, msg) do { \
    tests_run++; \
    if (std::fabs((a) - (b)) > (eps)) { \
        printf("FAIL: %s: expected %f, got %f\n", msg, (double)(b), (double)(a)); \
    } else { tests_passed++; } \
} while(0)

#define ASSERT_EQ(a, b, msg) do { \
    tests_run++; \
    if ((a) != (b)) { \
        printf("FAIL: %s: expected %d, got %d\n", msg, (int)(b), (int)(a)); \
    } else { tests_passed++; } \
} while(0)

static uint16_t f32_to_f16(float v) {
    uint16_t result;
    // Use the bit manipulation approach
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;
    if (exp <= 0) {
        result = (uint16_t)sign;
    } else if (exp >= 31) {
        result = (uint16_t)(sign | 0x7C00);
    } else {
        result = (uint16_t)(sign | (exp << 10) | mant);
    }
    return result;
}

static void build_q8_0_block(uint8_t* block, float d, const int8_t* quants) {
    uint16_t d_bits = f32_to_f16(d);
    std::memcpy(block, &d_bits, 2);
    std::memcpy(block + 2, quants, 32);
}

static void build_q4_k_block(uint8_t* block, float d, float dmin,
                              const uint8_t scales[8], const uint8_t mins[8],
                              const uint8_t qs[128]) {
    std::memset(block, 0, Q4_K_BYTES_PER_BLOCK);
    uint16_t d_bits = f32_to_f16(d);
    uint16_t dmin_bits = f32_to_f16(dmin);
    std::memcpy(block, &d_bits, 2);
    std::memcpy(block + 2, &dmin_bits, 2);

    // Pack scales/mins into 12 bytes (same layout as make_block in python test)
    uint8_t* s = block + 4;
    for (int j = 0; j < 4; j++) {
        s[j]     = (scales[j] & 63) | ((scales[j + 4] >> 4) << 6);
        s[j + 4] = (mins[j] & 63)   | ((mins[j + 4] >> 4) << 6);
    }
    for (int j = 0; j < 4; j++) {
        s[8 + j] = (scales[j + 4] & 0xF) | ((mins[j + 4] & 0xF) << 4);
    }

    std::memcpy(block + 16, qs, 128);
}

void test_q8_0_basic() {
    uint8_t block[Q8_0_BYTES_PER_BLOCK];
    int8_t quants[32];
    for (int i = 0; i < 32; i++) quants[i] = (int8_t)i;
    build_q8_0_block(block, 0.5f, quants);

    float out[32];
    cpu_dequant_q8_0(block, out);

    for (int i = 0; i < 32; i++) {
        ASSERT_NEAR(out[i], 0.5f * i, 0.01f, "q8_0 basic");
    }
}

void test_q8_0_negative() {
    uint8_t block[Q8_0_BYTES_PER_BLOCK];
    int8_t quants[32];
    for (int i = 0; i < 32; i++) quants[i] = (int8_t)(-i);
    build_q8_0_block(block, 2.0f, quants);

    float out[32];
    cpu_dequant_q8_0(block, out);

    for (int i = 0; i < 32; i++) {
        ASSERT_NEAR(out[i], -2.0f * i, 0.01f, "q8_0 negative");
    }
}

void test_q8_0_zero_scale() {
    uint8_t block[Q8_0_BYTES_PER_BLOCK];
    int8_t quants[32];
    for (int i = 0; i < 32; i++) quants[i] = 127;
    build_q8_0_block(block, 0.0f, quants);

    float out[32];
    cpu_dequant_q8_0(block, out);

    for (int i = 0; i < 32; i++) {
        ASSERT_NEAR(out[i], 0.0f, 0.001f, "q8_0 zero scale");
    }
}

void test_q8_0_multi_block() {
    uint8_t data[Q8_0_BYTES_PER_BLOCK * 2];
    int8_t q1[32], q2[32];
    for (int i = 0; i < 32; i++) { q1[i] = 1; q2[i] = 2; }
    build_q8_0_block(data, 1.0f, q1);
    build_q8_0_block(data + Q8_0_BYTES_PER_BLOCK, 3.0f, q2);

    auto out = cpu_dequant_q8_0_n(data, 2);
    ASSERT_EQ((int)out.size(), 64, "q8_0 multi size");
    ASSERT_NEAR(out[0], 1.0f, 0.01f, "q8_0 multi block0");
    ASSERT_NEAR(out[32], 6.0f, 0.01f, "q8_0 multi block1");
}

void test_q4_k_all_zeros() {
    uint8_t scales[8] = {1,1,1,1,1,1,1,1};
    uint8_t mins[8] = {0,0,0,0,0,0,0,0};
    uint8_t qs[128];
    std::memset(qs, 0, 128);

    uint8_t block[Q4_K_BYTES_PER_BLOCK];
    build_q4_k_block(block, 1.0f, 0.0f, scales, mins, qs);

    float out[256];
    cpu_dequant_q4_k_m(block, out);

    for (int i = 0; i < 256; i++) {
        ASSERT_NEAR(out[i], 0.0f, 0.001f, "q4_k zeros");
    }
}

void test_q4_k_all_max() {
    uint8_t scales[8] = {1,1,1,1,1,1,1,1};
    uint8_t mins[8] = {0,0,0,0,0,0,0,0};
    uint8_t qs[128];
    std::memset(qs, 0xFF, 128);

    uint8_t block[Q4_K_BYTES_PER_BLOCK];
    build_q4_k_block(block, 1.0f, 0.0f, scales, mins, qs);

    float out[256];
    cpu_dequant_q4_k_m(block, out);

    for (int i = 0; i < 256; i++) {
        ASSERT_NEAR(out[i], 15.0f, 0.01f, "q4_k max");
    }
}

void test_q4_k_with_min() {
    uint8_t scales[8] = {2,2,2,2,2,2,2,2};
    uint8_t mins[8] = {3,3,3,3,3,3,3,3};
    uint8_t qs[128];
    std::memset(qs, 0, 128);

    uint8_t block[Q4_K_BYTES_PER_BLOCK];
    build_q4_k_block(block, 1.0f, 1.0f, scales, mins, qs);

    float out[256];
    cpu_dequant_q4_k_m(block, out);

    // val = d * sc * 0 - dmin * mn = 0 - 3 = -3
    for (int i = 0; i < 256; i++) {
        ASSERT_NEAR(out[i], -3.0f, 0.01f, "q4_k with min");
    }
}

void test_q4_k_mixed_nibbles() {
    uint8_t scales[8] = {4,4,4,4,4,4,4,4};
    uint8_t mins[8] = {2,2,2,2,2,2,2,2};
    uint8_t qs[128];
    std::memset(qs, 0, 128);
    qs[0] = 0xA5; // low=5, high=A=10

    uint8_t block[Q4_K_BYTES_PER_BLOCK];
    build_q4_k_block(block, 0.5f, 0.25f, scales, mins, qs);

    float out[256];
    cpu_dequant_q4_k_m(block, out);

    // out[0] = d*sc*5 - dmin*mn = 0.5*4*5 - 0.25*2 = 10 - 0.5 = 9.5
    ASSERT_NEAR(out[0], 9.5f, 0.01f, "q4_k mixed low nibble");
    // out[16] = d*sc*10 - dmin*mn = 0.5*4*10 - 0.25*2 = 20 - 0.5 = 19.5
    ASSERT_NEAR(out[16], 19.5f, 0.01f, "q4_k mixed high nibble");
}

void test_q4_k_output_length() {
    uint8_t block[Q4_K_BYTES_PER_BLOCK];
    uint8_t scales[8] = {1,1,1,1,1,1,1,1};
    uint8_t mins[8] = {0,0,0,0,0,0,0,0};
    uint8_t qs[128];
    std::memset(qs, 0x55, 128);
    build_q4_k_block(block, 1.0f, 0.5f, scales, mins, qs);

    auto out = cpu_dequant_q4_k_m_n(block, 1);
    ASSERT_EQ((int)out.size(), 256, "q4_k output length");
}

void test_q4_k_d_scaling() {
    uint8_t scales[8] = {1,1,1,1,1,1,1,1};
    uint8_t mins[8] = {0,0,0,0,0,0,0,0};
    uint8_t qs[128];
    std::memset(qs, 0x11, 128);

    uint8_t block1[Q4_K_BYTES_PER_BLOCK], block2[Q4_K_BYTES_PER_BLOCK];
    build_q4_k_block(block1, 1.0f, 0.0f, scales, mins, qs);
    build_q4_k_block(block2, 2.0f, 0.0f, scales, mins, qs);

    float out1[256], out2[256];
    cpu_dequant_q4_k_m(block1, out1);
    cpu_dequant_q4_k_m(block2, out2);

    for (int i = 0; i < 32; i++) {
        if (out1[i] != 0.0f) {
            ASSERT_NEAR(out2[i] / out1[i], 2.0f, 0.01f, "q4_k d scaling");
        }
    }
}

void test_fp16_roundtrip() {
    float vals[] = {0.0f, 1.0f, -1.0f, 0.5f, 65504.0f, 0.00006103515625f};
    for (float v : vals) {
        uint16_t bits = f32_to_f16(v);
        float back = fp16_to_fp32(bits);
        ASSERT_NEAR(back, v, 0.001f * std::fabs(v) + 1e-7f, "fp16 roundtrip");
    }
}

void test_q4_k_high_subblock_scales() {
    // Test that sub-blocks 4-7 with larger scale values (using upper bits) work
    uint8_t scales[8] = {0,0,0,0, 33,45,50,60}; // >15 needs upper bits
    uint8_t mins[8] = {0,0,0,0, 20,25,30,35};
    uint8_t qs[128];
    std::memset(qs, 0x11, 128); // nibble=1 for both

    uint8_t block[Q4_K_BYTES_PER_BLOCK];
    build_q4_k_block(block, 1.0f, 1.0f, scales, mins, qs);

    float out[256];
    cpu_dequant_q4_k_m(block, out);

    // Sub-block 4, first value: d*sc[4]*1 - dmin*mn[4] = 33 - 20 = 13
    ASSERT_NEAR(out[4 * 32], 13.0f, 0.01f, "q4_k high subblock 4");
    // Sub-block 5: 45 - 25 = 20
    ASSERT_NEAR(out[5 * 32], 20.0f, 0.01f, "q4_k high subblock 5");
}

int main() {
    test_fp16_roundtrip();
    test_q8_0_basic();
    test_q8_0_negative();
    test_q8_0_zero_scale();
    test_q8_0_multi_block();
    test_q4_k_all_zeros();
    test_q4_k_all_max();
    test_q4_k_with_min();
    test_q4_k_mixed_nibbles();
    test_q4_k_output_length();
    test_q4_k_d_scaling();
    test_q4_k_high_subblock_scales();

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    if (tests_passed != tests_run) {
        printf("SOME TESTS FAILED\n");
        return 1;
    }
    printf("ALL TESTS PASSED\n");
    return 0;
}
