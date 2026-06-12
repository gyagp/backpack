#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>

#include "reference_dequant.h"

using float16_t = uint16_t;

inline float f16_to_f32(float16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            int e = 0;
            while (!(mant & 0x400)) { mant <<= 1; e++; }
            mant &= 0x3FF;
            f = (sign << 31) | ((127 - 14 - e) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }

    float result;
    std::memcpy(&result, &f, sizeof(f));
    return result;
}

inline float16_t f32_to_f16(float val) {
    uint32_t f;
    std::memcpy(&f, &val, sizeof(f));

    uint32_t sign = (f >> 31) & 0x1;
    int32_t exp = ((f >> 23) & 0xFF) - 127;
    uint32_t mant = f & 0x7FFFFF;

    uint16_t h;
    if (exp > 15) {
        h = (sign << 15) | (0x1F << 10);
    } else if (exp < -14) {
        h = (sign << 15);
    } else {
        h = (sign << 15) | ((exp + 15) << 10) | (mant >> 13);
    }
    return h;
}

// C = A * B, where A is [M x K], B is [K x N], C is [M x N], all in f16
inline void cpu_matmul_f16(const float16_t* A, const float16_t* B, float16_t* C,
                           size_t M, size_t N, size_t K) {
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; k++) {
                sum += f16_to_f32(A[m * K + k]) * f16_to_f32(B[k * N + n]);
            }
            C[m * N + n] = f32_to_f16(sum);
        }
    }
}

// Batched: A is [batch x M x K], B is [batch x K x N], C is [batch x M x N]
inline void cpu_batched_matmul_f16(const float16_t* A, const float16_t* B, float16_t* C,
                                   size_t batch, size_t M, size_t N, size_t K) {
    size_t a_stride = M * K;
    size_t b_stride = K * N;
    size_t c_stride = M * N;

    for (size_t b = 0; b < batch; b++) {
        cpu_matmul_f16(A + b * a_stride, B + b * b_stride, C + b * c_stride, M, N, K);
    }
}

// C = A * B, where A is [M x K] f16, B is [K x N] Q4_K_M quantized, C is [M x N] f16
// K must be a multiple of Q4_K_BLOCK_SIZE (256)
inline void cpu_matmul_q4km_f16(const float16_t* A, const uint8_t* B_q4,
                                float16_t* C, size_t M, size_t N, size_t K) {
    size_t blocks_per_col = K / Q4_K_BLOCK_SIZE;
    std::vector<float> deq(Q4_K_BLOCK_SIZE);

    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            float sum = 0.0f;
            for (size_t blk = 0; blk < blocks_per_col; blk++) {
                size_t b_block_idx = n * blocks_per_col + blk;
                cpu_dequant_q4_k_m(B_q4 + b_block_idx * Q4_K_BYTES_PER_BLOCK, deq.data());
                size_t k_offset = blk * Q4_K_BLOCK_SIZE;
                for (size_t i = 0; i < Q4_K_BLOCK_SIZE; i++) {
                    sum += f16_to_f32(A[m * K + k_offset + i]) * deq[i];
                }
            }
            C[m * N + n] = f32_to_f16(sum);
        }
    }
}

// Batched: A is [batch x M x K] f16, B is [batch x K x N] Q4_K_M, C is [batch x M x N] f16
inline void cpu_batched_matmul_q4km_f16(const float16_t* A, const uint8_t* B_q4,
                                        float16_t* C, size_t batch,
                                        size_t M, size_t N, size_t K) {
    size_t a_stride = M * K;
    size_t blocks_per_col = K / Q4_K_BLOCK_SIZE;
    size_t b_stride = N * blocks_per_col * Q4_K_BYTES_PER_BLOCK;
    size_t c_stride = M * N;

    for (size_t bi = 0; bi < batch; bi++) {
        cpu_matmul_q4km_f16(A + bi * a_stride, B_q4 + bi * b_stride,
                            C + bi * c_stride, M, N, K);
    }
}
