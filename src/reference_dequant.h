#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

// Q8_0: 32 values per block
// Layout: f16 scale (2 bytes) + 32 x int8 quants (32 bytes) = 34 bytes
static constexpr int Q8_0_BLOCK_SIZE = 32;
static constexpr int Q8_0_BYTES_PER_BLOCK = 34;

// Q4_K_M: 256 values per super-block
// Layout: f16 d (2) + f16 dmin (2) + 12 bytes scales + 128 bytes quants = 144 bytes
static constexpr int Q4_K_BLOCK_SIZE = 256;
static constexpr int Q4_K_BYTES_PER_BLOCK = 144;

static inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            float result;
            uint32_t bits = sign;
            std::memcpy(&result, &bits, 4);
            return result;
        }
        // subnormal
        while (!(mant & 0x400)) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x400;
    } else if (exp == 31) {
        uint32_t bits = sign | 0x7F800000 | (mant << 13);
        float result;
        std::memcpy(&result, &bits, 4);
        return result;
    }

    uint32_t bits = sign | ((exp + 112) << 23) | (mant << 13);
    float result;
    std::memcpy(&result, &bits, 4);
    return result;
}

// Dequantize one Q8_0 block (34 bytes in) -> 32 floats out
inline void cpu_dequant_q8_0(const uint8_t* block, float* out) {
    uint16_t d_bits;
    std::memcpy(&d_bits, block, 2);
    float d = fp16_to_fp32(d_bits);

    const int8_t* quants = reinterpret_cast<const int8_t*>(block + 2);
    for (int i = 0; i < Q8_0_BLOCK_SIZE; i++) {
        out[i] = d * quants[i];
    }
}

// Dequantize one Q4_K super-block (144 bytes in) -> 256 floats out
// Matches llama.cpp dequantize_row_q4_K logic
inline void cpu_dequant_q4_k_m(const uint8_t* block, float* out) {
    uint16_t d_bits, dmin_bits;
    std::memcpy(&d_bits, block, 2);
    std::memcpy(&dmin_bits, block + 2, 2);
    float d = fp16_to_fp32(d_bits);
    float dmin = fp16_to_fp32(dmin_bits);

    const uint8_t* scales = block + 4;   // 12 bytes of packed scales/mins
    const uint8_t* quants = block + 16;  // 128 bytes of 4-bit quants

    // Unpack the 12 bytes into 8 scales and 8 mins (6 bits each)
    uint8_t sc[8], mn[8];

    // First 4 sub-blocks: low 6 bits from bytes 0-3 (scales) and 4-7 (mins)
    for (int i = 0; i < 4; i++) {
        sc[i] = scales[i] & 0x3F;
        mn[i] = scales[i + 4] & 0x3F;
    }
    // Last 4 sub-blocks: low 4 bits from bytes 8-11, high 2 bits from top of bytes 0-7
    for (int i = 0; i < 4; i++) {
        sc[4 + i] = (scales[8 + i] & 0x0F) | ((scales[i] >> 6) << 4);
        mn[4 + i] = (scales[8 + i] >> 4)    | ((scales[i + 4] >> 6) << 4);
    }

    // Dequantize 8 sub-blocks of 32 values each
    for (int j = 0; j < 8; j++) {
        float scale = d * sc[j];
        float min = dmin * mn[j];
        const uint8_t* q = quants + j * 16; // 16 bytes = 32 nibbles

        for (int i = 0; i < 16; i++) {
            out[j * 32 + i]      = scale * (q[i] & 0xF) - min;
            out[j * 32 + i + 16] = scale * (q[i] >> 4)  - min;
        }
    }
}

// Convenience: dequantize multiple blocks
inline std::vector<float> cpu_dequant_q8_0_n(const uint8_t* data, int n_blocks) {
    std::vector<float> out(n_blocks * Q8_0_BLOCK_SIZE);
    for (int i = 0; i < n_blocks; i++) {
        cpu_dequant_q8_0(data + i * Q8_0_BYTES_PER_BLOCK, out.data() + i * Q8_0_BLOCK_SIZE);
    }
    return out;
}

inline std::vector<float> cpu_dequant_q4_k_m_n(const uint8_t* data, int n_blocks) {
    std::vector<float> out(n_blocks * Q4_K_BLOCK_SIZE);
    for (int i = 0; i < n_blocks; i++) {
        cpu_dequant_q4_k_m(data + i * Q4_K_BYTES_PER_BLOCK, out.data() + i * Q4_K_BLOCK_SIZE);
    }
    return out;
}
