#pragma once
/**
 * safetensors_loader.h -- Safetensors file parser.
 *
 * Parses the safetensors binary format:
 *   [8 bytes: header_size (u64 LE)]
 *   [header_size bytes: JSON header]
 *   [remainder: raw tensor data]
 *
 * The JSON header maps tensor names to metadata:
 *   { "tensor_name": { "dtype": "F16", "shape": [1024, 768], "data_offsets": [0, 1572864] }, ... }
 *
 * Supports dtypes: F32, F16, BF16, I32, I64, U8, I8, BOOL.
 * BF16 → F32 conversion: uint32_t bits = (uint32_t)bf16 << 16.
 */

#include "json_parser.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

enum class SafetensorsDtype { F32, F16, BF16, I32, I64, U8, I8, BOOL, Unknown };

struct SafetensorInfo {
    std::string name;
    SafetensorsDtype dtype = SafetensorsDtype::Unknown;
    std::vector<int64_t> shape;
    uint64_t dataOffset = 0;   // byte offset from start of data section
    uint64_t dataSize = 0;     // size in bytes
};

struct SafetensorsFile {
    uint64_t headerSize = 0;
    uint64_t dataOffset = 0;   // byte offset of raw data in file
    std::unordered_map<std::string, SafetensorInfo> tensors;

    /// Parse a safetensors file. Returns true on success.
    bool open(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;

        // Read 8-byte header size
        uint64_t hdrSize = 0;
        f.read(reinterpret_cast<char*>(&hdrSize), 8);
        if (!f || hdrSize == 0 || hdrSize > 100 * 1024 * 1024) return false;
        headerSize = hdrSize;
        dataOffset = 8 + hdrSize;

        // Read JSON header
        std::string hdrJson(hdrSize, '\0');
        f.read(&hdrJson[0], (std::streamsize)hdrSize);
        if (!f) return false;

        // Parse JSON
        JsonValue root;
        try {
            root = json_parse(hdrJson);
        } catch (...) {
            return false;
        }
        if (!root.is_object()) return false;

        for (auto& [key, val] : root.as_object()) {
            // Skip "__metadata__" key
            if (key == "__metadata__") continue;
            if (!val.is_object()) continue;

            SafetensorInfo info;
            info.name = key;

            // Parse dtype
            if (val.has("dtype")) {
                auto& dt = val["dtype"].as_string();
                if (dt == "F32") info.dtype = SafetensorsDtype::F32;
                else if (dt == "F16") info.dtype = SafetensorsDtype::F16;
                else if (dt == "BF16") info.dtype = SafetensorsDtype::BF16;
                else if (dt == "I32") info.dtype = SafetensorsDtype::I32;
                else if (dt == "I64") info.dtype = SafetensorsDtype::I64;
                else if (dt == "U8") info.dtype = SafetensorsDtype::U8;
                else if (dt == "I8") info.dtype = SafetensorsDtype::I8;
                else if (dt == "BOOL") info.dtype = SafetensorsDtype::BOOL;
            }

            // Parse shape
            if (val.has("shape")) {
                for (auto& dim : val["shape"].as_array())
                    info.shape.push_back((int64_t)dim.as_number());
            }

            // Parse data_offsets [begin, end]
            if (val.has("data_offsets")) {
                auto& offsets = val["data_offsets"].as_array();
                if (offsets.size() >= 2) {
                    info.dataOffset = (uint64_t)offsets[0].as_number();
                    info.dataSize = (uint64_t)offsets[1].as_number() - info.dataOffset;
                }
            }

            tensors[key] = info;
        }

        return true;
    }

    /// Read raw tensor bytes from file.
    std::vector<uint8_t> readTensor(const std::string& path,
                                     const SafetensorInfo& info) const {
        std::vector<uint8_t> data(info.dataSize);
        std::ifstream f(path, std::ios::binary);
        if (!f) return {};
        f.seekg((std::streamoff)(dataOffset + info.dataOffset));
        f.read(reinterpret_cast<char*>(data.data()), (std::streamsize)info.dataSize);
        if (!f) return {};
        return data;
    }

    /// Read tensor as f32 (with BF16/F16 conversion if needed).
    std::vector<float> readTensorFloat32(const std::string& path,
                                          const SafetensorInfo& info) const {
        auto raw = readTensor(path, info);
        if (raw.empty()) return {};

        int64_t nel = 1;
        for (auto d : info.shape) nel *= d;

        std::vector<float> out(nel);

        switch (info.dtype) {
            case SafetensorsDtype::F32:
                memcpy(out.data(), raw.data(), nel * sizeof(float));
                break;

            case SafetensorsDtype::BF16: {
                auto* src = reinterpret_cast<const uint16_t*>(raw.data());
                for (int64_t i = 0; i < nel; i++) {
                    uint32_t bits = (uint32_t)src[i] << 16;
                    memcpy(&out[i], &bits, sizeof(float));
                }
                break;
            }

            case SafetensorsDtype::F16: {
                auto* src = reinterpret_cast<const uint16_t*>(raw.data());
                for (int64_t i = 0; i < nel; i++) {
                    uint32_t h = src[i];
                    uint32_t sign = (h >> 15) & 1;
                    uint32_t exp = (h >> 10) & 0x1F;
                    uint32_t mant = h & 0x3FF;
                    uint32_t f;
                    if (exp == 0) f = (sign << 31) | (mant << 13);
                    else if (exp == 31) f = (sign << 31) | 0x7F800000 | (mant << 13);
                    else f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                    memcpy(&out[i], &f, sizeof(float));
                }
                break;
            }

            default:
                // For integer types, cast to float
                if (info.dtype == SafetensorsDtype::I32 && raw.size() >= (size_t)nel * 4) {
                    auto* src = reinterpret_cast<const int32_t*>(raw.data());
                    for (int64_t i = 0; i < nel; i++) out[i] = (float)src[i];
                } else if (info.dtype == SafetensorsDtype::I64 && raw.size() >= (size_t)nel * 8) {
                    auto* src = reinterpret_cast<const int64_t*>(raw.data());
                    for (int64_t i = 0; i < nel; i++) out[i] = (float)src[i];
                } else {
                    return {};
                }
                break;
        }

        return out;
    }

    /// Get element count for a tensor.
    int64_t tensorElements(const SafetensorInfo& info) const {
        int64_t n = 1;
        for (auto d : info.shape) n *= d;
        return n;
    }

    /// Get bytes per element for a dtype.
    static size_t dtypeSize(SafetensorsDtype dt) {
        switch (dt) {
            case SafetensorsDtype::F32:
            case SafetensorsDtype::I32: return 4;
            case SafetensorsDtype::F16:
            case SafetensorsDtype::BF16: return 2;
            case SafetensorsDtype::I64: return 8;
            case SafetensorsDtype::U8:
            case SafetensorsDtype::I8:
            case SafetensorsDtype::BOOL: return 1;
            default: return 0;
        }
    }
};
