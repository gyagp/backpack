#pragma once

#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "mmap_file.h"

constexpr uint32_t GGUF_MAGIC = 0x46475547;

enum class GGUFValueType : uint32_t {
    UINT8    = 0,
    INT8     = 1,
    UINT16   = 2,
    INT16    = 3,
    UINT32   = 4,
    INT32    = 5,
    FLOAT32  = 6,
    BOOL     = 7,
    STRING   = 8,
    ARRAY    = 9,
    UINT64   = 10,
    INT64    = 11,
    FLOAT64  = 12,
};

struct GGUFArray;

using GGUFValue = std::variant<
    uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, float, bool,
    std::string, GGUFArray, uint64_t, int64_t, double>;

struct GGUFArray {
    GGUFValueType element_type;
    std::vector<GGUFValue> values;
};

enum class GGUFDType : uint32_t {
    F32    = 0,
    F16    = 1,
    Q8_0   = 8,
    Q4_K_M = 15,
};

struct GGUFTensorInfo {
    std::string name;
    uint32_t ndims;
    std::vector<uint64_t> dimensions;
    GGUFDType dtype;
    uint64_t offset;
};

constexpr size_t GGUF_ALIGNMENT = 32;

struct GGUFFile {
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
    std::map<std::string, GGUFValue> metadata;
    std::vector<GGUFTensorInfo> tensors;
    uint64_t data_offset;
};

class GGUFParser {
public:
    static GGUFFile parse(const uint8_t* data, size_t size) {
        GGUFParser p(data, size);
        return p.parse_file();
    }

    static GGUFFile parse(const std::string& path) {
        MmapFile mmap(path);
        if (!mmap.is_open())
            throw std::runtime_error("GGUF: failed to open " + path);
        return parse(mmap.data(), mmap.size());
    }

private:
    const uint8_t* data_;
    size_t size_;
    size_t pos_ = 0;

    GGUFParser(const uint8_t* data, size_t size) : data_(data), size_(size) {}

    void check(size_t n) const {
        if (pos_ + n > size_) throw std::runtime_error("GGUF: unexpected end of data");
    }

    template<typename T> T read() {
        check(sizeof(T));
        T val;
        std::memcpy(&val, data_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return val;
    }

    std::string read_string() {
        uint64_t len = read<uint64_t>();
        check(len);
        std::string s(reinterpret_cast<const char*>(data_ + pos_), len);
        pos_ += len;
        return s;
    }

    GGUFValue read_value(GGUFValueType type) {
        switch (type) {
            case GGUFValueType::UINT8:   return read<uint8_t>();
            case GGUFValueType::INT8:    return read<int8_t>();
            case GGUFValueType::UINT16:  return read<uint16_t>();
            case GGUFValueType::INT16:   return read<int16_t>();
            case GGUFValueType::UINT32:  return read<uint32_t>();
            case GGUFValueType::INT32:   return read<int32_t>();
            case GGUFValueType::FLOAT32: return read<float>();
            case GGUFValueType::BOOL:    return static_cast<bool>(read<uint8_t>());
            case GGUFValueType::STRING:  return read_string();
            case GGUFValueType::UINT64:  return read<uint64_t>();
            case GGUFValueType::INT64:   return read<int64_t>();
            case GGUFValueType::FLOAT64: return read<double>();
            case GGUFValueType::ARRAY: {
                auto elem_type = static_cast<GGUFValueType>(read<uint32_t>());
                uint64_t count = read<uint64_t>();
                GGUFArray arr;
                arr.element_type = elem_type;
                arr.values.reserve(count);
                for (uint64_t i = 0; i < count; i++)
                    arr.values.push_back(read_value(elem_type));
                return arr;
            }
        }
        throw std::runtime_error("GGUF: unknown value type");
    }

    GGUFFile parse_file() {
        uint32_t magic = read<uint32_t>();
        if (magic != GGUF_MAGIC)
            throw std::runtime_error("GGUF: invalid magic number");

        GGUFFile file;
        file.version = read<uint32_t>();
        file.tensor_count = read<uint64_t>();
        file.metadata_kv_count = read<uint64_t>();

        for (uint64_t i = 0; i < file.metadata_kv_count; i++) {
            std::string key = read_string();
            auto vtype = static_cast<GGUFValueType>(read<uint32_t>());
            file.metadata[key] = read_value(vtype);
        }

        file.tensors.resize(file.tensor_count);
        for (uint64_t i = 0; i < file.tensor_count; i++) {
            GGUFTensorInfo& t = file.tensors[i];
            t.name = read_string();
            t.ndims = read<uint32_t>();
            t.dimensions.resize(t.ndims);
            for (uint32_t d = 0; d < t.ndims; d++)
                t.dimensions[d] = read<uint64_t>();
            t.dtype = static_cast<GGUFDType>(read<uint32_t>());
            t.offset = read<uint64_t>();
        }

        file.data_offset = (pos_ + GGUF_ALIGNMENT - 1) & ~(GGUF_ALIGNMENT - 1);

        return file;
    }
};
