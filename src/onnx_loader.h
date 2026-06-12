#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "mmap_file.h"

namespace onnx {

enum DataType : int32_t {
    UNDEFINED = 0,
    FLOAT     = 1,
    UINT8     = 2,
    INT8      = 3,
    UINT16    = 4,
    INT16     = 5,
    INT32     = 6,
    INT64     = 7,
    STRING    = 8,
    BOOL      = 9,
    FLOAT16   = 10,
    DOUBLE    = 11,
    UINT32    = 12,
    UINT64    = 13,
    BFLOAT16  = 16,
};

struct TensorShape {
    std::vector<int64_t> dims;
};

struct ValueInfo {
    std::string name;
    DataType elem_type = UNDEFINED;
    TensorShape shape;
};

struct TensorProto {
    std::string name;
    DataType data_type = UNDEFINED;
    std::vector<int64_t> dims;
    const uint8_t* raw_data = nullptr;
    size_t raw_data_size = 0;
    std::vector<float> float_data;
    std::vector<int32_t> int32_data;
    std::vector<int64_t> int64_data;
};

struct AttributeProto {
    std::string name;
    int32_t type = 0;
    float f = 0;
    int64_t i = 0;
    std::string s;
    std::vector<float> floats;
    std::vector<int64_t> ints;
};

struct NodeProto {
    std::string name;
    std::string op_type;
    std::string domain;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<AttributeProto> attributes;
};

struct GraphProto {
    std::string name;
    std::vector<NodeProto> nodes;
    std::vector<ValueInfo> inputs;
    std::vector<ValueInfo> outputs;
    std::vector<TensorProto> initializers;
};

struct ModelProto {
    int64_t ir_version = 0;
    int64_t opset_version = 0;
    std::string producer_name;
    std::string producer_version;
    std::string domain;
    GraphProto graph;
    MmapFile backing_store;
};

class PBReader {
    const uint8_t* data_;
    const uint8_t* end_;
public:
    PBReader(const uint8_t* data, size_t size)
        : data_(data), end_(data + size) {}

    bool done() const { return data_ >= end_; }
    size_t remaining() const { return static_cast<size_t>(end_ - data_); }

    uint64_t read_varint() {
        uint64_t result = 0;
        int shift = 0;
        while (data_ < end_) {
            uint8_t b = *data_++;
            result |= static_cast<uint64_t>(b & 0x7F) << shift;
            if ((b & 0x80) == 0) return result;
            shift += 7;
            if (shift >= 64) throw std::runtime_error("varint too long");
        }
        throw std::runtime_error("unexpected end of varint");
    }

    uint32_t read_fixed32() {
        if (remaining() < 4) throw std::runtime_error("unexpected end");
        uint32_t v;
        std::memcpy(&v, data_, 4);
        data_ += 4;
        return v;
    }

    uint64_t read_fixed64() {
        if (remaining() < 8) throw std::runtime_error("unexpected end");
        uint64_t v;
        std::memcpy(&v, data_, 8);
        data_ += 8;
        return v;
    }

    PBReader read_length_delimited() {
        uint64_t len = read_varint();
        if (len > remaining()) throw std::runtime_error("length exceeds buffer");
        PBReader sub(data_, static_cast<size_t>(len));
        data_ += len;
        return sub;
    }

    std::string read_string() {
        uint64_t len = read_varint();
        if (len > remaining()) throw std::runtime_error("string length exceeds buffer");
        std::string s(reinterpret_cast<const char*>(data_), static_cast<size_t>(len));
        data_ += len;
        return s;
    }

    const uint8_t* ptr() const { return data_; }

    void advance(size_t n) {
        if (n > remaining()) throw std::runtime_error("advance past end");
        data_ += n;
    }

    void skip_field(uint32_t wire_type) {
        switch (wire_type) {
        case 0: read_varint(); break;
        case 1: if (remaining() < 8) throw std::runtime_error("skip"); data_ += 8; break;
        case 2: { uint64_t len = read_varint(); if (len > remaining()) throw std::runtime_error("skip"); data_ += len; } break;
        case 5: if (remaining() < 4) throw std::runtime_error("skip"); data_ += 4; break;
        default: throw std::runtime_error("unknown wire type");
        }
    }

    struct Tag {
        uint32_t field;
        uint32_t wire_type;
    };

    Tag read_tag() {
        uint64_t v = read_varint();
        return {static_cast<uint32_t>(v >> 3), static_cast<uint32_t>(v & 0x7)};
    }
};

inline TensorShape parse_tensor_shape(PBReader r) {
    TensorShape shape;
    while (!r.done()) {
        auto tag = r.read_tag();
        if (tag.field == 1 && tag.wire_type == 2) {
            // dim message
            PBReader dim_r = r.read_length_delimited();
            while (!dim_r.done()) {
                auto dt = dim_r.read_tag();
                if (dt.field == 1 && dt.wire_type == 0) {
                    shape.dims.push_back(static_cast<int64_t>(dim_r.read_varint()));
                } else if (dt.field == 2 && dt.wire_type == 2) {
                    dim_r.read_string(); // dim_param string, skip
                } else {
                    dim_r.skip_field(dt.wire_type);
                }
            }
        } else {
            r.skip_field(tag.wire_type);
        }
    }
    return shape;
}

inline ValueInfo parse_value_info(PBReader r) {
    ValueInfo vi;
    while (!r.done()) {
        auto tag = r.read_tag();
        if (tag.field == 1 && tag.wire_type == 2) {
            vi.name = r.read_string();
        } else if (tag.field == 2 && tag.wire_type == 2) {
            // TypeProto
            PBReader type_r = r.read_length_delimited();
            while (!type_r.done()) {
                auto tt = type_r.read_tag();
                if (tt.field == 1 && tt.wire_type == 2) {
                    // tensor_type
                    PBReader tensor_r = type_r.read_length_delimited();
                    while (!tensor_r.done()) {
                        auto ttt = tensor_r.read_tag();
                        if (ttt.field == 1 && ttt.wire_type == 0) {
                            vi.elem_type = static_cast<DataType>(tensor_r.read_varint());
                        } else if (ttt.field == 2 && ttt.wire_type == 2) {
                            vi.shape = parse_tensor_shape(tensor_r.read_length_delimited());
                        } else {
                            tensor_r.skip_field(ttt.wire_type);
                        }
                    }
                } else {
                    type_r.skip_field(tt.wire_type);
                }
            }
        } else {
            r.skip_field(tag.wire_type);
        }
    }
    return vi;
}

inline TensorProto parse_tensor(PBReader r) {
    TensorProto t;
    while (!r.done()) {
        auto tag = r.read_tag();
        switch (tag.field) {
        case 1: // dims (repeated int64, packed or unpacked)
            if (tag.wire_type == 0) {
                t.dims.push_back(static_cast<int64_t>(r.read_varint()));
            } else if (tag.wire_type == 2) {
                PBReader packed = r.read_length_delimited();
                while (!packed.done())
                    t.dims.push_back(static_cast<int64_t>(packed.read_varint()));
            } else {
                r.skip_field(tag.wire_type);
            }
            break;
        case 2: // data_type
            t.data_type = static_cast<DataType>(r.read_varint());
            break;
        case 4: // float_data (packed)
            if (tag.wire_type == 2) {
                PBReader packed = r.read_length_delimited();
                while (!packed.done()) {
                    uint32_t bits = packed.read_fixed32();
                    float f;
                    std::memcpy(&f, &bits, 4);
                    t.float_data.push_back(f);
                }
            } else if (tag.wire_type == 5) {
                uint32_t bits = r.read_fixed32();
                float f;
                std::memcpy(&f, &bits, 4);
                t.float_data.push_back(f);
            } else {
                r.skip_field(tag.wire_type);
            }
            break;
        case 5: // int32_data (packed)
            if (tag.wire_type == 2) {
                PBReader packed = r.read_length_delimited();
                while (!packed.done())
                    t.int32_data.push_back(static_cast<int32_t>(packed.read_varint()));
            } else if (tag.wire_type == 0) {
                t.int32_data.push_back(static_cast<int32_t>(r.read_varint()));
            } else {
                r.skip_field(tag.wire_type);
            }
            break;
        case 7: // int64_data (packed)
            if (tag.wire_type == 2) {
                PBReader packed = r.read_length_delimited();
                while (!packed.done())
                    t.int64_data.push_back(static_cast<int64_t>(packed.read_varint()));
            } else if (tag.wire_type == 0) {
                t.int64_data.push_back(static_cast<int64_t>(r.read_varint()));
            } else {
                r.skip_field(tag.wire_type);
            }
            break;
        case 8: // name
            t.name = r.read_string();
            break;
        case 13: // raw_data
            if (tag.wire_type == 2) {
                uint64_t len = r.read_varint();
                t.raw_data = r.ptr();
                t.raw_data_size = static_cast<size_t>(len);
                r.advance(static_cast<size_t>(len));
            } else {
                r.skip_field(tag.wire_type);
            }
            break;
        default:
            r.skip_field(tag.wire_type);
            break;
        }
    }
    return t;
}

inline AttributeProto parse_attribute(PBReader r) {
    AttributeProto a;
    while (!r.done()) {
        auto tag = r.read_tag();
        switch (tag.field) {
        case 1: a.name = r.read_string(); break;
        case 2: { uint64_t v = r.read_varint(); a.type = static_cast<int32_t>(v); } break;
        case 3: a.i = static_cast<int64_t>(r.read_varint()); break;
        case 4:
            if (tag.wire_type == 5) {
                uint32_t bits = r.read_fixed32();
                std::memcpy(&a.f, &bits, 4);
            } else { r.skip_field(tag.wire_type); }
            break;
        case 5: a.s = r.read_string(); break;
        case 7: // floats
            if (tag.wire_type == 2) {
                PBReader packed = r.read_length_delimited();
                while (!packed.done()) {
                    uint32_t bits = packed.read_fixed32();
                    float f;
                    std::memcpy(&f, &bits, 4);
                    a.floats.push_back(f);
                }
            } else { r.skip_field(tag.wire_type); }
            break;
        case 8: // ints
            if (tag.wire_type == 2) {
                PBReader packed = r.read_length_delimited();
                while (!packed.done())
                    a.ints.push_back(static_cast<int64_t>(packed.read_varint()));
            } else if (tag.wire_type == 0) {
                a.ints.push_back(static_cast<int64_t>(r.read_varint()));
            } else { r.skip_field(tag.wire_type); }
            break;
        default:
            r.skip_field(tag.wire_type);
            break;
        }
    }
    return a;
}

inline NodeProto parse_node(PBReader r) {
    NodeProto n;
    while (!r.done()) {
        auto tag = r.read_tag();
        switch (tag.field) {
        case 1: n.inputs.push_back(r.read_string()); break;
        case 2: n.outputs.push_back(r.read_string()); break;
        case 3: n.name = r.read_string(); break;
        case 4: n.op_type = r.read_string(); break;
        case 5: // attribute
            if (tag.wire_type == 2)
                n.attributes.push_back(parse_attribute(r.read_length_delimited()));
            else r.skip_field(tag.wire_type);
            break;
        case 7: n.domain = r.read_string(); break;
        default: r.skip_field(tag.wire_type); break;
        }
    }
    return n;
}

inline GraphProto parse_graph(PBReader r) {
    GraphProto g;
    while (!r.done()) {
        auto tag = r.read_tag();
        switch (tag.field) {
        case 1: // node
            if (tag.wire_type == 2)
                g.nodes.push_back(parse_node(r.read_length_delimited()));
            else r.skip_field(tag.wire_type);
            break;
        case 2: g.name = r.read_string(); break;
        case 5: // initializer
            if (tag.wire_type == 2)
                g.initializers.push_back(parse_tensor(r.read_length_delimited()));
            else r.skip_field(tag.wire_type);
            break;
        case 11: // input
            if (tag.wire_type == 2)
                g.inputs.push_back(parse_value_info(r.read_length_delimited()));
            else r.skip_field(tag.wire_type);
            break;
        case 12: // output
            if (tag.wire_type == 2)
                g.outputs.push_back(parse_value_info(r.read_length_delimited()));
            else r.skip_field(tag.wire_type);
            break;
        default: r.skip_field(tag.wire_type); break;
        }
    }
    return g;
}

inline ModelProto parse_model(const uint8_t* data, size_t size) {
    PBReader r(data, size);
    ModelProto m;
    while (!r.done()) {
        auto tag = r.read_tag();
        switch (tag.field) {
        case 1: m.ir_version = static_cast<int64_t>(r.read_varint()); break;
        case 2: // opset_import
            if (tag.wire_type == 2) {
                PBReader opset_r = r.read_length_delimited();
                while (!opset_r.done()) {
                    auto ot = opset_r.read_tag();
                    if (ot.field == 2 && ot.wire_type == 0)
                        m.opset_version = static_cast<int64_t>(opset_r.read_varint());
                    else
                        opset_r.skip_field(ot.wire_type);
                }
            } else r.skip_field(tag.wire_type);
            break;
        case 3: m.producer_name = r.read_string(); break;
        case 4: m.producer_version = r.read_string(); break;
        case 5: m.domain = r.read_string(); break;
        case 7: // graph
            if (tag.wire_type == 2)
                m.graph = parse_graph(r.read_length_delimited());
            else r.skip_field(tag.wire_type);
            break;
        default: r.skip_field(tag.wire_type); break;
        }
    }
    return m;
}

inline ModelProto load_onnx(const std::string& path) {
    MmapFile mmap(path);
    if (!mmap.is_open())
        throw std::runtime_error("ONNX: failed to open " + path);
    ModelProto m = parse_model(mmap.data(), mmap.size());
    m.backing_store = std::move(mmap);
    return m;
}

} // namespace onnx
