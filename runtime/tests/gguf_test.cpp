#include "gguf_loader.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        GGUFFile metadata;
        metadata.metadata["general.architecture"] = std::string("qwen35");
        metadata.metadata["qwen35.block_count"] = uint32_t(65);
        metadata.metadata["qwen35.nextn_predict_layers"] = uint32_t(1);
        auto config = extractModelConfig(metadata);
        if (config.nLayer != 64 || !config.hasMtp || config.mtpNumLayers != 1)
            throw std::runtime_error("MTP block was not separated from target decoder");
        metadata.metadata["qwen35.nextn_predict_layers"] = uint32_t(0);
        if (extractModelConfig(metadata).nLayer != 65)
            throw std::runtime_error("ordinary decoder lost a layer");
        if (!extractModelConfig(metadata).tieWordEmbeddings)
            throw std::runtime_error("missing output tensor should imply tied embeddings");
        metadata.tensor_index["output.weight"] = 0;
        if (extractModelConfig(metadata).tieWordEmbeddings)
            throw std::runtime_error("output tensor was ignored when detecting untied embeddings");
        metadata.metadata["qwen35.nextn_predict_layers"] = uint32_t(65);
        bool rejected = false;
        try { extractModelConfig(metadata); }
        catch (const std::runtime_error&) { rejected = true; }
        if (!rejected) throw std::runtime_error("invalid MTP layer count accepted");

        for (int arg = 1; arg < argc; ++arg) {
            // Fixture: u32 type, rows, cols, byte_count; raw bytes; fp32 reference.
            std::ifstream file(argv[arg], std::ios::binary);
            uint32_t header[4]{};
            file.read(reinterpret_cast<char*>(header), sizeof(header));
            if (!file || header[1] == 0 || header[2] == 0 || header[3] > (1u << 24))
                throw std::runtime_error("invalid quantization fixture header");
            const size_t count = size_t(header[1]) * header[2];
            std::vector<uint8_t> raw(header[3]);
            std::vector<float> expected(count), actual(count);
            file.read(reinterpret_cast<char*>(raw.data()), raw.size());
            file.read(reinterpret_cast<char*>(expected.data()), count * sizeof(float));
            if (!file) throw std::runtime_error("truncated quantization fixture");
            dequant_tensor(raw.data(), actual.data(), header[1], header[2], GGUFType(header[0]));
            float maxError = 0;
            for (size_t i = 0; i < count; ++i) {
                const float error = std::abs(actual[i] - expected[i]);
                if (!std::isfinite(actual[i]) || error > 1e-6f + 1e-6f * std::abs(expected[i])) {
                    std::fprintf(stderr, "%s[%zu]: got %.9g expected %.9g\n", argv[arg], i, actual[i], expected[i]);
                    return 1;
                }
                maxError = std::max(maxError, error);
            }
            std::printf("PASS type=%u values=%zu max_error=%.9g\n", header[0], count, maxError);
        }
        std::puts("PASS GGUF target/MTP layer counts");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
