#pragma once
/**
 * wav_reader.h — Minimal WAV file reader for ASR input.
 *
 * Reads PCM WAV files (16-bit int or 32-bit float, mono) and returns
 * audio samples as a vector of float32. Validates 16kHz sample rate.
 */

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace asr {

struct WavHeader {
    uint32_t sampleRate = 0;
    uint16_t numChannels = 0;
    uint16_t bitsPerSample = 0;
    uint16_t audioFormat = 0;  // 1 = PCM int, 3 = IEEE float
    uint32_t dataSize = 0;
};

inline bool readWav(const std::string& path, std::vector<float>& samples,
                    int expectedSampleRate = 16000) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        fprintf(stderr, "Error: cannot open WAV file: %s\n", path.c_str());
        return false;
    }

    // Read RIFF header
    char riff[4];
    f.read(riff, 4);
    if (memcmp(riff, "RIFF", 4) != 0) {
        fprintf(stderr, "Error: not a RIFF file: %s\n", path.c_str());
        return false;
    }

    uint32_t fileSize;
    f.read(reinterpret_cast<char*>(&fileSize), 4);

    char wave[4];
    f.read(wave, 4);
    if (memcmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "Error: not a WAVE file: %s\n", path.c_str());
        return false;
    }

    WavHeader hdr{};

    // Scan chunks
    while (f.good()) {
        char chunkId[4];
        uint32_t chunkSize;
        f.read(chunkId, 4);
        f.read(reinterpret_cast<char*>(&chunkSize), 4);
        if (!f.good()) break;

        if (memcmp(chunkId, "fmt ", 4) == 0) {
            auto pos = f.tellg();
            f.read(reinterpret_cast<char*>(&hdr.audioFormat), 2);
            f.read(reinterpret_cast<char*>(&hdr.numChannels), 2);
            f.read(reinterpret_cast<char*>(&hdr.sampleRate), 4);
            uint32_t byteRate;
            f.read(reinterpret_cast<char*>(&byteRate), 4);
            uint16_t blockAlign;
            f.read(reinterpret_cast<char*>(&blockAlign), 2);
            f.read(reinterpret_cast<char*>(&hdr.bitsPerSample), 2);
            f.seekg(pos + (std::streamoff)chunkSize);
        } else if (memcmp(chunkId, "data", 4) == 0) {
            hdr.dataSize = chunkSize;

            // Validate format
            if (hdr.numChannels != 1) {
                fprintf(stderr, "Error: WAV must be mono, got %d channels\n",
                        hdr.numChannels);
                return false;
            }
            if ((int)hdr.sampleRate != expectedSampleRate) {
                fprintf(stderr, "Error: WAV must be %dHz, got %dHz\n",
                        expectedSampleRate, hdr.sampleRate);
                return false;
            }
            if (hdr.audioFormat != 1 && hdr.audioFormat != 3) {
                fprintf(stderr, "Error: unsupported WAV format %d "
                        "(only PCM int16 and float32)\n", hdr.audioFormat);
                return false;
            }

            // Read audio data
            if (hdr.audioFormat == 3 && hdr.bitsPerSample == 32) {
                // IEEE float32
                uint32_t numSamples = hdr.dataSize / 4;
                samples.resize(numSamples);
                f.read(reinterpret_cast<char*>(samples.data()),
                       numSamples * sizeof(float));
            } else if (hdr.audioFormat == 1 && hdr.bitsPerSample == 16) {
                // PCM int16
                uint32_t numSamples = hdr.dataSize / 2;
                std::vector<int16_t> raw(numSamples);
                f.read(reinterpret_cast<char*>(raw.data()),
                       numSamples * sizeof(int16_t));
                samples.resize(numSamples);
                for (uint32_t i = 0; i < numSamples; i++) {
                    samples[i] = (float)raw[i] / 32768.0f;
                }
            } else {
                fprintf(stderr, "Error: unsupported bit depth %d for "
                        "format %d\n", hdr.bitsPerSample, hdr.audioFormat);
                return false;
            }

            printf("  WAV: %dHz, %d-bit %s, %.2fs (%zu samples)\n",
                   hdr.sampleRate, hdr.bitsPerSample,
                   hdr.audioFormat == 3 ? "float" : "PCM",
                   (float)samples.size() / hdr.sampleRate, samples.size());
            return true;
        } else {
            // Skip unknown chunk
            f.seekg(chunkSize, std::ios::cur);
        }
    }

    fprintf(stderr, "Error: no data chunk found in WAV file\n");
    return false;
}

}  // namespace asr
