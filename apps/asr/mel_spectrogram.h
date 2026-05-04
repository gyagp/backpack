#pragma once
/**
 * mel_spectrogram.h — Log-mel spectrogram computation for ASR.
 *
 * Computes a librosa-compatible log-mel spectrogram from raw audio samples.
 * Parameters are tuned for Whisper/Qwen-ASR family models:
 *   N_FFT=400, HOP_LENGTH=160, N_MELS=128, SAMPLE_RATE=16000
 *
 * Output: [N_MELS, n_frames] float32, log-scaled and normalized.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace asr {

static constexpr int N_FFT = 400;
static constexpr int HOP_LENGTH = 160;
static constexpr int N_MELS = 128;
static constexpr int SAMPLE_RATE = 16000;

// FFT size: next power of 2 >= N_FFT
static constexpr int FFT_SIZE = 512;
static constexpr int N_FREQ = FFT_SIZE / 2 + 1;  // 257

// ─── Mel scale helpers ──────────────────────────────────────────────────────

inline float hzToMel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

inline float melToHz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

// ─── Mel filterbank ─────────────────────────────────────────────────────────
// Returns [N_MELS x N_FREQ] matrix (row-major).

inline std::vector<float> buildMelFilterbank() {
    const float fmin = 0.0f;
    const float fmax = (float)SAMPLE_RATE / 2.0f;  // 8000 Hz

    float melMin = hzToMel(fmin);
    float melMax = hzToMel(fmax);

    // N_MELS + 2 equally spaced points in mel scale
    std::vector<float> melPoints(N_MELS + 2);
    for (int i = 0; i < N_MELS + 2; i++) {
        melPoints[i] = melMin + (melMax - melMin) * i / (N_MELS + 1);
    }

    // Convert back to Hz, then to FFT bin indices
    std::vector<float> hzPoints(N_MELS + 2);
    std::vector<float> bins(N_MELS + 2);
    for (int i = 0; i < N_MELS + 2; i++) {
        hzPoints[i] = melToHz(melPoints[i]);
        bins[i] = hzPoints[i] * FFT_SIZE / SAMPLE_RATE;
    }

    // Build triangular filterbank with Slaney normalization
    std::vector<float> fb(N_MELS * N_FREQ, 0.0f);
    for (int m = 0; m < N_MELS; m++) {
        float left = bins[m];
        float center = bins[m + 1];
        float right = bins[m + 2];

        // Slaney normalization factor
        float enorm = 2.0f / (hzPoints[m + 2] - hzPoints[m]);

        for (int k = 0; k < N_FREQ; k++) {
            float fk = (float)k;
            float weight = 0.0f;
            if (fk >= left && fk <= center && center > left) {
                weight = (fk - left) / (center - left);
            } else if (fk > center && fk <= right && right > center) {
                weight = (right - fk) / (right - center);
            }
            fb[m * N_FREQ + k] = weight * enorm;
        }
    }
    return fb;
}

// ─── Radix-2 FFT ───────────────────────────────────────────────────────────
// In-place FFT on complex array of length N (must be power of 2).
// real[i] and imag[i] are interleaved.

inline void fft(float* real, float* imag, int N) {
    // Bit-reversal permutation
    for (int i = 1, j = 0; i < N; i++) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }

    // Cooley-Tukey butterfly
    for (int len = 2; len <= N; len <<= 1) {
        float angle = -2.0f * (float)M_PI / len;
        float wR = std::cos(angle);
        float wI = std::sin(angle);
        for (int i = 0; i < N; i += len) {
            float curR = 1.0f, curI = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                float tR = curR * real[i + j + len / 2] -
                           curI * imag[i + j + len / 2];
                float tI = curR * imag[i + j + len / 2] +
                           curI * real[i + j + len / 2];
                float uR = real[i + j];
                float uI = imag[i + j];
                real[i + j] = uR + tR;
                imag[i + j] = uI + tI;
                real[i + j + len / 2] = uR - tR;
                imag[i + j + len / 2] = uI - tI;
                float newCurR = curR * wR - curI * wI;
                curI = curR * wI + curI * wR;
                curR = newCurR;
            }
        }
    }
}

// ─── Log-mel spectrogram ────────────────────────────────────────────────────
// Returns [N_MELS, n_frames] in row-major order.

inline std::vector<float> computeLogMelSpectrogram(
    const std::vector<float>& audio, int& outFrames) {

    // Reflect-pad the audio by N_FFT/2 on each side
    int padLen = N_FFT / 2;
    int paddedLen = (int)audio.size() + 2 * padLen;
    std::vector<float> padded(paddedLen);

    // Left reflect pad
    for (int i = 0; i < padLen; i++) {
        padded[i] = audio[padLen - i];
    }
    // Center: original audio
    for (int i = 0; i < (int)audio.size(); i++) {
        padded[padLen + i] = audio[i];
    }
    // Right reflect pad
    for (int i = 0; i < padLen; i++) {
        int idx = (int)audio.size() - 2 - i;
        padded[padLen + (int)audio.size() + i] = (idx >= 0) ? audio[idx] : 0.0f;
    }

    // Hann window
    float hannWindow[N_FFT];
    for (int i = 0; i < N_FFT; i++) {
        hannWindow[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * i / N_FFT));
    }

    // Compute number of frames
    int nFrames = (paddedLen - N_FFT) / HOP_LENGTH + 1;
    outFrames = nFrames;

    // Allocate power spectrogram [N_FREQ, nFrames]
    std::vector<float> powerSpec(N_FREQ * nFrames, 0.0f);

    // FFT buffers (reused per frame)
    float fftReal[FFT_SIZE], fftImag[FFT_SIZE];

    for (int frame = 0; frame < nFrames; frame++) {
        int start = frame * HOP_LENGTH;

        // Apply window and zero-pad to FFT_SIZE
        for (int i = 0; i < FFT_SIZE; i++) {
            if (i < N_FFT && start + i < paddedLen) {
                fftReal[i] = padded[start + i] * hannWindow[i];
            } else {
                fftReal[i] = 0.0f;
            }
            fftImag[i] = 0.0f;
        }

        fft(fftReal, fftImag, FFT_SIZE);

        // Power spectrum: |X[k]|^2
        for (int k = 0; k < N_FREQ; k++) {
            powerSpec[k * nFrames + frame] =
                fftReal[k] * fftReal[k] + fftImag[k] * fftImag[k];
        }
    }

    // Build mel filterbank
    auto melFb = buildMelFilterbank();

    // Mel spectrogram: [N_MELS, nFrames] = melFb [N_MELS, N_FREQ] x powerSpec [N_FREQ, nFrames]
    std::vector<float> melSpec(N_MELS * nFrames, 0.0f);
    for (int m = 0; m < N_MELS; m++) {
        for (int t = 0; t < nFrames; t++) {
            float sum = 0.0f;
            for (int k = 0; k < N_FREQ; k++) {
                sum += melFb[m * N_FREQ + k] * powerSpec[k * nFrames + t];
            }
            melSpec[m * nFrames + t] = sum;
        }
    }

    // Log-mel scaling (Whisper-style):
    // log10(max(mel, 1e-10)), clamp to max-8, then (x+4)/4
    float maxLog = -1e30f;
    for (int i = 0; i < N_MELS * nFrames; i++) {
        melSpec[i] = std::log10(std::max(melSpec[i], 1e-10f));
        maxLog = std::max(maxLog, melSpec[i]);
    }
    float clampMin = maxLog - 8.0f;
    for (int i = 0; i < N_MELS * nFrames; i++) {
        melSpec[i] = (std::max(melSpec[i], clampMin) + 4.0f) / 4.0f;
    }

    printf("  Mel spectrogram: [%d, %d] (%.2fs audio)\n",
           N_MELS, nFrames,
           (float)audio.size() / SAMPLE_RATE);
    return melSpec;
}

}  // namespace asr
