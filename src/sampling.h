#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

inline uint32_t sample_greedy(const float* logits, uint32_t vocab_size) {
    uint32_t best = 0;
    float best_val = logits[0];
    for (uint32_t i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best = i;
        }
    }
    return best;
}

inline uint32_t sample_topk(const float* logits, uint32_t vocab_size,
                            uint32_t k, float temperature,
                            std::mt19937& rng) {
    k = (std::min)(k, vocab_size);

    std::vector<uint32_t> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                      [&](uint32_t a, uint32_t b) { return logits[a] > logits[b]; });
    indices.resize(k);

    float inv_temp = 1.0f / temperature;
    std::vector<float> probs(k);
    float max_val = logits[indices[0]];
    for (uint32_t i = 0; i < k; i++)
        probs[i] = std::exp((logits[indices[i]] - max_val) * inv_temp);

    float sum = 0.0f;
    for (uint32_t i = 0; i < k; i++)
        sum += probs[i];
    for (uint32_t i = 0; i < k; i++)
        probs[i] /= sum;

    std::discrete_distribution<uint32_t> dist(probs.begin(), probs.end());
    return indices[dist(rng)];
}

inline uint32_t sample_topp(const float* logits, uint32_t vocab_size,
                            float p, float temperature,
                            std::mt19937& rng) {
    float inv_temp = 1.0f / temperature;
    float max_val = logits[0];
    for (uint32_t i = 1; i < vocab_size; i++)
        max_val = (std::max)(max_val, logits[i]);

    std::vector<std::pair<float, uint32_t>> scored(vocab_size);
    float sum = 0.0f;
    for (uint32_t i = 0; i < vocab_size; i++) {
        float prob = std::exp((logits[i] - max_val) * inv_temp);
        scored[i] = {prob, i};
        sum += prob;
    }
    for (auto& s : scored)
        s.first /= sum;

    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    float cumulative = 0.0f;
    uint32_t cutoff = 0;
    for (uint32_t i = 0; i < vocab_size; i++) {
        cumulative += scored[i].first;
        cutoff = i + 1;
        if (cumulative >= p)
            break;
    }

    std::vector<float> probs(cutoff);
    for (uint32_t i = 0; i < cutoff; i++)
        probs[i] = scored[i].first;

    std::discrete_distribution<uint32_t> dist(probs.begin(), probs.end());
    return scored[dist(rng)].second;
}
