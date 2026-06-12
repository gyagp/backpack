#include "../src/sampling.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

static void test_greedy_returns_argmax() {
    float logits[] = {1.0f, 5.0f, 3.0f, 2.0f, 4.0f};
    uint32_t result = sample_greedy(logits, 5);
    assert(result == 1 && "greedy should return index of max value");
    printf("  PASS: greedy returns argmax\n");
}

static void test_greedy_first_occurrence() {
    float logits[] = {3.0f, 3.0f, 1.0f};
    uint32_t result = sample_greedy(logits, 3);
    assert(result == 0 && "greedy should return first occurrence on tie");
    printf("  PASS: greedy first occurrence on tie\n");
}

static void test_topk_restricts_candidates() {
    float logits[] = {1.0f, 10.0f, 2.0f, 9.0f, 0.5f};
    uint32_t k = 2;
    std::mt19937 rng(42);
    std::map<uint32_t, int> counts;
    for (int i = 0; i < 1000; i++)
        counts[sample_topk(logits, 5, k, 1.0f, rng)]++;

    assert(counts.size() <= 2 && "top-k=2 should only produce 2 distinct tokens");
    for (auto& [tok, _] : counts)
        assert((tok == 1 || tok == 3) && "top-k=2 should only select indices 1 and 3");
    printf("  PASS: top-k restricts candidates to k\n");
}

static void test_topk_temperature() {
    float logits[] = {0.0f, 10.0f, 9.9f};
    std::mt19937 rng(123);

    std::map<uint32_t, int> low_temp_counts;
    for (int i = 0; i < 1000; i++)
        low_temp_counts[sample_topk(logits, 3, 2, 0.01f, rng)]++;
    assert(low_temp_counts[1] > 950 && "low temperature should strongly prefer max");

    std::map<uint32_t, int> high_temp_counts;
    for (int i = 0; i < 1000; i++)
        high_temp_counts[sample_topk(logits, 3, 2, 100.0f, rng)]++;
    assert(high_temp_counts[1] < 700 && "high temperature should spread probability");
    printf("  PASS: top-k temperature affects distribution\n");
}

static void test_topp_filters_by_cumulative_probability() {
    float logits[10];
    logits[0] = 100.0f;
    for (int i = 1; i < 10; i++)
        logits[i] = 0.0f;

    std::mt19937 rng(42);
    std::map<uint32_t, int> counts;
    for (int i = 0; i < 500; i++)
        counts[sample_topp(logits, 10, 0.5f, 1.0f, rng)]++;

    assert(counts.size() == 1 && counts[0] == 500 &&
           "with one dominant logit, top-p should only select it");
    printf("  PASS: top-p filters by cumulative probability (dominant)\n");
}

static void test_topp_includes_enough_tokens() {
    float logits[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    std::mt19937 rng(42);
    std::map<uint32_t, int> counts;
    for (int i = 0; i < 2000; i++)
        counts[sample_topp(logits, 5, 0.99f, 1.0f, rng)]++;

    assert(counts.size() == 5 && "p=0.99 with uniform logits should include all tokens");
    printf("  PASS: top-p includes enough tokens for high p\n");
}

static void test_topp_nucleus_cutoff() {
    float logits[] = {10.0f, 9.0f, 0.0f, 0.0f, 0.0f};
    std::mt19937 rng(42);
    std::map<uint32_t, int> counts;
    for (int i = 0; i < 1000; i++)
        counts[sample_topp(logits, 5, 0.5f, 1.0f, rng)]++;

    for (auto& [tok, _] : counts)
        assert(tok <= 1 && "p=0.5 should only select top 1-2 tokens from skewed distribution");
    printf("  PASS: top-p nucleus cutoff excludes low-probability tail\n");
}

int main() {
    printf("test_sampling:\n");
    test_greedy_returns_argmax();
    test_greedy_first_occurrence();
    test_topk_restricts_candidates();
    test_topk_temperature();
    test_topp_filters_by_cumulative_probability();
    test_topp_includes_enough_tokens();
    test_topp_nucleus_cutoff();
    printf("All sampling tests passed.\n");
    return 0;
}
