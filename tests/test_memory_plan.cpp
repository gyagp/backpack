#include <algorithm>
#include <cassert>
#include <cstdio>
#include <map>
#include <numeric>
#include <string>
#include <vector>

// Copy structs from execution_context.h to make this self-contained
struct LifetimeInterval {
    std::string name;
    int firstUse;
    int lastUse;
    size_t sizeBytes;
};

struct BufferAssignment {
    int slotId;
    size_t slotSizeBytes;
};

// Copy of GraphExecutor::buildMemoryPlan
std::map<std::string, BufferAssignment> buildMemoryPlan(
    const std::vector<LifetimeInterval>& intervals) {

    std::vector<size_t> order(intervals.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return intervals[a].sizeBytes > intervals[b].sizeBytes;
    });

    struct Slot {
        int id;
        size_t sizeBytes;
        std::vector<std::pair<int, int>> occupied;
    };
    std::vector<Slot> slots;

    std::map<std::string, BufferAssignment> plan;
    size_t totalIndividual = 0;

    for (size_t idx : order) {
        auto& iv = intervals[idx];
        totalIndividual += iv.sizeBytes;

        int bestSlot = -1;
        size_t bestWaste = SIZE_MAX;
        for (size_t s = 0; s < slots.size(); ++s) {
            if (slots[s].sizeBytes < iv.sizeBytes) continue;
            bool overlaps = false;
            for (auto& [a, b] : slots[s].occupied) {
                if (iv.firstUse <= b && iv.lastUse >= a) {
                    overlaps = true;
                    break;
                }
            }
            if (overlaps) continue;
            size_t waste = slots[s].sizeBytes - iv.sizeBytes;
            if (waste < bestWaste) {
                bestWaste = waste;
                bestSlot = static_cast<int>(s);
            }
        }

        if (bestSlot >= 0) {
            slots[bestSlot].occupied.push_back({iv.firstUse, iv.lastUse});
            plan[iv.name] = {slots[bestSlot].id, slots[bestSlot].sizeBytes};
        } else {
            int id = static_cast<int>(slots.size());
            slots.push_back({id, iv.sizeBytes, {{iv.firstUse, iv.lastUse}}});
            plan[iv.name] = {id, iv.sizeBytes};
        }
    }

    return plan;
}

int tests_run = 0;
int tests_passed = 0;

#define TEST(name) \
    do { \
        tests_run++; \
        fprintf(stderr, "  TEST: %s ... ", #name); \
    } while(0)

#define PASS() do { tests_passed++; fprintf(stderr, "PASSED\n"); } while(0)
#define FAIL(msg) do { fprintf(stderr, "FAILED: %s\n", msg); } while(0)

// AC1: BufferAssignment struct has slotId and slotSizeBytes
void test_struct_fields() {
    TEST(struct_fields);
    BufferAssignment ba{3, 1024};
    assert(ba.slotId == 3);
    assert(ba.slotSizeBytes == 1024);
    PASS();
}

// AC2: buildMemoryPlan returns map<string, BufferAssignment>
void test_returns_map() {
    TEST(returns_map);
    std::vector<LifetimeInterval> intervals = {
        {"t0", 0, 2, 1024},
        {"t1", 1, 3, 2048},
    };
    auto plan = buildMemoryPlan(intervals);
    assert(plan.size() == 2);
    assert(plan.count("t0") == 1);
    assert(plan.count("t1") == 1);
    PASS();
}

// AC3: Greedy algorithm — non-overlapping tensors reuse slots
void test_non_overlapping_reuse() {
    TEST(non_overlapping_reuse);
    // t0: [0,1] 1024B, t1: [2,3] 1024B — should share a slot
    std::vector<LifetimeInterval> intervals = {
        {"t0", 0, 1, 1024},
        {"t1", 2, 3, 1024},
    };
    auto plan = buildMemoryPlan(intervals);
    assert(plan["t0"].slotId == plan["t1"].slotId);
    PASS();
}

// AC3: Overlapping tensors get different slots
void test_overlapping_separate() {
    TEST(overlapping_separate);
    std::vector<LifetimeInterval> intervals = {
        {"t0", 0, 2, 1024},
        {"t1", 1, 3, 1024},
    };
    auto plan = buildMemoryPlan(intervals);
    assert(plan["t0"].slotId != plan["t1"].slotId);
    PASS();
}

// AC3: Largest-first ordering — large tensor gets its own slot, smaller fits in existing
void test_largest_first() {
    TEST(largest_first);
    // t_big: [0,1] 4096B, t_small: [2,3] 1024B
    // t_big processed first (larger), t_small can reuse its slot
    std::vector<LifetimeInterval> intervals = {
        {"t_small", 2, 3, 1024},
        {"t_big", 0, 1, 4096},
    };
    auto plan = buildMemoryPlan(intervals);
    assert(plan["t_small"].slotId == plan["t_big"].slotId);
    assert(plan["t_small"].slotSizeBytes == 4096); // uses big slot
    PASS();
}

// AC4: Transformer-like pattern achieves <50% of naive
void test_transformer_memory_savings() {
    TEST(transformer_memory_savings);
    // Simulate a transformer: many intermediate tensors with short lifetimes
    // 12 layers, each producing 3 intermediates that die before next layer
    std::vector<LifetimeInterval> intervals;
    size_t tensorSize = 4 * 1024 * 1024; // 4MB each
    for (int layer = 0; layer < 12; layer++) {
        int base = layer * 3;
        intervals.push_back({"attn_" + std::to_string(layer), base, base + 1, tensorSize});
        intervals.push_back({"ffn1_" + std::to_string(layer), base + 1, base + 2, tensorSize});
        intervals.push_back({"ffn2_" + std::to_string(layer), base + 2, base + 3, tensorSize});
    }
    // 36 tensors × 4MB = 144MB naive
    auto plan = buildMemoryPlan(intervals);

    size_t totalNaive = 0;
    for (auto& iv : intervals) totalNaive += iv.sizeBytes;

    // Count unique slots and their sizes
    std::map<int, size_t> slotSizes;
    for (auto& [name, ba] : plan) {
        slotSizes[ba.slotId] = ba.slotSizeBytes;
    }
    size_t totalPlanned = 0;
    for (auto& [id, sz] : slotSizes) totalPlanned += sz;

    double ratio = (double)totalPlanned / totalNaive;
    fprintf(stderr, "(ratio=%.1f%%) ", ratio * 100);
    assert(ratio <= 0.50);
    PASS();
}

// Edge case: empty input
void test_empty_input() {
    TEST(empty_input);
    auto plan = buildMemoryPlan({});
    assert(plan.empty());
    PASS();
}

// Edge case: single tensor
void test_single_tensor() {
    TEST(single_tensor);
    std::vector<LifetimeInterval> intervals = {{"only", 0, 5, 2048}};
    auto plan = buildMemoryPlan(intervals);
    assert(plan.size() == 1);
    assert(plan["only"].slotId == 0);
    assert(plan["only"].slotSizeBytes == 2048);
    PASS();
}

// Best-fit: picks slot with least waste
void test_best_fit() {
    TEST(best_fit);
    // Create two slots of different sizes, then a small tensor that fits both
    std::vector<LifetimeInterval> intervals = {
        {"big", 0, 1, 8192},
        {"medium", 0, 1, 4096},
        {"small", 2, 3, 3000}, // fits in both slots, should pick medium (less waste)
    };
    auto plan = buildMemoryPlan(intervals);
    assert(plan["small"].slotId == plan["medium"].slotId);
    PASS();
}

int main() {
    fprintf(stderr, "Running buildMemoryPlan tests:\n");
    test_struct_fields();
    test_returns_map();
    test_non_overlapping_reuse();
    test_overlapping_separate();
    test_largest_first();
    test_transformer_memory_savings();
    test_empty_input();
    test_single_tensor();
    test_best_fit();
    fprintf(stderr, "\n%d/%d tests passed.\n", tests_passed, tests_run);
    return tests_passed == tests_run ? 0 : 1;
}
