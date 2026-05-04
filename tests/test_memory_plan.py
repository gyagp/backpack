"""Test suite for buildMemoryPlan greedy buffer assignment algorithm.
Reimplements the C++ algorithm in Python to verify correctness."""

import sys

def build_memory_plan(intervals):
    """intervals: list of dict {name, firstUse, lastUse, sizeBytes}"""
    order = sorted(range(len(intervals)), key=lambda i: intervals[i]["sizeBytes"], reverse=True)

    slots = []  # list of {id, sizeBytes, occupied: [(first, last)]}
    plan = {}

    for idx in order:
        iv = intervals[idx]
        best_slot = -1
        best_waste = float("inf")
        for s_idx, slot in enumerate(slots):
            if slot["sizeBytes"] < iv["sizeBytes"]:
                continue
            overlaps = any(iv["firstUse"] <= b and iv["lastUse"] >= a for a, b in slot["occupied"])
            if overlaps:
                continue
            waste = slot["sizeBytes"] - iv["sizeBytes"]
            if waste < best_waste:
                best_waste = waste
                best_slot = s_idx

        if best_slot >= 0:
            slots[best_slot]["occupied"].append((iv["firstUse"], iv["lastUse"]))
            plan[iv["name"]] = {"slotId": slots[best_slot]["id"], "slotSizeBytes": slots[best_slot]["sizeBytes"]}
        else:
            sid = len(slots)
            slots.append({"id": sid, "sizeBytes": iv["sizeBytes"], "occupied": [(iv["firstUse"], iv["lastUse"])]})
            plan[iv["name"]] = {"slotId": sid, "slotSizeBytes": iv["sizeBytes"]}

    return plan, slots


passed = 0
failed = 0

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name} — {detail}")


# AC1: BufferAssignment has slotId and slotSizeBytes
plan, _ = build_memory_plan([{"name": "t0", "firstUse": 0, "lastUse": 2, "sizeBytes": 1024}])
test("AC1: struct fields", "slotId" in plan["t0"] and "slotSizeBytes" in plan["t0"])

# AC2: returns map<string, BufferAssignment>
intervals = [
    {"name": "t0", "firstUse": 0, "lastUse": 2, "sizeBytes": 1024},
    {"name": "t1", "firstUse": 1, "lastUse": 3, "sizeBytes": 2048},
]
plan, _ = build_memory_plan(intervals)
test("AC2: returns map with all tensors", len(plan) == 2 and "t0" in plan and "t1" in plan)

# AC3: Non-overlapping reuse
intervals = [
    {"name": "t0", "firstUse": 0, "lastUse": 1, "sizeBytes": 1024},
    {"name": "t1", "firstUse": 2, "lastUse": 3, "sizeBytes": 1024},
]
plan, _ = build_memory_plan(intervals)
test("AC3a: non-overlapping reuse same slot", plan["t0"]["slotId"] == plan["t1"]["slotId"])

# AC3: Overlapping → separate slots
intervals = [
    {"name": "t0", "firstUse": 0, "lastUse": 2, "sizeBytes": 1024},
    {"name": "t1", "firstUse": 1, "lastUse": 3, "sizeBytes": 1024},
]
plan, _ = build_memory_plan(intervals)
test("AC3b: overlapping get separate slots", plan["t0"]["slotId"] != plan["t1"]["slotId"])

# AC3: Largest-first ordering
intervals = [
    {"name": "t_small", "firstUse": 2, "lastUse": 3, "sizeBytes": 1024},
    {"name": "t_big", "firstUse": 0, "lastUse": 1, "sizeBytes": 4096},
]
plan, _ = build_memory_plan(intervals)
test("AC3c: largest first, small reuses big slot",
     plan["t_small"]["slotId"] == plan["t_big"]["slotId"] and plan["t_small"]["slotSizeBytes"] == 4096)

# AC3: Best-fit selection
intervals = [
    {"name": "big", "firstUse": 0, "lastUse": 1, "sizeBytes": 8192},
    {"name": "medium", "firstUse": 0, "lastUse": 1, "sizeBytes": 4096},
    {"name": "small", "firstUse": 2, "lastUse": 3, "sizeBytes": 3000},
]
plan, _ = build_memory_plan(intervals)
test("AC3d: best-fit picks least waste", plan["small"]["slotId"] == plan["medium"]["slotId"])

# AC4: Transformer-like achieves ≤50% of naive
intervals = []
tensor_size = 4 * 1024 * 1024
for layer in range(12):
    base = layer * 3
    intervals.append({"name": f"attn_{layer}", "firstUse": base, "lastUse": base + 1, "sizeBytes": tensor_size})
    intervals.append({"name": f"ffn1_{layer}", "firstUse": base + 1, "lastUse": base + 2, "sizeBytes": tensor_size})
    intervals.append({"name": f"ffn2_{layer}", "firstUse": base + 2, "lastUse": base + 3, "sizeBytes": tensor_size})

plan, slots = build_memory_plan(intervals)
total_naive = sum(iv["sizeBytes"] for iv in intervals)
total_planned = sum(s["sizeBytes"] for s in slots)
ratio = total_planned / total_naive
test(f"AC4: transformer ≤50% (got {ratio*100:.1f}%)", ratio <= 0.50)

# Edge cases
plan, _ = build_memory_plan([])
test("Edge: empty input", len(plan) == 0)

plan, _ = build_memory_plan([{"name": "only", "firstUse": 0, "lastUse": 5, "sizeBytes": 2048}])
test("Edge: single tensor", len(plan) == 1 and plan["only"]["slotId"] == 0 and plan["only"]["slotSizeBytes"] == 2048)

print(f"\n{passed}/{passed+failed} tests passed.")
sys.exit(0 if failed == 0 else 1)
