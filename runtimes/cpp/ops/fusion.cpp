/**
 * fusion.cpp — Graph-level op fusion pass.
 *
 * Analyzes the ONNX graph to detect fuseable patterns and generates
 * combined shaders at pipeline creation time (one-time cost).
 *
 * Currently supported fusions:
 *   1. Elementwise chains: consecutive unary/binary ops on the same tensor
 *      (e.g., Neg → Sigmoid → Mul, Softplus → Sigmoid → Mul)
 *
 * The fusion pass runs once after graph loading. Fused nodes are skipped
 * during Execute() and replaced by a single dispatch of the fused kernel.
 */

#include "../graph_executor.h"
#include "../wgsl_shaders.h"
#include "../wgsl_template.h"
#include <cstdio>
#include <sstream>

// ─── Elementwise op classification ───────────────────────────────────────────

struct ElemwiseOpInfo {
    bool isUnary;         // true = unary (1 input), false = binary (2 inputs)
    int opCode;           // maps to compute_unary/compute_binary op codes
    const char* wgslExpr; // inline WGSL expression for fusion
};

static const std::unordered_map<std::string, ElemwiseOpInfo>& getElemwiseOps() {
    static const std::unordered_map<std::string, ElemwiseOpInfo> ops = {
        // Unary ops
        {"Sigmoid",  {true,  0, "1.0 / (1.0 + exp(-${x}))"}},
        {"Tanh",     {true,  1, "tanh(${x})"}},
        {"Neg",      {true,  2, "-(${x})"}},
        {"Sqrt",     {true,  3, "sqrt(max(${x}, 0.0))"}},
        {"Sin",      {true,  4, "sin(${x})"}},
        {"Cos",      {true,  5, "cos(${x})"}},
        {"Gelu",     {true,  6, "(${x}) * 0.5 * (1.0 + tanh(0.7978845608 * ((${x}) + 0.044715 * (${x}) * (${x}) * (${x}))))"}},
        {"Silu",     {true,  7, "(${x}) / (1.0 + exp(-(${x})))"}},
        {"Relu",     {true, 10, "max(${x}, 0.0)"}},
        {"Exp",      {true, 11, "exp(${x})"}},
        {"Log",      {true, 12, "log(max(${x}, 1e-10))"}},
        {"Abs",      {true, 13, "abs(${x})"}},
        {"Softplus", {true, 18, "select(log(exp(${x}) + 1.0), ${x}, ${x} > 20.0)"}},
        // Binary ops
        {"Add",      {false, 0, "(${a}) + (${b})"}},
        {"Sub",      {false, 1, "(${a}) - (${b})"}},
        {"Mul",      {false, 2, "(${a}) * (${b})"}},
        {"Div",      {false, 3, "(${a}) / (${b})"}},
    };
    return ops;
}

static bool isElemwiseOp(const std::string& opType) {
    return getElemwiseOps().count(opType) > 0;
}

static bool isUnaryElemwise(const std::string& opType) {
    auto it = getElemwiseOps().find(opType);
    return it != getElemwiseOps().end() && it->second.isUnary;
}

// ─── Build consumer map ──────────────────────────────────────────────────────

struct ConsumerInfo {
    std::vector<size_t> consumers;  // node indices that consume this tensor
};

static std::unordered_map<std::string, ConsumerInfo> buildConsumerMap(
    const OnnxGraph& graph) {
    std::unordered_map<std::string, ConsumerInfo> map;
    for (size_t i = 0; i < graph.nodes.size(); i++) {
        for (auto& inp : graph.nodes[i].inputs) {
            if (!inp.empty()) map[inp].consumers.push_back(i);
        }
    }
    return map;
}

// ─── Detect elementwise chains ───────────────────────────────────────────────

/// Find chains of elementwise ops where each intermediate result has exactly
/// one consumer (no branching). Returns chain as list of node indices.
static std::vector<std::vector<size_t>> findElementwiseChains(
    const OnnxGraph& graph,
    const std::unordered_map<std::string, ConsumerInfo>& consumers) {

    std::set<size_t> visited;
    std::vector<std::vector<size_t>> chains;

    for (size_t i = 0; i < graph.nodes.size(); i++) {
        if (visited.count(i)) continue;
        auto& node = graph.nodes[i];
        if (!isElemwiseOp(node.opType)) continue;

        // Try to extend chain forward from this node
        std::vector<size_t> chain = {i};
        visited.insert(i);

        size_t cur = i;
        while (true) {
            auto& curNode = graph.nodes[cur];
            if (curNode.outputs.empty()) break;
            const auto& outName = curNode.outputs[0];

            // Check: output has exactly one consumer
            auto cIt = consumers.find(outName);
            if (cIt == consumers.end() || cIt->second.consumers.size() != 1) break;

            size_t next = cIt->second.consumers[0];
            if (visited.count(next)) break;
            auto& nextNode = graph.nodes[next];
            if (!isElemwiseOp(nextNode.opType)) break;

            // For binary ops, one input must come from the chain and
            // the other must be an external input (not part of chain)
            if (!isUnaryElemwise(nextNode.opType)) {
                // Binary op: check that exactly one input is from chain
                bool hasChainInput = false;
                for (auto& inp : nextNode.inputs) {
                    if (inp == outName) { hasChainInput = true; break; }
                }
                if (!hasChainInput) break;
            }

            chain.push_back(next);
            visited.insert(next);
            cur = next;
        }

        if (chain.size() >= 2) {
            chains.push_back(std::move(chain));
        }
    }
    return chains;
}

// ─── Generate fused elementwise shader ───────────────────────────────────────

struct FusedElemStep {
    std::string opType;
    bool isUnary;
    int extraBindingIdx;   // for binary ops: which extra binding has the 2nd input (-1 = use X)
    bool chainIsSecond;    // true if chain value is the 2nd operand (b), external is 1st (a)
};

static std::string generateFusedElementwiseShader(
    const std::vector<FusedElemStep>& steps,
    TensorDtype dtype) {

    // Count bindings: binding 0 = input(X), binding 1 = output(Y), binding 2 = params
    // bindings 3+ = extra inputs for binary ops in the chain
    // Binary ops with extraBindingIdx == -1 reuse X (binding 0) — no extra binding
    int nextBinding = 3;
    std::vector<int> stepExtraBindings;  // shader binding index for each step's extra
    for (auto& step : steps) {
        if (!step.isUnary && step.extraBindingIdx >= 0) {
            stepExtraBindings.push_back(nextBinding++);
        } else {
            stepExtraBindings.push_back(-1);
        }
    }
    int totalBindings = nextBinding;

    std::ostringstream ss;

    // Header: t_read / t_write2 helpers
    ss << "${T_READ}\n${T_WRITE2}\n\n";

    // Bindings
    ss << "@group(0) @binding(0) var<storage, read> X: array<${T}>;\n";
    ss << "@group(0) @binding(1) var<storage, read_write> Y: array<${T}>;\n";
    ss << "@group(0) @binding(2) var<storage, read> _params_: array<u32>;\n";
    for (int b = 3; b < totalBindings; b++) {
        ss << "@group(0) @binding(" << b << ") var<storage, read> E"
           << (b - 3) << ": array<${T}>;\n";
    }

    ss << "\n@compute @workgroup_size(256)\n";
    ss << "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n";
    ss << "    let N = _params_[0];\n";
    // Extra N values for broadcasting
    for (int b = 3; b < totalBindings; b++) {
        ss << "    let N_E" << (b - 3) << " = _params_[" << (b - 2) << "];\n";
    }
    ss << "    let base = gid.x * 2u;\n";
    ss << "    if (base >= N) { return; }\n\n";

    // Process pair of elements
    ss << "    // Element 0\n";
    ss << "    var v0: f32 = t_read(&X, base);\n";

    const auto& opsMap = getElemwiseOps();
    // Helper: generate expression for one element at a given index expr
    auto genStepExpr = [&](const char* varName, const char* idxExpr,
                           const char* indent) {
        for (size_t si = 0; si < steps.size(); si++) {
            auto it = opsMap.find(steps[si].opType);
            if (it == opsMap.end()) continue;
            if (steps[si].isUnary) {
                std::string expr = it->second.wgslExpr;
                replaceAll(expr, "${x}", varName);
                ss << indent << varName << " = " << expr << ";\n";
            } else {
                std::string expr = it->second.wgslExpr;
                // Determine the "other" read expression
                std::string otherRead;
                int shaderBinding = stepExtraBindings[si];
                if (shaderBinding >= 0) {
                    int bi = shaderBinding - 3;
                    otherRead = "t_read(&E" + std::to_string(bi) +
                        ", select(" + idxExpr + ", " + idxExpr + " % N_E" +
                        std::to_string(bi) + ", N_E" + std::to_string(bi) +
                        " < N && N_E" + std::to_string(bi) + " > 0u))";
                } else {
                    // Reuse X (the chain's primary input)
                    otherRead = std::string("t_read(&X, ") + idxExpr + ")";
                }
                // Assign a and b based on operand order
                std::string aVal, bVal;
                if (steps[si].chainIsSecond) {
                    aVal = otherRead;
                    bVal = varName;
                } else {
                    aVal = varName;
                    bVal = otherRead;
                }
                replaceAll(expr, "${a}", aVal);
                replaceAll(expr, "${b}", bVal);
                ss << indent << varName << " = " << expr << ";\n";
            }
        }
    };

    genStepExpr("v0", "base", "    ");

    ss << "\n    // Element 1\n";
    ss << "    var v1: f32 = 0.0;\n";
    ss << "    if (base + 1u < N) {\n";
    ss << "        v1 = t_read(&X, base + 1u);\n";

    genStepExpr("v1", "base + 1u", "        ");

    ss << "    }\n";
    ss << "    t_write2(&Y, base, v0, v1);\n";
    ss << "}\n";

    return instantiateTemplate(ss.str().c_str(), dtype);
}

// ─── DetectFusions — main entry point ────────────────────────────────────────

void GraphExecutor::DetectFusions() {
    fusedGroups_.clear();
    fusedNodeIndices_.clear();

    // Fusion disabled for now — focus on dynamic shader generation first
    return;

    auto consumers = buildConsumerMap(graph_);
    auto chains = findElementwiseChains(graph_, consumers);

    int totalFused = 0;
    for (auto& chain : chains) {
        if (chain.size() < 2) continue;

        // Build step descriptors
        std::vector<FusedElemStep> steps;
        std::vector<std::string> externalInputs;
        std::string chainInputName;  // first input to the chain

        for (size_t ci = 0; ci < chain.size(); ci++) {
            auto& node = graph_.nodes[chain[ci]];
            auto it = getElemwiseOps().find(node.opType);
            if (it == getElemwiseOps().end()) break;

            FusedElemStep step;
            step.opType = node.opType;
            step.isUnary = it->second.isUnary;
            step.extraBindingIdx = -1;
            step.chainIsSecond = false;

            if (ci == 0) {
                // First node: primary input comes from outside
                chainInputName = node.inputs[0];
            }

            if (!step.isUnary && node.inputs.size() >= 2) {
                // Binary op: figure out which input is from chain and which is external
                std::string prevOutput;
                if (ci > 0) prevOutput = graph_.nodes[chain[ci - 1]].outputs[0];

                // Determine chain input (from previous step or X)
                std::string chainIn = ci > 0 ? prevOutput : chainInputName;

                // Identify operand order: is chain value input[0] or input[1]?
                std::string otherInput;
                if (node.inputs[0] == chainIn) {
                    // chain is a, other is b
                    otherInput = node.inputs[1];
                    step.chainIsSecond = false;
                } else if (node.inputs[1] == chainIn) {
                    // chain is b, other is a
                    otherInput = node.inputs[0];
                    step.chainIsSecond = true;
                } else {
                    // Neither input is from chain — shouldn't happen
                    break;
                }

                // Check if the other input is the chain's primary input (X)
                if (otherInput == chainInputName) {
                    // Use X directly — no extra binding needed
                    step.extraBindingIdx = -1;
                } else {
                    externalInputs.push_back(otherInput);
                    step.extraBindingIdx = (int)externalInputs.size() - 1;
                }
            }
            steps.push_back(step);
        }

        if (steps.size() < 2) continue;

        // Build fusion group
        FusedGroup group;
        group.nodeIndices = chain;
        group.externalInputs = {chainInputName};
        for (auto& ext : externalInputs) group.externalInputs.push_back(ext);
        group.outputName = graph_.nodes[chain.back()].outputs[0];
        group.numBindings = 3 + (uint32_t)externalInputs.size();  // X, Y, params, extras

        // Build pipeline name from op sequence
        std::string pname = "fused";
        for (auto& s : steps) pname += "_" + s.opType;

        // Determine dtype from first node's expected input type
        // (will be resolved at dispatch time based on actual tensor dtype)
        group.pipelineName = pname;

        // Capture steps for deferred shader generation
        auto capturedSteps = steps;
        group.shaderGenerator = [capturedSteps](TensorDtype dt) {
            return generateFusedElementwiseShader(capturedSteps, dt);
        };

        size_t firstIdx = chain[0];
        fusedGroups_[firstIdx] = std::move(group);
        for (size_t idx : chain) {
            fusedNodeIndices_.insert(idx);
        }
        totalFused += (int)chain.size();
    }

    if (!fusedGroups_.empty()) {
        fprintf(stderr, "  [fusion] Detected %zu fuseable groups (%d nodes) from %zu total\n",
                fusedGroups_.size(), totalFused, graph_.nodes.size());
        for (auto& [idx, g] : fusedGroups_) {
            fprintf(stderr, "    %s (%zu ops → 1 dispatch)\n",
                    g.pipelineName.c_str(), g.nodeIndices.size());
        }
        fflush(stderr);
    }
}
