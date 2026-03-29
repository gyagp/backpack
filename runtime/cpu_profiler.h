#pragma once
/**
 * cpu_profiler.h -- Hierarchical CPU profiler with RAII scope guards.
 *
 * Records CPU events with begin/end timestamps, hierarchical scope nesting,
 * and automatic depth tracking. Events can be exported for HTML timeline
 * visualization alongside GPU profiler data.
 *
 * Usage:
 *   CPUProfiler profiler;
 *   {
 *       auto guard = profiler.scope("load_weights");
 *       // ... work ...
 *       {
 *           auto inner = profiler.scope("upload_gpu");
 *           // ... nested work ...
 *       }
 *   }
 *   // Events now contain "load_weights" (depth 0) and "upload_gpu" (depth 1)
 */

#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct CPUProfiler {
    struct Event {
        std::string name;
        std::string scope;     // dot-separated scope path (e.g. "load.weights")
        uint64_t beginNs;
        uint64_t endNs;
        int depth;
    };

    struct ScopeGuard {
        CPUProfiler& profiler;
        ~ScopeGuard() { profiler.popScope(); }
        // Move-only
        ScopeGuard(CPUProfiler& p) : profiler(p) {}
        ScopeGuard(ScopeGuard&& o) noexcept : profiler(o.profiler) {}
        ScopeGuard(const ScopeGuard&) = delete;
        ScopeGuard& operator=(const ScopeGuard&) = delete;
    };

    std::vector<Event> events;

    /// Enter a named scope. Returns RAII guard that exits the scope on destruction.
    ScopeGuard scope(const std::string& name) {
        scopeStack_.push_back(name);
        Event e;
        e.name = name;
        e.scope = currentScopePath();
        e.beginNs = nowNs();
        e.endNs = 0;
        e.depth = (int)scopeStack_.size() - 1;
        activeIndices_.push_back(events.size());
        events.push_back(e);
        return ScopeGuard(*this);
    }

    /// Record a point event (instantaneous marker).
    void mark(const std::string& name) {
        Event e;
        e.name = name;
        e.scope = currentScopePath();
        e.beginNs = nowNs();
        e.endNs = e.beginNs;
        e.depth = (int)scopeStack_.size();
        events.push_back(e);
    }

    /// Begin a named event (manual pairing with end()).
    void begin(const std::string& name) {
        Event e;
        e.name = name;
        e.scope = currentScopePath();
        e.beginNs = nowNs();
        e.endNs = 0;
        e.depth = (int)scopeStack_.size();
        beginIndices_[name] = events.size();
        events.push_back(e);
    }

    /// End a previously begun event.
    void end(const std::string& name) {
        auto it = beginIndices_.find(name);
        if (it != beginIndices_.end()) {
            events[it->second].endNs = nowNs();
            beginIndices_.erase(it);
        }
    }

    /// Clear all recorded events.
    void clear() {
        events.clear();
        scopeStack_.clear();
        activeIndices_.clear();
        beginIndices_.clear();
    }

    /// Get total elapsed time for all events at depth 0 (top-level scopes).
    double totalMs() const {
        double total = 0;
        for (auto& e : events) {
            if (e.depth == 0 && e.endNs > e.beginNs)
                total += (double)(e.endNs - e.beginNs) / 1e6;
        }
        return total;
    }

private:
    std::vector<std::string> scopeStack_;
    std::vector<size_t> activeIndices_;  // stack of event indices
    std::unordered_map<std::string, size_t> beginIndices_;  // for manual begin/end

    void popScope() {
        if (!activeIndices_.empty()) {
            events[activeIndices_.back()].endNs = nowNs();
            activeIndices_.pop_back();
        }
        if (!scopeStack_.empty()) scopeStack_.pop_back();
    }

    std::string currentScopePath() const {
        std::string path;
        for (size_t i = 0; i < scopeStack_.size(); i++) {
            if (i > 0) path += '.';
            path += scopeStack_[i];
        }
        return path;
    }

    static uint64_t nowNs() {
        auto t = std::chrono::high_resolution_clock::now();
        return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
            t.time_since_epoch()).count();
    }

    friend struct ScopeGuard;
};
