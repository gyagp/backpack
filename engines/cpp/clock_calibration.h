#pragma once
/**
 * clock_calibration.h -- CPU/GPU clock calibration for profiling.
 *
 * Provides precise CPU<->GPU timeline alignment using native APIs:
 *   D3D12:  ID3D12CommandQueue::GetClockCalibration() + GetTimestampFrequency()
 *   Vulkan: vkGetCalibratedTimestampsEXT() with VK_TIME_DOMAIN_DEVICE + QPC
 *
 * Dawn normalizes GPU timestamps to nanoseconds, so the calibration data
 * maps GPU nanosecond timestamps to the CPU QPC/perf_counter timeline.
 */

#include <webgpu/webgpu.h>
#include <cstdint>

struct ClockCalibration {
    uint64_t gpuTimestampNs;   // GPU timestamp at calibration point (ns)
    uint64_t cpuTimestampNs;   // CPU timestamp at same instant (ns)
    uint64_t maxDeviationNs;   // upper bound on measurement error (ns)
    bool valid;

    /// Convert a GPU timestamp (in nanoseconds, as Dawn provides) to CPU ns.
    uint64_t gpuNsToCpuNs(uint64_t gpuNs) const {
        if (!valid) return gpuNs;
        int64_t delta = (int64_t)gpuNs - (int64_t)gpuTimestampNs;
        return (uint64_t)((int64_t)cpuTimestampNs + delta);
    }
};

/// Acquire clock calibration for the given Dawn device and backend.
/// On D3D12: uses GetClockCalibration via Dawn's native D3D12 queue access.
/// On Vulkan: uses vkGetCalibratedTimestampsEXT via Dawn's VkInstance access.
ClockCalibration acquireClockCalibration(WGPUDevice device,
                                         WGPUBackendType backend);
