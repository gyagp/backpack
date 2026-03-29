#pragma once
/**
 * profile_html.h -- Generate interactive HTML profiling report.
 *
 * Writes a self-contained HTML file with a canvas-based GPU timeline,
 * optional CPU flamechart lane, summary tables, and zoom/pan support.
 * Same visual format as the Python profiler_html.py output.
 */

#include "gpu_context.h"
#include "clock_calibration.h"
#include "cpu_profiler.h"
#include <string>

/// Generate an HTML profiling report from GPU timestamp data.
/// The profiler's readbackBuf must already be mapped via printProfileReport.
void generateProfileHTML(
    GPUContext& gpu,
    GPUProfiler& profiler,
    const ClockCalibration* calibration,
    const uint64_t* timestampData,   // resolved GPU timestamps (ns)
    int nDecodeTokens,               // number of decode steps
    int nPrefillTokens,              // number of prefill tokens
    double prefillMs,                // prefill wall time
    double decodeMs,                  // decode wall time
    const std::string& outputPath = "profile.html",
    const CPUProfiler* cpuProfiler = nullptr);  // optional CPU events
