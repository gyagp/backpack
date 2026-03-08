#include "clock_calibration.h"
#include <cstdio>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

// Get current CPU time in nanoseconds (QPC on Windows, clock_gettime on Linux)
static uint64_t cpuTimeNs() {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (uint64_t)((double)counter.QuadPart / freq.QuadPart * 1e9);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
#endif
}

// ─── D3D12 Calibration ──────────────────────────────────────────────────────

#ifdef _WIN32
#include <d3d12.h>
#include <wrl/client.h>

// Load Dawn's GetD3D12CommandQueue dynamically from webgpu_dawn.dll
// The mangled name is found via dumpbin /exports.
typedef Microsoft::WRL::ComPtr<ID3D12CommandQueue> (*PFN_GetD3D12CommandQueue)(WGPUDevice);

static ClockCalibration calibrateD3D12(WGPUDevice device) {
    ClockCalibration cal{};

    HMODULE dawn = GetModuleHandleA("webgpu_dawn.dll");
    if (!dawn) {
        fprintf(stderr, "Clock calibration: webgpu_dawn.dll not loaded\n");
        return cal;
    }

    // Mangled name: ?GetD3D12CommandQueue@d3d12@native@dawn@@YA?AV?$ComPtr@UID3D12CommandQueue@@@WRL@Microsoft@@PEAUWGPUDeviceImpl@@@Z
    auto getQueue = (PFN_GetD3D12CommandQueue)GetProcAddress(dawn,
        "?GetD3D12CommandQueue@d3d12@native@dawn@@YA?AV?$ComPtr@UID3D12CommandQueue@@@WRL@Microsoft@@PEAUWGPUDeviceImpl@@@Z");
    if (!getQueue) {
        fprintf(stderr, "Clock calibration: GetD3D12CommandQueue not found in Dawn\n");
        return cal;
    }

    auto queue = getQueue(device);
    if (!queue) {
        fprintf(stderr, "Clock calibration: failed to get D3D12 command queue\n");
        return cal;
    }

    // Get GPU timestamp frequency (ticks per second)
    UINT64 gpuFreq = 0;
    HRESULT hr = queue->GetTimestampFrequency(&gpuFreq);
    if (FAILED(hr) || gpuFreq == 0) {
        fprintf(stderr, "Clock calibration: GetTimestampFrequency failed\n");
        return cal;
    }

    // Get correlated GPU/CPU timestamps
    UINT64 gpuTick = 0, cpuTick = 0;
    hr = queue->GetClockCalibration(&gpuTick, &cpuTick);
    if (FAILED(hr)) {
        fprintf(stderr, "Clock calibration: GetClockCalibration failed\n");
        return cal;
    }

    // Convert GPU tick to nanoseconds
    // Dawn normalizes D3D12 timestamps to nanoseconds, so we need to match.
    // D3D12 raw ticks * (1e9 / gpuFreq) = nanoseconds
    cal.gpuTimestampNs = (uint64_t)((double)gpuTick * 1e9 / gpuFreq);

    // CPU QPC tick to nanoseconds
    LARGE_INTEGER qpcFreq;
    QueryPerformanceFrequency(&qpcFreq);
    cal.cpuTimestampNs = (uint64_t)((double)cpuTick * 1e9 / qpcFreq.QuadPart);

    // D3D12 doesn't report deviation; assume <1us
    cal.maxDeviationNs = 1000;
    cal.valid = true;

    return cal;
}
#endif

// ─── Vulkan Calibration ──────────────────────────────────────────────────────

// Dawn native API: loaded dynamically from webgpu_dawn.dll / .so
typedef void* VkInstance_T;
typedef void* VkPhysicalDevice_T;
typedef void* VkDevice_T;
typedef void (*PFN_vkVoidFunction)();

typedef VkInstance_T* (*PFN_DawnGetVkInstance)(WGPUDevice);
typedef PFN_vkVoidFunction (*PFN_DawnGetVkInstanceProcAddr)(WGPUDevice, const char*);

static ClockCalibration calibrateVulkan(WGPUDevice device) {
    ClockCalibration cal{};

    // Load Dawn's Vulkan accessors dynamically
#ifdef _WIN32
    HMODULE dawn = GetModuleHandleA("webgpu_dawn.dll");
    if (!dawn) return cal;

    // Mangled names from dumpbin:
    auto dawnGetInstance = (PFN_DawnGetVkInstance)GetProcAddress(dawn,
        "?GetInstance@vulkan@native@dawn@@YAPEAUVkInstance_T@@PEAUWGPUDeviceImpl@@@Z");
    auto dawnGetProcAddr = (PFN_DawnGetVkInstanceProcAddr)GetProcAddress(dawn,
        "?GetInstanceProcAddr@vulkan@native@dawn@@YAP6AXXZPEAUWGPUDeviceImpl@@PEBD@Z");
#else
    // On Linux, symbols are in the loaded .so
    auto dawnGetInstance = (PFN_DawnGetVkInstance)dlsym(RTLD_DEFAULT,
        "_ZN4dawn6native6vulkan11GetInstanceEP14WGPUDeviceImpl");
    auto dawnGetProcAddr = (PFN_DawnGetVkInstanceProcAddr)dlsym(RTLD_DEFAULT,
        "_ZN4dawn6native6vulkan18GetInstanceProcAddrEP14WGPUDeviceImplPKc");
#endif

    if (!dawnGetInstance || !dawnGetProcAddr) {
        fprintf(stderr, "Clock calibration: Dawn Vulkan native API not found\n");
        return cal;
    }

    auto vkInstance = dawnGetInstance(device);
    if (!vkInstance) {
        fprintf(stderr, "Clock calibration: failed to get VkInstance\n");
        return cal;
    }

    auto getProc = [&](const char* name) -> PFN_vkVoidFunction {
        return dawnGetProcAddr(device, name);
    };

    // Vulkan constants
    constexpr uint32_t VK_TIME_DOMAIN_DEVICE_EXT = 0;
#ifdef _WIN32
    constexpr uint32_t VK_TIME_DOMAIN_CPU_EXT = 3; // QPC
#else
    constexpr uint32_t VK_TIME_DOMAIN_CPU_EXT = 1; // CLOCK_MONOTONIC
#endif

    // Vulkan structs
    struct VkCalibratedTimestampInfoEXT {
        uint32_t sType; const void* pNext; uint32_t timeDomain;
    };
    struct VkDeviceQueueCreateInfo {
        uint32_t sType; const void* pNext; uint32_t flags;
        uint32_t queueFamilyIndex; uint32_t queueCount;
        const float* pQueuePriorities;
    };
    struct VkDeviceCreateInfo {
        uint32_t sType; const void* pNext; uint32_t flags;
        uint32_t queueCreateInfoCount;
        const VkDeviceQueueCreateInfo* pQueueCreateInfos;
        uint32_t enabledLayerCount; const char* const* ppEnabledLayerNames;
        uint32_t enabledExtensionCount; const char* const* ppEnabledExtensionNames;
        const void* pEnabledFeatures;
    };

    // Vulkan function pointer types
    typedef int32_t (*FN_vkEnumPhys)(VkInstance_T*, uint32_t*, VkPhysicalDevice_T**);
    typedef int32_t (*FN_vkCreateDev)(VkPhysicalDevice_T*, const VkDeviceCreateInfo*,
                                      const void*, VkDevice_T**);
    typedef void (*FN_vkDestroyDev)(VkDevice_T*, const void*);
    typedef PFN_vkVoidFunction (*FN_vkGetDevProc)(VkDevice_T*, const char*);
    typedef int32_t (*FN_vkGetCalDomains)(VkPhysicalDevice_T*, uint32_t*, uint32_t*);
    typedef int32_t (*FN_vkGetCalTimestamps)(VkDevice_T*, uint32_t,
        const VkCalibratedTimestampInfoEXT*, uint64_t*, uint64_t*);

    auto vkEnumPhysDevs = (FN_vkEnumPhys)getProc("vkEnumeratePhysicalDevices");
    auto vkCreateDev = (FN_vkCreateDev)getProc("vkCreateDevice");
    auto vkDestroyDev = (FN_vkDestroyDev)getProc("vkDestroyDevice");
    auto vkGetDevProc = (FN_vkGetDevProc)getProc("vkGetDeviceProcAddr");
    auto vkGetPhysDevCalTimeDomains = (FN_vkGetCalDomains)getProc(
        "vkGetPhysicalDeviceCalibrateableTimeDomainsEXT");

    if (!vkEnumPhysDevs || !vkCreateDev || !vkDestroyDev || !vkGetDevProc) {
        fprintf(stderr, "Clock calibration: missing Vulkan functions\n");
        return cal;
    }

    if (!vkGetPhysDevCalTimeDomains) {
        // Try KHR variant (Vulkan 1.3+)
        vkGetPhysDevCalTimeDomains = (FN_vkGetCalDomains)getProc(
                "vkGetPhysicalDeviceCalibrateableTimeDomainsKHR");
    }

    if (!vkGetPhysDevCalTimeDomains) {
        fprintf(stderr, "Clock calibration: VK_EXT_calibrated_timestamps not available\n");
        return cal;
    }

    // Enumerate physical devices — use the first one (same as Dawn)
    uint32_t physDevCount = 0;
    vkEnumPhysDevs(vkInstance, &physDevCount, nullptr);
    if (physDevCount == 0) return cal;

    VkPhysicalDevice_T* physDev = nullptr;
    VkPhysicalDevice_T* physDevs[8] = {};
    uint32_t cnt = physDevCount < 8 ? physDevCount : 8;
    vkEnumPhysDevs(vkInstance, &cnt, physDevs);
    physDev = physDevs[0];

    // Check supported time domains
    uint32_t domainCount = 0;
    vkGetPhysDevCalTimeDomains(physDev, &domainCount, nullptr);
    if (domainCount == 0) return cal;

    uint32_t domains[16] = {};
    uint32_t dc = domainCount < 16 ? domainCount : 16;
    vkGetPhysDevCalTimeDomains(physDev, &dc, domains);

    bool hasDevice = false, hasCpu = false;
    uint32_t cpuDomain = VK_TIME_DOMAIN_CPU_EXT;
    for (uint32_t i = 0; i < dc; i++) {
        if (domains[i] == VK_TIME_DOMAIN_DEVICE_EXT) hasDevice = true;
        if (domains[i] == cpuDomain) hasCpu = true;
    }
    if (!hasDevice || !hasCpu) {
        fprintf(stderr, "Clock calibration: required time domains not supported\n");
        return cal;
    }

    // Create a temporary VkDevice for vkGetCalibratedTimestampsEXT
    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = 2;  // VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO
    qci.queueFamilyIndex = 0;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;

    const char* ext = "VK_EXT_calibrated_timestamps";
    VkDeviceCreateInfo dci{};
    dci.sType = 3;  // VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1;
    dci.ppEnabledExtensionNames = &ext;

    VkDevice_T* vkDev = nullptr;
    int32_t vkResult = vkCreateDev(physDev, &dci, nullptr, &vkDev);
    if (vkResult != 0 || !vkDev) {
        // Try KHR extension name (might be promoted)
        ext = "VK_KHR_calibrated_timestamps";
        vkResult = vkCreateDev(physDev, &dci, nullptr, &vkDev);
        if (vkResult != 0 || !vkDev) {
            // Try without the extension (might be in core 1.3)
            dci.enabledExtensionCount = 0;
            dci.ppEnabledExtensionNames = nullptr;
            vkResult = vkCreateDev(physDev, &dci, nullptr, &vkDev);
            if (vkResult != 0 || !vkDev) {
                fprintf(stderr, "Clock calibration: failed to create temp VkDevice\n");
                return cal;
            }
        }
    }

    // Get device-level calibration function
    auto vkGetCalTimestamps = (FN_vkGetCalTimestamps)
        vkGetDevProc(vkDev, "vkGetCalibratedTimestampsEXT");
    if (!vkGetCalTimestamps) {
        vkGetCalTimestamps = (FN_vkGetCalTimestamps)
            vkGetDevProc(vkDev, "vkGetCalibratedTimestampsKHR");
    }

    if (!vkGetCalTimestamps) {
        fprintf(stderr, "Clock calibration: vkGetCalibratedTimestamps not found\n");
        vkDestroyDev(vkDev, nullptr);
        return cal;
    }

    // Get calibrated timestamps: [GPU device, CPU QPC/monotonic]
    VkCalibratedTimestampInfoEXT infos[2] = {};
    infos[0].sType = 1000184000;  // VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT
    infos[0].timeDomain = VK_TIME_DOMAIN_DEVICE_EXT;
    infos[1].sType = 1000184000;
    infos[1].timeDomain = cpuDomain;

    uint64_t timestamps[2] = {};
    uint64_t maxDeviation = 0;
    vkResult = vkGetCalTimestamps(vkDev, 2, infos, timestamps, &maxDeviation);

    vkDestroyDev(vkDev, nullptr);

    if (vkResult != 0) {
        fprintf(stderr, "Clock calibration: vkGetCalibratedTimestamps failed (%d)\n",
                vkResult);
        return cal;
    }

    // On NVIDIA Vulkan, timestampPeriod is 1.0 (timestamps already in ns).
    // Dawn also normalizes to nanoseconds. So timestamps[0] is in nanoseconds.
    cal.gpuTimestampNs = timestamps[0];

#ifdef _WIN32
    // timestamps[1] is QPC ticks — convert to nanoseconds
    LARGE_INTEGER qpcFreq;
    QueryPerformanceFrequency(&qpcFreq);
    cal.cpuTimestampNs = (uint64_t)((double)timestamps[1] * 1e9 / qpcFreq.QuadPart);
#else
    // timestamps[1] is CLOCK_MONOTONIC nanoseconds
    cal.cpuTimestampNs = timestamps[1];
#endif

    cal.maxDeviationNs = maxDeviation;
    cal.valid = true;

    return cal;
}

// ─── Public API ──────────────────────────────────────────────────────────────

ClockCalibration acquireClockCalibration(WGPUDevice device,
                                         WGPUBackendType backend) {
#ifdef _WIN32
    if (backend == WGPUBackendType_D3D12) {
        return calibrateD3D12(device);
    }
#endif
    if (backend == WGPUBackendType_Vulkan) {
        return calibrateVulkan(device);
    }

    // Unsupported backend — return invalid calibration
    return {};
}
