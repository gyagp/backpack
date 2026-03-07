import ctypes
import ctypes.util
import os
import sys
import time


FEATURE_ONLY_SHADER = """
enable subgroups;
enable chromium_experimental_subgroup_matrix;

@group(0) @binding(0) var<storage, read_write> Out: array<f32>;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    Out[lid.x] = 1.0;
}
"""


TYPE_DECL_SHADER = """
enable subgroups;
enable chromium_experimental_subgroup_matrix;

@group(0) @binding(0) var<storage, read_write> Out: array<f32>;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    var matA: subgroup_matrix_left<f32, 8, 8>;
    var matB: subgroup_matrix_right<f32, 8, 8>;
    var matC: subgroup_matrix_result<f32, 8, 8>;
    Out[lid.x] = 0.0;
}
"""


MMA_SHADER = """
enable subgroups;
enable chromium_experimental_subgroup_matrix;

@group(0) @binding(0) var<storage, read_write> Out: array<f32>;

var<workgroup> TileA: array<f32, 64>;
var<workgroup> TileB: array<f32, 64>;
var<workgroup> TileC: array<f32, 64>;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = lid.x;
    TileA[idx] = 1.0;
    TileA[idx + 32u] = 1.0;
    TileB[idx] = 1.0;
    TileB[idx + 32u] = 1.0;
    TileC[idx] = 0.0;
    TileC[idx + 32u] = 0.0;
    workgroupBarrier();

    let matA = subgroupMatrixLoad<subgroup_matrix_left<f32, 8, 8>>(
        &TileA, 0u, false, 8u);
    let matB = subgroupMatrixLoad<subgroup_matrix_right<f32, 8, 8>>(
        &TileB, 0u, true, 8u);
    var matC: subgroup_matrix_result<f32, 8, 8>;
    matC = subgroupMatrixMultiplyAccumulate(matA, matB, matC);
    subgroupMatrixStore(&TileC, 0u, matC, false, 8u);
    workgroupBarrier();

    Out[idx] = TileC[idx];
    Out[idx + 32u] = TileC[idx + 32u];
}
"""


WGPUInstance = ctypes.c_void_p
WGPUAdapter = ctypes.c_void_p
WGPUDevice = ctypes.c_void_p
WGPUQueue = ctypes.c_void_p
WGPUShaderModule = ctypes.c_void_p
WGPUComputePipeline = ctypes.c_void_p
WGPUBindGroupLayout = ctypes.c_void_p
WGPUPipelineLayout = ctypes.c_void_p
WGPUSurface = ctypes.c_void_p
WGPUBool = ctypes.c_uint32
WGPUFlags = ctypes.c_uint64
SIZE_T = ctypes.c_size_t


class WGPUSType:
    ShaderSourceWGSL = 0x00000002
    DawnTogglesDescriptor = 0x0005000A
    AdapterPropertiesSubgroupMatrixConfigs = 0x0005003B


class WGPUCallbackMode:
    WaitAnyOnly = 0x00000001
    AllowProcessEvents = 0x00000002


class WGPURequestAdapterStatus:
    Success = 0x00000001


class WGPUCreatePipelineAsyncStatus:
    Success = 0x00000001


class WGPUBackendType:
    D3D11 = 0x00000003
    D3D12 = 0x00000004
    Metal = 0x00000005
    Vulkan = 0x00000006


class WGPUFeatureLevel:
    Core = 0x00000002


class WGPUPowerPreference:
    HighPerformance = 0x00000002


class WGPUBufferBindingType:
    Storage = 0x00000003


SHADER_STAGE_COMPUTE = 0x0004


class WGPUStringView(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_char_p),
        ("length", SIZE_T),
    ]

    @staticmethod
    def from_str(value: str):
        encoded = value.encode("utf-8")
        return WGPUStringView(encoded, len(encoded))


class WGPUChainedStruct(ctypes.Structure):
    pass


WGPUChainedStruct._fields_ = [
    ("next", ctypes.POINTER(WGPUChainedStruct)),
    ("sType", ctypes.c_uint32),
]


class WGPUDawnTogglesDescriptor(ctypes.Structure):
    _fields_ = [
        ("chain", WGPUChainedStruct),
        ("enabledToggleCount", SIZE_T),
        ("enabledToggles", ctypes.POINTER(ctypes.c_char_p)),
        ("disabledToggleCount", SIZE_T),
        ("disabledToggles", ctypes.POINTER(ctypes.c_char_p)),
    ]


class WGPUFuture(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint64)]


class WGPUFutureWaitInfo(ctypes.Structure):
    _fields_ = [
        ("future", WGPUFuture),
        ("completed", WGPUBool),
    ]


RequestAdapterCallback = ctypes.CFUNCTYPE(
    None, ctypes.c_uint32, WGPUAdapter, WGPUStringView,
    ctypes.c_void_p, ctypes.c_void_p,
)


UncapturedErrorCallback = ctypes.CFUNCTYPE(
    None, ctypes.c_void_p, ctypes.c_uint32, WGPUStringView,
    ctypes.c_void_p, ctypes.c_void_p,
)


CreateComputePipelineAsyncCallback = ctypes.CFUNCTYPE(
    None, ctypes.c_uint32, WGPUComputePipeline, WGPUStringView,
    ctypes.c_void_p, ctypes.c_void_p,
)


class WGPURequestAdapterCallbackInfo(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("mode", ctypes.c_uint32),
        ("callback", RequestAdapterCallback),
        ("userdata1", ctypes.c_void_p),
        ("userdata2", ctypes.c_void_p),
    ]


class WGPUCreateComputePipelineAsyncCallbackInfo(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("mode", ctypes.c_uint32),
        ("callback", CreateComputePipelineAsyncCallback),
        ("userdata1", ctypes.c_void_p),
        ("userdata2", ctypes.c_void_p),
    ]


class WGPUInstanceDescriptor(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("requiredFeatureCount", SIZE_T),
        ("requiredFeatures", ctypes.c_void_p),
        ("requiredLimits", ctypes.c_void_p),
    ]


class WGPURequestAdapterOptions(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("featureLevel", ctypes.c_uint32),
        ("powerPreference", ctypes.c_uint32),
        ("forceFallbackAdapter", WGPUBool),
        ("backendType", ctypes.c_uint32),
        ("compatibleSurface", WGPUSurface),
    ]


class WGPUAdapterInfo(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("vendor", WGPUStringView),
        ("architecture", WGPUStringView),
        ("device", WGPUStringView),
        ("description", WGPUStringView),
        ("backendType", ctypes.c_uint32),
        ("adapterType", ctypes.c_uint32),
        ("vendorID", ctypes.c_uint32),
        ("deviceID", ctypes.c_uint32),
    ]


class WGPUSubgroupMatrixComponentType:
    F32 = 0x00000001
    F16 = 0x00000002
    U32 = 0x00000003
    I32 = 0x00000004
    U8 = 0x00000005
    I8 = 0x00000006


class WGPUSubgroupMatrixConfig(ctypes.Structure):
    _fields_ = [
        ("componentType", ctypes.c_uint32),
        ("resultComponentType", ctypes.c_uint32),
        ("M", ctypes.c_uint32),
        ("N", ctypes.c_uint32),
        ("K", ctypes.c_uint32),
    ]


class WGPUAdapterPropertiesSubgroupMatrixConfigs(ctypes.Structure):
    _fields_ = [
        ("chain", WGPUChainedStruct),
        ("configCount", SIZE_T),
        ("configs", ctypes.POINTER(WGPUSubgroupMatrixConfig)),
    ]


class WGPUQueueDescriptor(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("label", WGPUStringView),
    ]


class WGPUDeviceLostCallbackInfo(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("mode", ctypes.c_uint32),
        ("callback", ctypes.c_void_p),
        ("userdata1", ctypes.c_void_p),
        ("userdata2", ctypes.c_void_p),
    ]


class WGPUUncapturedErrorCallbackInfo(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("callback", ctypes.c_void_p),
        ("userdata1", ctypes.c_void_p),
        ("userdata2", ctypes.c_void_p),
    ]


class WGPULimits(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("maxTextureDimension1D", ctypes.c_uint32),
        ("maxTextureDimension2D", ctypes.c_uint32),
        ("maxTextureDimension3D", ctypes.c_uint32),
        ("maxTextureArrayLayers", ctypes.c_uint32),
        ("maxBindGroups", ctypes.c_uint32),
        ("maxBindGroupsPlusVertexBuffers", ctypes.c_uint32),
        ("maxBindingsPerBindGroup", ctypes.c_uint32),
        ("maxDynamicUniformBuffersPerPipelineLayout", ctypes.c_uint32),
        ("maxDynamicStorageBuffersPerPipelineLayout", ctypes.c_uint32),
        ("maxSampledTexturesPerShaderStage", ctypes.c_uint32),
        ("maxSamplersPerShaderStage", ctypes.c_uint32),
        ("maxStorageBuffersPerShaderStage", ctypes.c_uint32),
        ("maxStorageTexturesPerShaderStage", ctypes.c_uint32),
        ("maxUniformBuffersPerShaderStage", ctypes.c_uint32),
        ("maxUniformBufferBindingSize", ctypes.c_uint64),
        ("maxStorageBufferBindingSize", ctypes.c_uint64),
        ("minUniformBufferOffsetAlignment", ctypes.c_uint32),
        ("minStorageBufferOffsetAlignment", ctypes.c_uint32),
        ("maxVertexBuffers", ctypes.c_uint32),
        ("maxBufferSize", ctypes.c_uint64),
        ("maxVertexAttributes", ctypes.c_uint32),
        ("maxVertexBufferArrayStride", ctypes.c_uint32),
        ("maxInterStageShaderVariables", ctypes.c_uint32),
        ("maxColorAttachments", ctypes.c_uint32),
        ("maxColorAttachmentBytesPerSample", ctypes.c_uint32),
        ("maxComputeWorkgroupStorageSize", ctypes.c_uint32),
        ("maxComputeInvocationsPerWorkgroup", ctypes.c_uint32),
        ("maxComputeWorkgroupSizeX", ctypes.c_uint32),
        ("maxComputeWorkgroupSizeY", ctypes.c_uint32),
        ("maxComputeWorkgroupSizeZ", ctypes.c_uint32),
        ("maxComputeWorkgroupsPerDimension", ctypes.c_uint32),
        ("maxImmediateSize", ctypes.c_uint32),
    ]


class WGPUDeviceDescriptor(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("label", WGPUStringView),
        ("requiredFeatureCount", SIZE_T),
        ("requiredFeatures", ctypes.c_void_p),
        ("requiredLimits", ctypes.POINTER(WGPULimits)),
        ("defaultQueue", WGPUQueueDescriptor),
        ("deviceLostCallbackInfo", WGPUDeviceLostCallbackInfo),
        ("uncapturedErrorCallbackInfo", WGPUUncapturedErrorCallbackInfo),
    ]


class WGPUShaderModuleDescriptor(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("label", WGPUStringView),
    ]


class WGPUShaderSourceWGSL(ctypes.Structure):
    _fields_ = [
        ("chain", WGPUChainedStruct),
        ("code", WGPUStringView),
    ]


class WGPUBufferBindingLayout(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("type", ctypes.c_uint32),
        ("hasDynamicOffset", WGPUBool),
        ("minBindingSize", ctypes.c_uint64),
    ]


class WGPUSamplerBindingLayout(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("type", ctypes.c_uint32),
    ]


class WGPUTextureBindingLayout(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("sampleType", ctypes.c_uint32),
        ("viewDimension", ctypes.c_uint32),
        ("multisampled", WGPUBool),
    ]


class WGPUStorageTextureBindingLayout(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("access", ctypes.c_uint32),
        ("format", ctypes.c_uint32),
        ("viewDimension", ctypes.c_uint32),
    ]


class WGPUBindGroupLayoutEntry(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("binding", ctypes.c_uint32),
        ("visibility", WGPUFlags),
        ("bindingArraySize", ctypes.c_uint32),
        ("buffer", WGPUBufferBindingLayout),
        ("sampler", WGPUSamplerBindingLayout),
        ("texture", WGPUTextureBindingLayout),
        ("storageTexture", WGPUStorageTextureBindingLayout),
    ]


class WGPUBindGroupLayoutDescriptor(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("label", WGPUStringView),
        ("entryCount", SIZE_T),
        ("entries", ctypes.POINTER(WGPUBindGroupLayoutEntry)),
    ]


class WGPUPipelineLayoutDescriptor(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("label", WGPUStringView),
        ("bindGroupLayoutCount", SIZE_T),
        ("bindGroupLayouts", ctypes.POINTER(WGPUBindGroupLayout)),
        ("immediateSize", ctypes.c_uint32),
    ]


class WGPUComputeState(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("module", WGPUShaderModule),
        ("entryPoint", WGPUStringView),
        ("constantCount", SIZE_T),
        ("constants", ctypes.c_void_p),
    ]


class WGPUComputePipelineDescriptor(ctypes.Structure):
    _fields_ = [
        ("nextInChain", ctypes.POINTER(WGPUChainedStruct)),
        ("label", WGPUStringView),
        ("layout", WGPUPipelineLayout),
        ("compute", WGPUComputeState),
    ]


def _find_dawn_library(repo_root: str) -> str:
    candidates = [
        os.environ.get('DAWN_PATH'),
        os.path.join(repo_root, 'third_party', 'triton-windows', 'third_party', 'webgpu', 'dawn', 'build', 'webgpu_dawn.dll'),
        os.path.join(repo_root, 'third_party', 'triton-windows', 'third_party', 'webgpu', 'dawn', 'build', 'libwebgpu_dawn.so'),
        os.path.join(repo_root, 'third_party', 'triton-windows', 'third_party', 'webgpu', 'dawn', 'build', 'libwebgpu_dawn.dylib'),
        ctypes.util.find_library('webgpu_dawn'),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
        if candidate and candidate in {'webgpu_dawn', 'libwebgpu_dawn.so', 'libwebgpu_dawn.dylib'}:
            return candidate
    raise RuntimeError('Unable to locate Dawn library. Set DAWN_PATH if needed.')


def _setup_prototypes(lib):
    lib.wgpuCreateInstance.argtypes = [ctypes.POINTER(WGPUInstanceDescriptor)]
    lib.wgpuCreateInstance.restype = WGPUInstance

    lib.wgpuInstanceRequestAdapter.argtypes = [
        WGPUInstance,
        ctypes.POINTER(WGPURequestAdapterOptions),
        WGPURequestAdapterCallbackInfo,
    ]
    lib.wgpuInstanceRequestAdapter.restype = WGPUFuture

    lib.wgpuInstanceWaitAny.argtypes = [
        WGPUInstance,
        SIZE_T,
        ctypes.POINTER(WGPUFutureWaitInfo),
        ctypes.c_uint64,
    ]
    lib.wgpuInstanceWaitAny.restype = ctypes.c_uint32

    lib.wgpuInstanceProcessEvents.argtypes = [WGPUInstance]
    lib.wgpuInstanceProcessEvents.restype = None

    lib.wgpuAdapterGetInfo.argtypes = [WGPUAdapter, ctypes.POINTER(WGPUAdapterInfo)]
    lib.wgpuAdapterGetInfo.restype = ctypes.c_uint32

    lib.wgpuAdapterGetLimits.argtypes = [WGPUAdapter, ctypes.POINTER(WGPULimits)]
    lib.wgpuAdapterGetLimits.restype = ctypes.c_uint32

    lib.wgpuAdapterHasFeature.argtypes = [WGPUAdapter, ctypes.c_uint32]
    lib.wgpuAdapterHasFeature.restype = ctypes.c_uint32

    lib.wgpuAdapterCreateDevice.argtypes = [WGPUAdapter, ctypes.POINTER(WGPUDeviceDescriptor)]
    lib.wgpuAdapterCreateDevice.restype = WGPUDevice

    lib.wgpuDeviceCreateShaderModule.argtypes = [WGPUDevice, ctypes.POINTER(WGPUShaderModuleDescriptor)]
    lib.wgpuDeviceCreateShaderModule.restype = WGPUShaderModule

    lib.wgpuDeviceCreateBindGroupLayout.argtypes = [WGPUDevice, ctypes.POINTER(WGPUBindGroupLayoutDescriptor)]
    lib.wgpuDeviceCreateBindGroupLayout.restype = WGPUBindGroupLayout

    lib.wgpuDeviceCreatePipelineLayout.argtypes = [WGPUDevice, ctypes.POINTER(WGPUPipelineLayoutDescriptor)]
    lib.wgpuDeviceCreatePipelineLayout.restype = WGPUPipelineLayout

    lib.wgpuDeviceCreateComputePipelineAsync.argtypes = [
        WGPUDevice,
        ctypes.POINTER(WGPUComputePipelineDescriptor),
        WGPUCreateComputePipelineAsyncCallbackInfo,
    ]
    lib.wgpuDeviceCreateComputePipelineAsync.restype = WGPUFuture


def _decode_string_view(view: WGPUStringView) -> str:
    if view.data and view.length > 0:
        return view.data[:view.length].decode('utf-8', errors='replace')
    return ''


def _backend_name(backend_type: int) -> str:
    names = {
        WGPUBackendType.D3D11: 'D3D11',
        WGPUBackendType.D3D12: 'D3D12',
        WGPUBackendType.Metal: 'Metal',
        WGPUBackendType.Vulkan: 'Vulkan',
    }
    return names.get(backend_type, f'Unknown({backend_type})')


def _create_instance(lib):
    desc = WGPUInstanceDescriptor()
    desc.nextInChain = None
    desc.requiredFeatureCount = 0
    desc.requiredFeatures = None
    desc.requiredLimits = None
    instance = lib.wgpuCreateInstance(ctypes.byref(desc))
    if not instance:
        raise RuntimeError('Failed to create Dawn instance')
    return instance


def _request_adapter(lib, instance):
    adapter_holder = {'adapter': None, 'error': None}

    @RequestAdapterCallback
    def on_adapter(status, adapter, message, userdata1, userdata2):
        if status == WGPURequestAdapterStatus.Success:
            adapter_holder['adapter'] = adapter
        else:
            adapter_holder['error'] = _decode_string_view(message)

    toggle_names = [b'allow_unsafe_apis', b'vulkan_enable_f16_on_nvidia']
    toggle_ptrs = (ctypes.c_char_p * len(toggle_names))(*toggle_names)
    toggles = WGPUDawnTogglesDescriptor()
    toggles.chain.next = None
    toggles.chain.sType = WGPUSType.DawnTogglesDescriptor
    toggles.enabledToggleCount = len(toggle_names)
    toggles.enabledToggles = toggle_ptrs
    toggles.disabledToggleCount = 0
    toggles.disabledToggles = None

    opts = WGPURequestAdapterOptions()
    opts.nextInChain = ctypes.cast(ctypes.pointer(toggles), ctypes.POINTER(WGPUChainedStruct))
    opts.featureLevel = WGPUFeatureLevel.Core
    opts.powerPreference = WGPUPowerPreference.HighPerformance
    opts.forceFallbackAdapter = 0
    opts.backendType = WGPUBackendType.Vulkan
    opts.compatibleSurface = None

    cb_info = WGPURequestAdapterCallbackInfo()
    cb_info.nextInChain = None
    cb_info.mode = WGPUCallbackMode.AllowProcessEvents
    cb_info.callback = on_adapter
    cb_info.userdata1 = None
    cb_info.userdata2 = None

    lib.wgpuInstanceRequestAdapter(instance, ctypes.byref(opts), cb_info)
    deadline = time.time() + 10.0
    while not adapter_holder['adapter'] and adapter_holder['error'] is None:
        lib.wgpuInstanceProcessEvents(instance)
        if time.time() > deadline:
            adapter_holder['error'] = 'timed out waiting for adapter callback'
            break
        time.sleep(0.01)

    if not adapter_holder['adapter']:
        raise RuntimeError(f"Failed to request adapter: {adapter_holder['error'] or 'unknown error'}")
    return adapter_holder['adapter']


def _get_adapter_info(lib, adapter):
    info = WGPUAdapterInfo()
    ctypes.memset(ctypes.byref(info), 0, ctypes.sizeof(WGPUAdapterInfo))
    lib.wgpuAdapterGetInfo(adapter, ctypes.byref(info))
    return {
        'vendor': _decode_string_view(info.vendor),
        'architecture': _decode_string_view(info.architecture),
        'device': _decode_string_view(info.device),
        'description': _decode_string_view(info.description),
        'backend': _backend_name(info.backendType),
        'vendorID': info.vendorID,
        'deviceID': info.deviceID,
    }


def _component_type_name(component_type: int) -> str:
    names = {
        WGPUSubgroupMatrixComponentType.F32: 'F32',
        WGPUSubgroupMatrixComponentType.F16: 'F16',
        WGPUSubgroupMatrixComponentType.U32: 'U32',
        WGPUSubgroupMatrixComponentType.I32: 'I32',
        WGPUSubgroupMatrixComponentType.U8: 'U8',
        WGPUSubgroupMatrixComponentType.I8: 'I8',
    }
    return names.get(component_type, f'Unknown({component_type})')


def _get_subgroup_matrix_configs(lib, adapter):
    info = WGPUAdapterInfo()
    ctypes.memset(ctypes.byref(info), 0, ctypes.sizeof(WGPUAdapterInfo))

    subgroup_matrix_configs = WGPUAdapterPropertiesSubgroupMatrixConfigs()
    ctypes.memset(ctypes.byref(subgroup_matrix_configs), 0, ctypes.sizeof(WGPUAdapterPropertiesSubgroupMatrixConfigs))
    subgroup_matrix_configs.chain.next = None
    subgroup_matrix_configs.chain.sType = WGPUSType.AdapterPropertiesSubgroupMatrixConfigs
    info.nextInChain = ctypes.cast(
        ctypes.pointer(subgroup_matrix_configs.chain), ctypes.POINTER(WGPUChainedStruct)
    )

    status = lib.wgpuAdapterGetInfo(adapter, ctypes.byref(info))
    if status != 1:
        raise RuntimeError(f'Failed to query subgroup matrix configs: status={status}')

    configs = []
    for i in range(int(subgroup_matrix_configs.configCount)):
        config = subgroup_matrix_configs.configs[i]
        configs.append({
            'M': config.M,
            'N': config.N,
            'K': config.K,
            'componentType': _component_type_name(config.componentType),
            'resultComponentType': _component_type_name(config.resultComponentType),
        })
    return configs


def _create_device(lib, adapter):
    FEATURE_SHADER_F16_IDS = [0x0000000B, 0x0000000A]
    FEATURE_SUBGROUPS_IDS = [0x00000012, 0x00000011]
    FEATURE_SUBGROUP_MATRIX_ID = 0x00050034

    requested_features = []
    for feature_ids in (FEATURE_SHADER_F16_IDS, FEATURE_SUBGROUPS_IDS):
        for feature_id in feature_ids:
            if lib.wgpuAdapterHasFeature(adapter, feature_id):
                requested_features.append(feature_id)
                break
    has_subgroup_matrix = bool(lib.wgpuAdapterHasFeature(adapter, FEATURE_SUBGROUP_MATRIX_ID))
    if has_subgroup_matrix:
        requested_features.append(FEATURE_SUBGROUP_MATRIX_ID)

    limits = WGPULimits()
    ctypes.memset(ctypes.byref(limits), 0, ctypes.sizeof(WGPULimits))
    lib.wgpuAdapterGetLimits(adapter, ctypes.byref(limits))
    required_limits = WGPULimits()
    ctypes.memmove(ctypes.byref(required_limits), ctypes.byref(limits), ctypes.sizeof(WGPULimits))
    required_limits.nextInChain = None

    @UncapturedErrorCallback
    def on_uncaptured_error(device, error_type, message, userdata1, userdata2):
        print('[DAWN ERROR]', _decode_string_view(message), flush=True)

    dev_desc = WGPUDeviceDescriptor()
    ctypes.memset(ctypes.byref(dev_desc), 0, ctypes.sizeof(WGPUDeviceDescriptor))
    dev_desc.nextInChain = None
    dev_desc.label = WGPUStringView.from_str('raw-repro')
    if requested_features:
        feature_array = (ctypes.c_uint32 * len(requested_features))(*requested_features)
        dev_desc.requiredFeatureCount = len(requested_features)
        dev_desc.requiredFeatures = ctypes.cast(feature_array, ctypes.c_void_p)
    else:
        feature_array = None
        dev_desc.requiredFeatureCount = 0
        dev_desc.requiredFeatures = None
    dev_desc.requiredLimits = ctypes.pointer(required_limits)
    dev_desc.defaultQueue = WGPUQueueDescriptor(None, WGPUStringView.from_str(''))
    dev_desc.deviceLostCallbackInfo = WGPUDeviceLostCallbackInfo()
    dev_desc.uncapturedErrorCallbackInfo.callback = ctypes.cast(on_uncaptured_error, ctypes.c_void_p)
    dev_desc.uncapturedErrorCallbackInfo.userdata1 = None
    dev_desc.uncapturedErrorCallbackInfo.userdata2 = None

    device = lib.wgpuAdapterCreateDevice(adapter, ctypes.byref(dev_desc))
    if not device:
        raise RuntimeError('Failed to create device')
    return device, has_subgroup_matrix, on_uncaptured_error, feature_array, required_limits


def _create_bind_group_layout(lib, device):
    entry = WGPUBindGroupLayoutEntry()
    ctypes.memset(ctypes.byref(entry), 0, ctypes.sizeof(WGPUBindGroupLayoutEntry))
    entry.binding = 0
    entry.visibility = SHADER_STAGE_COMPUTE
    entry.bindingArraySize = 0
    entry.buffer.type = WGPUBufferBindingType.Storage
    entry.buffer.hasDynamicOffset = 0
    entry.buffer.minBindingSize = 0

    entries = (WGPUBindGroupLayoutEntry * 1)(entry)
    desc = WGPUBindGroupLayoutDescriptor()
    desc.nextInChain = None
    desc.label = WGPUStringView.from_str('')
    desc.entryCount = 1
    desc.entries = entries
    return lib.wgpuDeviceCreateBindGroupLayout(device, ctypes.byref(desc))


def _create_pipeline_layout(lib, device, bind_group_layout):
    layouts = (WGPUBindGroupLayout * 1)(bind_group_layout)
    desc = WGPUPipelineLayoutDescriptor()
    desc.nextInChain = None
    desc.label = WGPUStringView.from_str('')
    desc.bindGroupLayoutCount = 1
    desc.bindGroupLayouts = layouts
    desc.immediateSize = 0
    return lib.wgpuDeviceCreatePipelineLayout(device, ctypes.byref(desc))


def _compile_pipeline(lib, instance, device, pipeline_layout, wgsl_code: str):
    shader_source = WGPUShaderSourceWGSL()
    shader_source.chain.next = None
    shader_source.chain.sType = WGPUSType.ShaderSourceWGSL
    wgsl_bytes = wgsl_code.encode('utf-8')
    shader_source.code = WGPUStringView(wgsl_bytes, len(wgsl_bytes))

    shader_desc = WGPUShaderModuleDescriptor()
    shader_desc.nextInChain = ctypes.cast(ctypes.pointer(shader_source.chain), ctypes.POINTER(WGPUChainedStruct))
    shader_desc.label = WGPUStringView.from_str('raw-repro-shader')
    shader_module = lib.wgpuDeviceCreateShaderModule(device, ctypes.byref(shader_desc))
    if not shader_module:
        raise RuntimeError('Failed to create shader module')

    cp_desc = WGPUComputePipelineDescriptor()
    cp_desc.nextInChain = None
    cp_desc.label = WGPUStringView.from_str('raw-repro-pipeline')
    cp_desc.layout = pipeline_layout
    cp_desc.compute.nextInChain = None
    cp_desc.compute.module = shader_module
    cp_desc.compute.entryPoint = WGPUStringView.from_str('main')
    cp_desc.compute.constantCount = 0
    cp_desc.compute.constants = None

    holder = {'pipeline': None, 'error': None}

    @CreateComputePipelineAsyncCallback
    def on_pipeline(status, pipeline, message, userdata1, userdata2):
        if status == WGPUCreatePipelineAsyncStatus.Success:
            holder['pipeline'] = pipeline
        else:
            holder['error'] = _decode_string_view(message)

    cb_info = WGPUCreateComputePipelineAsyncCallbackInfo()
    cb_info.nextInChain = None
    cb_info.mode = WGPUCallbackMode.AllowProcessEvents
    cb_info.callback = on_pipeline
    cb_info.userdata1 = None
    cb_info.userdata2 = None

    lib.wgpuDeviceCreateComputePipelineAsync(device, ctypes.byref(cp_desc), cb_info)
    deadline = time.time() + 10.0
    while not holder['pipeline'] and holder['error'] is None:
        lib.wgpuInstanceProcessEvents(instance)
        if time.time() > deadline:
            holder['error'] = 'timed out waiting for pipeline callback'
            break
        time.sleep(0.01)

    if not holder['pipeline']:
        raise RuntimeError(holder['error'] or 'unknown pipeline creation error')
    return holder['pipeline']


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dawn_lib = _find_dawn_library(repo_root)
    if sys.platform == 'win32':
        kernel32 = ctypes.windll.kernel32
        for dll in ('d3dcompiler_47.dll', 'dxgi.dll', 'vulkan-1.dll'):
            try:
                kernel32.LoadLibraryW(dll)
            except Exception:
                pass

    lib = ctypes.CDLL(dawn_lib)
    _setup_prototypes(lib)

    instance = _create_instance(lib)
    adapter = _request_adapter(lib, instance)
    adapter_info = _get_adapter_info(lib, adapter)
    subgroup_matrix_configs = _get_subgroup_matrix_configs(lib, adapter)
    device, has_subgroup_matrix, _error_cb, _features, _limits = _create_device(lib, adapter)
    bind_group_layout = _create_bind_group_layout(lib, device)
    pipeline_layout = _create_pipeline_layout(lib, device, bind_group_layout)

    print('Adapter:', adapter_info)
    print('ChromiumExperimentalSubgroupMatrix:', has_subgroup_matrix)
    print('Advertised subgroup matrix configs:')
    for config in subgroup_matrix_configs:
        print(' ', config)

    cases = [
        ('Feature-only shader', FEATURE_ONLY_SHADER),
        ('Type declaration shader', TYPE_DECL_SHADER),
        ('Load + MMA + Store shader', MMA_SHADER),
    ]
    for index, (name, shader) in enumerate(cases, start=1):
        print(f'\n[{index}/{len(cases)}] {name}')
        try:
            _compile_pipeline(lib, instance, device, pipeline_layout, shader)
            print('Compiled: True')
        except Exception as exc:
            print('Observed failure type:', type(exc).__name__)
            print('Observed failure:')
            print(str(exc))

    return 0


if __name__ == '__main__':
    raise SystemExit(main())