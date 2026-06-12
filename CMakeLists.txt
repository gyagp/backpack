cmake_minimum_required(VERSION 3.24)
if(POLICY CMP0180)
  cmake_policy(SET CMP0180 OLD)
endif()
project(backpack LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

FetchContent_Declare(
  dawn
  GIT_REPOSITORY https://dawn.googlesource.com/dawn
  GIT_TAG        main
  GIT_SHALLOW    ON
  GIT_SUBMODULES ""
  GIT_PROGRESS   ON
)

set(DAWN_FETCH_DEPENDENCIES ON)
set(DAWN_BUILD_SAMPLES OFF)

if(WIN32)
  set(DAWN_ENABLE_BACKEND_D3D12 ON)
  set(DAWN_ENABLE_BACKEND_VULKAN ON)
endif()

FetchContent_MakeAvailable(dawn)

# --- ONNX proto (uses protobuf bundled with Dawn) ---
FetchContent_Declare(
  onnx
  GIT_REPOSITORY https://github.com/onnx/onnx.git
  GIT_TAG        v1.16.0
  GIT_SHALLOW    ON
  GIT_SUBMODULES ""
  SOURCE_SUBDIR  DO_NOT_BUILD
)

FetchContent_MakeAvailable(onnx)

set(ONNX_PROTO_FILE "${onnx_SOURCE_DIR}/onnx/onnx.proto")

add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/onnx/onnx.pb.h"
         "${CMAKE_CURRENT_BINARY_DIR}/onnx/onnx.pb.cc"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/onnx"
  COMMAND protobuf::protoc
    --proto_path="${onnx_SOURCE_DIR}/onnx"
    --cpp_out="${CMAKE_CURRENT_BINARY_DIR}/onnx"
    "${ONNX_PROTO_FILE}"
  DEPENDS "${ONNX_PROTO_FILE}" protobuf::protoc
  COMMENT "Generating ONNX protobuf sources"
)

add_library(onnx_proto STATIC
  "${CMAKE_CURRENT_BINARY_DIR}/onnx/onnx.pb.cc"
)
target_include_directories(onnx_proto PUBLIC "${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(onnx_proto PUBLIC protobuf::libprotobuf)

add_executable(backpack
  src/main.cpp
  src/gpu_context.cpp
)
target_link_libraries(backpack PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(backpack PRIVATE src)

add_executable(test_gpu_context_backend
  tests/test_gpu_context_backend.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_gpu_context_backend PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_gpu_context_backend PRIVATE src)

add_executable(test_dequant_q4_k_m
  tests/test_dequant_q4_k_m.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_dequant_q4_k_m PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_dequant_q4_k_m PRIVATE src)

add_executable(test_dequant_q8_0
  tests/test_dequant_q8_0.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_dequant_q8_0 PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_dequant_q8_0 PRIVATE src)

add_executable(test_gguf_parser
  tests/test_gguf_parser.cpp
)
target_include_directories(test_gguf_parser PRIVATE src)

add_executable(test_mmap_file
  tests/test_mmap_file.cpp
)
target_include_directories(test_mmap_file PRIVATE src)

add_executable(test_batched_matmul
  tests/test_batched_matmul.cpp
)
target_include_directories(test_batched_matmul PRIVATE src)

add_executable(bench_matmul
  tests/bench_matmul.cpp
)
target_include_directories(bench_matmul PRIVATE src)

add_executable(test_tokenizer
  tests/test_tokenizer.cpp
)
target_include_directories(test_tokenizer PRIVATE src)

add_executable(test_sampling
  tests/test_sampling.cpp
)
target_include_directories(test_sampling PRIVATE src)

add_executable(test_chat_template
  tests/test_chat_template.cpp
)
target_include_directories(test_chat_template PRIVATE src)

add_executable(test_model_config
  tests/test_model_config.cpp
)
target_include_directories(test_model_config PRIVATE src)

add_executable(test_model_arch
  tests/test_model_arch.cpp
)
target_include_directories(test_model_arch PRIVATE src)

add_executable(test_buffer_pool
  tests/test_buffer_pool.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_buffer_pool PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_buffer_pool PRIVATE src)

add_executable(test_tensor
  tests/test_tensor.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_tensor PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_tensor PRIVATE src)

add_executable(test_weight_loader
  tests/test_weight_loader.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_weight_loader PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_weight_loader PRIVATE src)

add_executable(test_tensor_standalone
  tests/test_tensor_standalone.cpp
)
target_include_directories(test_tensor_standalone PRIVATE src)

add_executable(test_dispatch
  tests/test_dispatch.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_dispatch PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_dispatch PRIVATE src)

add_executable(test_elementwise_shaders
  tests/test_elementwise_shaders.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_elementwise_shaders PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_elementwise_shaders PRIVATE src)

add_executable(bench_dequant
  tests/bench_dequant.cpp
  src/gpu_context.cpp
)
target_link_libraries(bench_dequant PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(bench_dequant PRIVATE src)

add_executable(bench_matmul_gpu
  tests/bench_matmul_gpu.cpp
  src/gpu_context.cpp
)
target_link_libraries(bench_matmul_gpu PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(bench_matmul_gpu PRIVATE src)

add_executable(test_matmul_q4km
  tests/test_matmul_q4km.cpp
)
target_include_directories(test_matmul_q4km PRIVATE src)

add_executable(test_matmul_tiled_f16_gpu
  tests/test_matmul_tiled_f16_gpu.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_matmul_tiled_f16_gpu PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_matmul_tiled_f16_gpu PRIVATE src)

add_executable(test_rmsnorm_gpu
  tests/test_rmsnorm_gpu.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_rmsnorm_gpu PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_rmsnorm_gpu PRIVATE src)

add_executable(test_silu_gpu
  tests/test_silu_gpu.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_silu_gpu PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_silu_gpu PRIVATE src)

add_executable(test_rope_gpu
  tests/test_rope_gpu.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_rope_gpu PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_rope_gpu PRIVATE src)

add_executable(test_softmax_gpu
  tests/test_softmax_gpu.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_softmax_gpu PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_softmax_gpu PRIVATE src)

add_executable(test_attention_gpu
  tests/test_attention_gpu.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_attention_gpu PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_attention_gpu PRIVATE src)

add_executable(test_kv_cache_gpu
  tests/test_kv_cache_gpu.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_kv_cache_gpu PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_kv_cache_gpu PRIVATE src)

add_executable(test_flash_attention_gpu
  tests/test_flash_attention_gpu.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_flash_attention_gpu PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_flash_attention_gpu PRIVATE src)

add_executable(test_batched_matmul_gpu
  tests/test_batched_matmul_gpu.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_batched_matmul_gpu PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_batched_matmul_gpu PRIVATE src)

add_executable(test_matmul_q4km_gpu
  tests/test_matmul_q4km_gpu.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_matmul_q4km_gpu PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_matmul_q4km_gpu PRIVATE src)

add_executable(test_ffn_gpu
  tests/test_ffn_gpu.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_ffn_gpu PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_ffn_gpu PRIVATE src)

add_executable(test_transformer_layer_gpu
  tests/test_transformer_layer_gpu.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_transformer_layer_gpu PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_transformer_layer_gpu PRIVATE src)

add_executable(bench_backpack
  src/bench_backpack.cpp
  src/gpu_context.cpp
)
target_link_libraries(bench_backpack PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(bench_backpack PRIVATE src)

add_executable(test_lm_head_gpu
  tests/test_lm_head_gpu.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_lm_head_gpu PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_lm_head_gpu PRIVATE src)

add_executable(test_onnx_runtime
  tests/test_onnx_runtime.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_onnx_runtime PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_onnx_runtime PRIVATE src)

add_executable(test_onnx_inference
  tests/test_onnx_inference.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_onnx_inference PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_onnx_inference PRIVATE src)

add_executable(test_gguf_inference_smoke
  tests/test_gguf_inference_smoke.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_gguf_inference_smoke PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_gguf_inference_smoke PRIVATE src)

add_executable(test_gguf_inference_smoke_vulkan
  tests/test_gguf_inference_smoke_vulkan.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_gguf_inference_smoke_vulkan PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_gguf_inference_smoke_vulkan PRIVATE src)

add_executable(test_gguf_inference_smoke_granite
  tests/test_gguf_inference_smoke_granite.cpp
  src/gpu_context.cpp
)
target_link_libraries(test_gguf_inference_smoke_granite PRIVATE
  dawn::dawn_native
  dawn::dawn_proc
  dawn::dawn_common
)
target_include_directories(test_gguf_inference_smoke_granite PRIVATE src)
