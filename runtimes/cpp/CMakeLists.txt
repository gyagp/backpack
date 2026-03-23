cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0091 NEW)  # MSVC runtime library selection via CMAKE_MSVC_RUNTIME_LIBRARY
project(backpack_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Use shared CRT (/MD) for all targets — required for passing C++ objects
# (std::string, std::vector) across the DLL boundary safely.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL" CACHE STRING "MSVC runtime")

# Dawn paths (relative to runtimes/cpp/)
set(DAWN_SRC "${CMAKE_SOURCE_DIR}/../../third_party/dawn" CACHE PATH "Dawn source")
set(DAWN_BUILD "${DAWN_SRC}/build" CACHE PATH "Dawn build (gen headers)")
set(DAWN_LIB "${CMAKE_SOURCE_DIR}/../../third_party/triton-windows/third_party/webgpu/dawn/build"
    CACHE PATH "Dawn shared library")

# ─── Internal object library (shared between DLL and internal CLI) ───────────

set(BACKPACK_CORE_SOURCES
    gpu_context.cpp
    model_runner.cpp
    gguf_loader.cpp
    onnx_loader.cpp
    tokenizer.cpp
    onnx_tokenizer.cpp
    json_parser.cpp
    clock_calibration.cpp
    profile_html.cpp
)

add_library(backpack_core OBJECT ${BACKPACK_CORE_SOURCES})
target_include_directories(backpack_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${DAWN_SRC}/include
    ${DAWN_BUILD}/gen/include
    ${DAWN_SRC}/include/dawn/native
)
set_target_properties(backpack_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ─── Public shared library (backpack.dll / libbackpack.so) ───────────────────

add_library(backpack SHARED backpack.cpp $<TARGET_OBJECTS:backpack_core>)
target_compile_definitions(backpack PRIVATE BACKPACK_EXPORTS)
target_include_directories(backpack PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${DAWN_SRC}/include
    ${DAWN_BUILD}/gen/include
    ${DAWN_SRC}/include/dawn/native
)

# Put DLL in the same output dir as executables
set_target_properties(backpack PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
)

# ─── Internal CLI (links object files directly for internal access) ──────────

add_executable(backpack_runtime runtime.cpp $<TARGET_OBJECTS:backpack_core>)
target_include_directories(backpack_runtime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${DAWN_SRC}/include
    ${DAWN_BUILD}/gen/include
    ${DAWN_SRC}/include/dawn/native
)

# ─── LLM application (links shared library — public API only) ───────────────

set(BACKPACK_LLM_APP "${CMAKE_SOURCE_DIR}/../../apps/cpp/llm/main.cpp")
if(EXISTS ${BACKPACK_LLM_APP})
    add_executable(backpack_llm ${BACKPACK_LLM_APP})
    target_link_libraries(backpack_llm PRIVATE backpack)
endif()

# ─── Link Dawn ───────────────────────────────────────────────────────────────

if(WIN32)
    set(DAWN_NATIVE_LIB "${DAWN_SRC}/build/src/dawn/native/webgpu_dawn.lib"
        CACHE PATH "Dawn import library")
    # Dawn needed by core objects (used in both DLL and CLI)
    target_link_libraries(backpack PRIVATE
        ${DAWN_NATIVE_LIB} d3d12.lib dxgi.lib)
    target_link_libraries(backpack_runtime PRIVATE
        ${DAWN_NATIVE_LIB} d3d12.lib dxgi.lib)
    # Copy DLLs next to executables
    foreach(_target backpack_runtime backpack_llm)
        if(TARGET ${_target})
            add_custom_command(TARGET ${_target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DAWN_LIB}/webgpu_dawn.dll
                ${DAWN_LIB}/dxcompiler.dll
                ${DAWN_LIB}/dxil.dll
                $<TARGET_FILE_DIR:${_target}>)
        endif()
    endforeach()
else()
    target_link_libraries(backpack PRIVATE ${DAWN_LIB}/libwebgpu_dawn.so)
    target_link_libraries(backpack_runtime PRIVATE ${DAWN_LIB}/libwebgpu_dawn.so)
endif()
