cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0091 NEW)  # MSVC runtime library selection via CMAKE_MSVC_RUNTIME_LIBRARY

if(WIN32)
    set(_winsdk_bin "C:/Program Files (x86)/Windows Kits/10/bin")
    if(EXISTS "${_winsdk_bin}")
        file(GLOB _winsdk_versions LIST_DIRECTORIES true "${_winsdk_bin}/10.0.*")
        list(SORT _winsdk_versions COMPARE NATURAL ORDER DESCENDING)

        foreach(_winsdk_dir IN LISTS _winsdk_versions)
            if(NOT DEFINED CMAKE_MT OR CMAKE_MT MATCHES "NOTFOUND")
                if(EXISTS "${_winsdk_dir}/x64/mt.exe")
                    set(CMAKE_MT "${_winsdk_dir}/x64/mt.exe" CACHE FILEPATH "Manifest tool" FORCE)
                endif()
            endif()
            if(NOT DEFINED CMAKE_RC_COMPILER OR CMAKE_RC_COMPILER MATCHES "-NOTFOUND$" OR CMAKE_RC_COMPILER STREQUAL "rc")
                if(EXISTS "${_winsdk_dir}/x64/rc.exe")
                    set(CMAKE_RC_COMPILER "${_winsdk_dir}/x64/rc.exe" CACHE FILEPATH "Windows resource compiler" FORCE)
                endif()
            endif()
            if((DEFINED CMAKE_MT AND NOT CMAKE_MT MATCHES "NOTFOUND") AND
               (DEFINED CMAKE_RC_COMPILER AND NOT CMAKE_RC_COMPILER MATCHES "-NOTFOUND$" AND NOT CMAKE_RC_COMPILER STREQUAL "rc"))
                break()
            endif()
        endforeach()
    endif()
endif()

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
    graph_executor.cpp
    ops/elementwise.cpp
    ops/shape.cpp
    ops/matmul.cpp
    ops/norm.cpp
    ops/attention.cpp
    ops/conv.cpp
    ops/moe.cpp
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

# ─── Public library ──────────────────────────────────────────────────────────

add_library(backpack SHARED backpack.cpp $<TARGET_OBJECTS:backpack_core>)
target_compile_definitions(backpack PRIVATE BACKPACK_BUILDING)
set_target_properties(backpack PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
)
target_include_directories(backpack PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${DAWN_SRC}/include
    ${DAWN_BUILD}/gen/include
    ${DAWN_SRC}/include/dawn/native
)

# ─── Internal CLI (links object files directly for internal access) ──────────

add_executable(backpack_runtime runtime.cpp $<TARGET_OBJECTS:backpack_core>)
target_include_directories(backpack_runtime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${DAWN_SRC}/include
    ${DAWN_BUILD}/gen/include
    ${DAWN_SRC}/include/dawn/native
)

# ─── LLM application (uses runtime API + internal LLM logic) ─────────────────

set(BACKPACK_LLM_APP "${CMAKE_SOURCE_DIR}/../../apps/cpp/llm/main.cpp")
if(EXISTS ${BACKPACK_LLM_APP})
    add_executable(backpack_llm ${BACKPACK_LLM_APP} $<TARGET_OBJECTS:backpack_core>)
    target_link_libraries(backpack_llm PRIVATE backpack)
    target_include_directories(backpack_llm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${DAWN_SRC}/include
        ${DAWN_BUILD}/gen/include
        ${DAWN_SRC}/include/dawn/native
    )
endif()

# ─── Image generation application ────────────────────────────────────────────

set(BACKPACK_IMAGE_APP "${CMAKE_SOURCE_DIR}/../../apps/cpp/image/main.cpp")
if(EXISTS ${BACKPACK_IMAGE_APP})
    add_executable(backpack_image ${BACKPACK_IMAGE_APP} $<TARGET_OBJECTS:backpack_core>)
    target_link_libraries(backpack_image PRIVATE backpack)
    target_include_directories(backpack_image PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${DAWN_SRC}/include
        ${DAWN_BUILD}/gen/include
        ${DAWN_SRC}/include/dawn/native
    )
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
    if(TARGET backpack_llm)
        target_link_libraries(backpack_llm PRIVATE
            ${DAWN_NATIVE_LIB} d3d12.lib dxgi.lib)
    endif()
    if(TARGET backpack_image)
        target_link_libraries(backpack_image PRIVATE
            ${DAWN_NATIVE_LIB} d3d12.lib dxgi.lib)
    endif()
    # Copy DLLs next to executables
    foreach(_target backpack_runtime backpack_llm backpack_image)
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
    if(TARGET backpack_llm)
        target_link_libraries(backpack_llm PRIVATE ${DAWN_LIB}/libwebgpu_dawn.so)
    endif()
    if(TARGET backpack_image)
        target_link_libraries(backpack_image PRIVATE ${DAWN_LIB}/libwebgpu_dawn.so)
    endif()
endif()
