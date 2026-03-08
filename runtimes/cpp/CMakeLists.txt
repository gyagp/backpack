cmake_minimum_required(VERSION 3.16)
project(backpack_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dawn paths (relative to runtimes/cpp/)
set(DAWN_SRC "${CMAKE_SOURCE_DIR}/../../third_party/dawn" CACHE PATH "Dawn source")
set(DAWN_BUILD "${DAWN_SRC}/build" CACHE PATH "Dawn build (gen headers)")
set(DAWN_LIB "${CMAKE_SOURCE_DIR}/../../third_party/triton-windows/third_party/webgpu/dawn/build"
    CACHE PATH "Dawn shared library")

add_executable(backpack_runtime
    engine.cpp
    gpu_context.cpp
    model_runner.cpp
    gguf_loader.cpp
    tokenizer.cpp
    clock_calibration.cpp
    profile_html.cpp
)

target_include_directories(backpack_runtime PRIVATE
    ${DAWN_SRC}/include
    ${DAWN_BUILD}/gen/include
    ${DAWN_SRC}/include/dawn/native
)

# Link Dawn
if(WIN32)
    # Find webgpu_dawn.lib from Dawn build
    set(DAWN_NATIVE_LIB "${DAWN_SRC}/build/src/dawn/native/webgpu_dawn.lib"
        CACHE PATH "Dawn import library")
    target_link_libraries(backpack_runtime PRIVATE
        ${DAWN_NATIVE_LIB}
        d3d12.lib dxgi.lib)
    add_custom_command(TARGET backpack_runtime POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${DAWN_LIB}/webgpu_dawn.dll $<TARGET_FILE_DIR:backpack_runtime>)
else()
    target_link_libraries(backpack_runtime PRIVATE ${DAWN_LIB}/libwebgpu_dawn.so)
endif()
