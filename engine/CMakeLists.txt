cmake_minimum_required(VERSION 3.16)
project(backpack_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dawn paths
set(DAWN_SRC "${CMAKE_SOURCE_DIR}/../third_party/dawn" CACHE PATH "Dawn source")
set(DAWN_BUILD "${DAWN_SRC}/build" CACHE PATH "Dawn build (gen headers)")
set(DAWN_LIB "${CMAKE_SOURCE_DIR}/../third_party/triton-windows/third_party/webgpu/dawn/build"
    CACHE PATH "Dawn shared library")

add_executable(backpack_engine
    engine.cpp
    gpu_context.cpp
    model_runner.cpp
    gguf_loader.cpp
    tokenizer.cpp
    clock_calibration.cpp
    profile_html.cpp
)

target_include_directories(backpack_engine PRIVATE
    ${DAWN_SRC}/include
    ${DAWN_BUILD}/gen/include
    ${DAWN_SRC}/include/dawn/native
)

# Link Dawn — use generated import library on Windows
if(WIN32)
    target_link_libraries(backpack_engine PRIVATE
        ${CMAKE_BINARY_DIR}/webgpu_dawn.lib
        d3d12.lib dxgi.lib)
    add_custom_command(TARGET backpack_engine POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${DAWN_LIB}/webgpu_dawn.dll $<TARGET_FILE_DIR:backpack_engine>)
else()
    target_link_libraries(backpack_engine PRIVATE ${DAWN_LIB}/libwebgpu_dawn.so)
endif()
