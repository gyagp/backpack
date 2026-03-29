cmake_minimum_required(VERSION 3.16)
project(backpack_llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the backpack library (built from runtime/)
# When building standalone, set BACKPACK_DIR to the runtime build directory.
set(BACKPACK_DIR "${CMAKE_SOURCE_DIR}/../../runtime" CACHE PATH
    "Path to runtime/ source (contains backpack.h)")

# Add the runtime as a subdirectory so we get the 'backpack' library target
add_subdirectory(${BACKPACK_DIR} ${CMAKE_BINARY_DIR}/backpack_runtime)

add_executable(backpack_llm main.cpp)
target_link_libraries(backpack_llm PRIVATE backpack)
