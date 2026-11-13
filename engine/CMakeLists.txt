cmake_minimum_required(VERSION 3.18)
project(ttf_engine C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# vendored ggml (our fork: TQ1_P, GGML_OP_TTF_QUANT / GGML_OP_TTF_MM); static, one binary
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
# ggml's own threadpool: fewer cycles per node barrier than OpenMP (decode 340M, 6 threads: 122 vs 99 tok/s)
set(GGML_OPENMP OFF CACHE BOOL "")
# CUDA graphs (ggml defaults them on only inside llama.cpp): one launch per decode step
set(GGML_CUDA_GRAPHS ON CACHE BOOL "")
add_subdirectory(ggml)
# the TTF ops round like torch: no fused multiply-add across separately rounded steps
set_source_files_properties(ggml/src/ggml-cpu/ttf-ops.cpp TARGET_DIRECTORY ggml-cpu PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

add_library(ttf_engine STATIC src/ttf_model.cpp src/ttf_session.cpp src/ttf_speak.cpp)
target_include_directories(ttf_engine PUBLIC include)
target_link_libraries(ttf_engine PUBLIC ggml)
target_compile_options(ttf_engine PRIVATE -ffp-contract=off)
if (GGML_CUDA)
    target_compile_definitions(ttf_engine PUBLIC TTF_CUDA)
endif()

add_executable(ttf-run tools/ttf_run.cpp)
target_link_libraries(ttf-run PRIVATE ttf_engine)

add_executable(ttf-speak tools/ttf_speak.cpp)
target_link_libraries(ttf-speak PRIVATE ttf_engine)

add_executable(test-ttf-ops tests/test_ttf_ops.cpp)
target_link_libraries(test-ttf-ops PRIVATE ttf_engine)
