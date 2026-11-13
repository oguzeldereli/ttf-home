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
# -march=native on a CPU with AVX-VNNI emits vpdpbusd; binutils < 2.36 cannot assemble it (lab: binutils 2.35).
# Check the toolchain end to end (compile + assemble); if it fails, keep native but without AVX-VNNI.
if (GGML_NATIVE OR NOT DEFINED GGML_NATIVE)
    include(CheckCSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "-march=native")
    check_c_source_compiles("
        #include <immintrin.h>
        int main(void) {
        #ifdef __AVXVNNI__
            __m256i a = _mm256_set1_epi32(1);
            a = _mm256_dpbusd_avx_epi32(a, a, a);
            return _mm256_extract_epi32(a, 0);
        #else
            return 0;
        #endif
        }" TTF_NATIVE_VNNI_ASSEMBLES)
    unset(CMAKE_REQUIRED_FLAGS)
    if (NOT TTF_NATIVE_VNNI_ASSEMBLES)
        message(STATUS "ttf: the assembler rejects AVX-VNNI under -march=native; building with -mno-avxvnni")
        add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:-mno-avxvnni>)
    endif()
endif()
add_subdirectory(ggml)
# the TTF ops round like torch: no fused multiply-add across separately rounded steps
set_source_files_properties(ggml/src/ggml-cpu/ttf-ops.cpp TARGET_DIRECTORY ggml-cpu PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

add_library(ttf_engine STATIC src/ttf_model.cpp src/ttf_session.cpp src/ttf_speak.cpp src/ttf_tokenizer.cpp)
target_include_directories(ttf_engine PUBLIC include)
target_link_libraries(ttf_engine PUBLIC ggml)
target_compile_options(ttf_engine PRIVATE -ffp-contract=off)
if (GGML_CUDA)
    target_compile_definitions(ttf_engine PUBLIC TTF_CUDA)
endif()

# the one program: any .ttf model (text or speech), benchmarks, tokenizer, per-graph checks
add_executable(ttf-home tools/ttf_home.cpp)
target_link_libraries(ttf-home PRIVATE ttf_engine)

add_executable(test-ttf-ops tests/test_ttf_ops.cpp)
target_link_libraries(test-ttf-ops PRIVATE ttf_engine)
