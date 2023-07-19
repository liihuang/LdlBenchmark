cmake_minimum_required(VERSION 3.18)

project(LdlBenchmark LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(CUDAToolkit REQUIRED)

add_library(cuda_target INTERFACE)

target_link_libraries(cuda_target INTERFACE CUDA::cudart)

include_directories("eigen-3.4.0")

add_executable(LdlBenchmark ldl_benchmark.cu)

set_property(TARGET LdlBenchmark PROPERTY CXX_STANDARD 17)
    
target_compile_options(LdlBenchmark PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --generate-line-info
        --use_fast_math
    >)

    
target_link_libraries(LdlBenchmark cuda_target )