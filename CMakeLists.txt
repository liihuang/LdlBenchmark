cmake_minimum_required(VERSION 3.8)

project(LdlBenchmark)

enable_language(CUDA)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# find_package(Eigen3)
# if(EIGEN3_FOUND)
#   include_directories(${EIGEN3_INCLUDE_DIR})
#   message(STATUS "EIGEN3_INCLUDE_DIR = ${EIGEN3_INCLUDE_DIR}")
# else()
  include_directories("eigen-3.4.0")
# endif()

add_executable(LdlBenchmark ldl_benchmark.cu)

target_link_libraries(LdlBenchmark ${LINK_LIBRARIES})