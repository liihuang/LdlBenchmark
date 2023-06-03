cmake_minimum_required(VERSION 3.10)

project(LdlBenchmark)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
  set(CUDA_SEPARABLE_COMPILATION ON)
  include_directories(${CUDA_INCLUDE_DIRS})
  set(LINK_LIBRARIES ${LINK_LIBRARIES} ${CUDA_LIBRARIES})
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--expt-extended-lambda;-rdc=true;--ptxas-options=-v")
else (CUDA_FOUND)
  message (SEND_ERROR "Unable to locate CUDA")
endif()

find_package(Eigen3 REQUIRED)
if(EIGEN3_FOUND)
  include_directories(${EIGEN3_INCLUDE_DIR})
  message(STATUS "EIGEN3_INCLUDE_DIR = ${EIGEN3_INCLUDE_DIR}")
else()
  include_directories("eigen-3.4.0")
endif()

# find_package(MKL CONFIG REQUIRED)
# if (MKL_ENV)
#   message("Found MKL.")
#   include_directories($<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
#   set(LINK_LIBRARIES ${LINK_LIBRARIES} $<LINK_ONLY:MKL::MKL>)
# endif()

# message(STATUS "CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")

cuda_add_executable(LdlBenchmark ldl_benchmark.cu)

target_link_libraries(LdlBenchmark ${LINK_LIBRARIES})