
cmake_minimum_required(VERSION 3.0)
project(tinytrt)
set(CMAKE_CXX_FLAGS "-std=c++11")
# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib CACHE PATH "")
option(BUILD_PYTHON "compile python api" OFF)
option(BUILD_TEST "compile test" ON)
find_package(CUDA REQUIRED)
include(../cmake/CUDA_utils.cmake)
set(SM_VERSION "" CACHE STRING "Description")
if(SM_VERSION)
    set(CUDA_targeted_archs ${SM_VERSION})
    CUDA_get_gencode_args(CUDA_gencode_flags ${CUDA_targeted_archs})
else()
    # Discover what architectures does nvcc support
    CUDA_find_supported_arch_values(CUDA_supported_archs ${CUDA_known_archs})
    message(STATUS "CUDA supported archs: ${CUDA_supported_archs}")

    set(CUDA_TARGET_ARCHS_SORTED ${CUDA_TARGET_ARCHS})
    list(SORT CUDA_TARGET_ARCHS_SORTED)
    CUDA_find_supported_arch_values(CUDA_targeted_archs ${CUDA_TARGET_ARCHS_SORTED})
    message(STATUS "CUDA targeted archs: ${CUDA_targeted_archs}")
    if (NOT CUDA_targeted_archs)
        message(FATAL_ERROR "None of the provided CUDA architectures ({${CUDA_TARGET_ARCHS}}) is supported by nvcc, use one or more of: ${CUDA_supported_archs}")
    endif()
    CUDA_get_gencode_args(CUDA_gencode_flags ${CUDA_targeted_archs})
endif()

# Add ptx & bin flags for cuda
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${CUDA_gencode_flags}")

option(ENABLE_PRELU_FP16 "" OFF)
if(ENABLE_PRELU_FP16)
    add_definitions(-DFP16_PRELU)
endif()
include_directories(./spdlog)
include_directories(./pybind11/include)
include_directories(../include)
include_directories(./)
include_directories(./utils)
include_directories(./plugin)
link_libraries(/disk2/my_trt/my_version/pose/lib)
message(STATUS "
Generated gencode flags: ${CUDA_gencode_flags}
BUILD_PYTHON : ${BUILD_PYTHON}
BUILD_TEST : ${BUILD_TEST}
ENABLE_PRELU_FP16 : ${ENABLE_PRELU_FP16}
")
file(GLOB_RECURSE trt_source
        Trt.cpp
        Int8EntropyCalibrator.cpp
        plugin/*.cu
        plugin/*.cpp
        utils/*.cpp
        utils/*.cu
        )
cuda_add_library(tinytrt SHARED ${trt_source})
target_compile_options(tinytrt PUBLIC -std=c++11 -Wall -Wfloat-conversion)
set_target_properties(tinytrt PROPERTIES POSITION_INDEPENDENT_CODE ON)
