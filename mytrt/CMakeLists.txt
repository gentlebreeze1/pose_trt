project(tinytrt)
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
