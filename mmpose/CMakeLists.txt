project(testmmpose)
file(GLOB mmpose_source
        ./*.cpp
        ./*.cu
        )
cuda_add_library(mmpose SHARED ${mmpose_source})
target_compile_options(mmpose PUBLIC -std=c++11 -Wall -Wfloat-conversion)
set_target_properties(mmpose PROPERTIES POSITION_INDEPENDENT_CODE ON)
file(GLOB test_source
        test/demo_mmpose.cpp
        )
add_executable(testmmpose ${test_source})
target_link_libraries(testmmpose ${PROJECT_SOURCE_DIR}/../lib/libmmpose.so)
target_link_libraries(testmmpose ${PROJECT_SOURCE_DIR}/../lib/libtinytrt.so)
target_compile_options(testmmpose PUBLIC -std=c++11 -Wall -Wfloat-conversion)
target_link_libraries(testmmpose ${PROJECT_SOURCE_DIR}/../lib/libnvinfer.so)
target_link_libraries(testmmpose ${PROJECT_SOURCE_DIR}/../lib/libnvinfer_plugin.so)
target_link_libraries(testmmpose ${PROJECT_SOURCE_DIR}/../lib/libnvparsers.so)
target_link_libraries(testmmpose ${PROJECT_SOURCE_DIR}/../lib/libnvonnxparser.so)
target_link_libraries(testmmpose ${PROJECT_SOURCE_DIR}/../lib/libnvcaffe_parser.so)
target_link_libraries(testmmpose ${PROJECT_SOURCE_DIR}/../lib/libopencv_core.so )
target_link_libraries(testmmpose ${PROJECT_SOURCE_DIR}/../lib/libopencv_highgui.so )
target_link_libraries(testmmpose ${PROJECT_SOURCE_DIR}/../lib/libopencv_imgproc.so )
target_link_libraries(testmmpose ${PROJECT_SOURCE_DIR}/../lib/libopencv_imgcodecs.so )

#target_link_libraries(testopenpose /disk2/my_trt/my_version/pose/lib/libopencv_highgui.so )
