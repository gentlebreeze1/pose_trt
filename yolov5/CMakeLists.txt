project(testyolo)
file(GLOB yolo_source
        ./*.cpp
        ./*.cu
        )
cuda_add_library(yolo SHARED ${yolo_source})
target_compile_options(yolo PUBLIC -std=c++11 -Wall -Wfloat-conversion)
set_target_properties(yolo PROPERTIES POSITION_INDEPENDENT_CODE ON)
file(GLOB yolo_source
        test/test_yolo.cpp
        )
add_executable(testyolo ${yolo_source})
target_link_libraries(testyolo ${PROJECT_SOURCE_DIR}/../lib/libyolo.so)
target_link_libraries(testyolo ${PROJECT_SOURCE_DIR}/../lib/libtinytrt.so)
target_compile_options(testyolo PUBLIC -std=c++11 -Wall -Wfloat-conversion)
target_link_libraries(testyolo ${PROJECT_SOURCE_DIR}/../lib/libnvinfer.so)
target_link_libraries(testyolo ${PROJECT_SOURCE_DIR}/../lib/libnvinfer_plugin.so)
target_link_libraries(testyolo ${PROJECT_SOURCE_DIR}/../lib/libnvparsers.so)
target_link_libraries(testyolo ${PROJECT_SOURCE_DIR}/../lib/libnvonnxparser.so)
target_link_libraries(testyolo ${PROJECT_SOURCE_DIR}/../lib/libnvcaffe_parser.so)
target_link_libraries(testyolo ${PROJECT_SOURCE_DIR}/../lib/libopencv_core.so )
target_link_libraries(testyolo ${PROJECT_SOURCE_DIR}/../lib/libopencv_highgui.so )
target_link_libraries(testyolo ${PROJECT_SOURCE_DIR}/../lib/libopencv_imgproc.so )
target_link_libraries(testyolo ${PROJECT_SOURCE_DIR}/../lib/libopencv_imgcodecs.so )

#target_link_libraries(testopenpose /disk2/my_trt/my_version/pose/lib/libopencv_highgui.so )
