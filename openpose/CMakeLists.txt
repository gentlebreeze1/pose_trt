project(testopenpose)
include_directories(../mytrt)
file(GLOB openpose_source
        ./include/parse/*.cpp
        ./include/parse/utils/*.cpp
        ./*.cpp
        ./*.cu

        )
cuda_add_library(pose SHARED ${openpose_source})
target_compile_options(pose PUBLIC -std=c++11 -Wall -Wfloat-conversion)
set_target_properties(pose PROPERTIES POSITION_INDEPENDENT_CODE ON)
file(GLOB test_source
        test/testopenpose.cpp
        )
add_executable(testopenpose ${test_source})
target_link_libraries(testopenpose ${PROJECT_SOURCE_DIR}/../lib/libpose.so)
target_link_libraries(testopenpose ${PROJECT_SOURCE_DIR}/../lib/libtinytrt.so)
target_compile_options(testopenpose PUBLIC -std=c++11 -Wall -Wfloat-conversion)
target_link_libraries(testopenpose ${PROJECT_SOURCE_DIR}/../lib/libnvinfer.so)
target_link_libraries(testopenpose ${PROJECT_SOURCE_DIR}/../lib/libnvinfer_plugin.so)
target_link_libraries(testopenpose ${PROJECT_SOURCE_DIR}/../lib/libnvparsers.so)
target_link_libraries(testopenpose ${PROJECT_SOURCE_DIR}/../lib/libnvonnxparser.so)
target_link_libraries(testopenpose ${PROJECT_SOURCE_DIR}/../lib/libnvcaffe_parser.so)
target_link_libraries(testopenpose ${PROJECT_SOURCE_DIR}/../lib/libopencv_core.so )
target_link_libraries(testopenpose ${PROJECT_SOURCE_DIR}/../lib/libopencv_highgui.so )
target_link_libraries(testopenpose ${PROJECT_SOURCE_DIR}/../lib/libopencv_imgproc.so )
target_link_libraries(testopenpose ${PROJECT_SOURCE_DIR}/../lib/libopencv_imgcodecs.so )

#target_link_libraries(testopenpose /disk2/my_trt/my_version/pose/lib/libopencv_highgui.so )
