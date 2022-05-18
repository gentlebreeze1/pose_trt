//
// Created by zhangyong on 6/28/20.
//

#ifndef TINYTRT_DECODE_H
#define TINYTRT_DECODE_H
#include "Trt.h"
#include "utils.h"
#include "YoloLayerPlugin/YoloLayerPlugin.hpp"
using namespace std;
struct Bbox {
    int left, right, top, bottom;
    int clsId;
    float score;
};
//struct YoloKernel
//{
//    int width;
//    int height;
//    float anchors[6];
//};
cudaError_t decode_gpu(vector<float> input,YoloKernel yolo_kernel,vector<Detection>& output);
float my_decode(vector<float> intput,YoloKernel yolo_kernel,vector<Detection>& output);
#endif //TINYTRT_DECODE_H
