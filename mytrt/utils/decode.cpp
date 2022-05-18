//
// Created by zhangyong on 6/28/20.
//
#include "decode.h"

float my_Logist(float data){ return 1./(1. + exp(-data)); }

float my_decode(vector<float> intput,YoloKernel yolo_kernel,vector<Detection>& output)
{
    YoloKernel yolo=yolo_kernel;
    int stride=yolo.width*yolo.height;

    for(int i=0;i<yolo.width*yolo.height;i++)
    {
        for(int j=0;j<3;j++)
        {
            int begin_id=7*stride*j+i;
            int obj_id=begin_id+4*stride;
            float obj_prob=my_Logist(intput[obj_id]);
            if(obj_prob<0.7)
                continue;
            int class_id=-1;
            float max_prob=0.7;
            for(int k=0;k<2;k++)
            {
                float temp_prob=my_Logist(intput[begin_id+(5+k)*stride])*obj_prob;
                if(temp_prob>max_prob)
                {
                    class_id=k;
                    max_prob=temp_prob;
                }
                if(class_id>=0)
                {
                    Detection det;
                    int row=i/yolo.width;
                    int cols=i%yolo.height;
                    float a=my_Logist(intput[begin_id]);
                    float b=my_Logist(intput[begin_id+stride]);
                    det.bbox[0]=(cols+a)/yolo.width;
                    det.bbox[1]=(row+b)/yolo.height;
                    det.bbox[2]=exp(intput[begin_id+2*stride])*yolo.anchors[2*j];
                    det.bbox[3]=exp(intput[begin_id+3*stride])*yolo.anchors[2*j+1];
                    det.classId=class_id;
                    det.prob=max_prob;
                    output.emplace_back(det);
                }
            }
        }
    }

}
