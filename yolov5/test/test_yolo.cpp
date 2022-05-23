#include "yolo.h"
#include "opencv2/opencv.hpp"
#include <vector>
#include <string>
#include "time.h"
#include <fstream>
#include <iostream>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
using namespace cv;
using namespace std;
float point_thresh=0.5;
using namespace std;
using namespace cv;
void DoNms(std::vector<mybox>& detections,int classes ,float nmsThresh)
{
    using namespace std;
    // auto t_start = chrono::high_resolution_clock::now();

    std::vector<std::vector<mybox>> resClass;
    resClass.resize(classes);

    for (const auto& item : detections)
        resClass[item.cla].push_back(item);

    auto iouCompute = [](float * lbox, float* rbox)
    {
        float interBox[] = {
                max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
                min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
                max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
                min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
        };

        if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
            return 0.0f;

        float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
        return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
    };

    std::vector<mybox> result;
    for (int i = 0;i<classes;++i)
    {
        auto& dets =resClass[i];
        if(dets.size() == 0)
            continue;

        sort(dets.begin(),dets.end(),[=](const mybox& left,const mybox& right){
            return left.prob > right.prob;
        });

        for (unsigned int m = 0;m < dets.size() ; ++m)
        {
            auto& item = dets[m];
            result.push_back(item);
            float t1[4];

            t1[0]=item.xmin+(item.xmax-item.xmin)/2.0;
            t1[1]=item.ymin+(item.ymax-item.ymin)/2.0;
            t1[2]=item.xmax-item.xmin;
            t1[3]=item.ymax-item.ymin;
            for(unsigned int n = m + 1;n < dets.size() ; ++n)
            {
                float t2[4];

                t2[0]=dets[n].xmin+(dets[n].xmax-dets[n].xmin)/2.0;
                t2[1]=dets[n].ymin+(dets[n].ymax-dets[n].ymin)/2.0;
                t2[2]=dets[n].xmax-dets[n].xmin;
                t2[3]=dets[n].ymax-dets[n].ymin;
                float t=iouCompute(t1,t2);
                if (iouCompute(t1,t2) > nmsThresh)
                {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }

    //swap(detections,result);
    detections = move(result);

    // auto t_end = chrono::high_resolution_clock::now();
    // float total = chrono::duration<float, milli>(t_end - t_start).count();
    // cout << "Time taken for nms is " << total << " ms." << endl;
}
int main(int argc, char** argv) {
    const std::string onnx_path ="/disk2/my_trt/my_version/pose/models/sim.onnx";
    const std::string engine_path="/disk2/my_trt/my_version/pose/models/sim.engine";
    const std::string pic_name="/disk2/my_trt/my_version/pose/images/3391.jpg";
    Mat img=cv::imread(pic_name);
    vector<cv::Mat> mat_vec;
    mat_vec.push_back(img);
    int max_size= 32;
    std::vector<std::string> outputBlobname{"output"};
    std::vector<std::vector<float>> calibratorData;
    calibratorData.resize(3);
    for(size_t i = 0;i<calibratorData.size();i++) {
        calibratorData[i].resize(3*224*224);
        for(size_t j=0;j<calibratorData[i].size();j++) {
            calibratorData[i][j] = 0.05;
        }
    }
    ifstream file(engine_path.c_str(),std::ifstream::binary);
    if(!file.is_open())
    {
        yolo* mypose = new yolo(onnx_path,
                                    engine_path,
                                    outputBlobname,
                                    calibratorData,
                                    max_size
        );
        vector<float> cpuCmapBuffer;
        vector<float> cpuPafBuffer;
       // mypose->DoInference(mat_vec,cpuCmapBuffer,cpuPafBuffer);
    } else{
        vector<float> cpuCmapBuffer;
        vector<float> cpuPafBuffer;
        yolo* mypose = new yolo(engine_path);
        mypose->DoInference(mat_vec,cpuCmapBuffer,cpuPafBuffer);
        map<int,vector<mybox> >result;
        mypose->postProcess(mat_vec,&cpuCmapBuffer[0],result);
        for (int i=0;i<mat_vec.size();i++)
        {
            vector<mybox> tem=result[i];
            Mat tem_max=mat_vec[i];
            DoNms(tem,4,0.5);
            for (int j=0;j<tem.size();j++)
            {
                cv::rectangle(tem_max, cv::Point(tem[j].xmin, tem[j].ymin), cv::Point(tem[j].xmax, tem[j].ymax), cv::Scalar(0, 255, 0), 2);
            }
            string pic_name="/disk2/my_trt/my_version/pose/images/yolo_"+to_string(i)+".jpg";
            cv::imwrite(pic_name,tem_max);
        }


//        for(int i=0;i<mat_vec.size();i++)
//        {
//            mybox tem_box=result[i];
//            Mat temp_vec=mat_vec[i];
//
//        }



    }



    return 0;

}
