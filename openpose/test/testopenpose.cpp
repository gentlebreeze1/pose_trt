#include "openpose.h"
#include "opencv2/opencv.hpp"
#include <vector>
#include <string>
#include "time.h"
#include <fstream>
#include <iostream>
#include "getresult.h"
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
//    const std::string onnx_model =string(argv[1]);
//    const std::string onnx_path =onnx_model+".onnx";
//    const std::string engine_path=onnx_model+".engine";
//    int max_size= long (argv[2]);

    const std::string onnx_path ="/disk2/my_trt/my_version/pose/models/pose.onnx";
    const std::string engine_path="/disk2/my_trt/my_version/pose/models/pose.engine";
    const std::string pic_name="/disk2/my_trt/my_version/pose/images/test1.jpg";
    Mat img=cv::imread(pic_name);
    vector<cv::Mat> mat_vec;
    mat_vec.push_back(img);
    int max_size= 32;
    std::vector<std::string> outputBlobname{"output_0","output_1"};
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
        openpose* mypose = new openpose(onnx_path,
                                        engine_path,
                                        outputBlobname,
                                        calibratorData,
                                        max_size
        );
        vector<float> cpuCmapBuffer;
        vector<float> cpuPafBuffer;
        mypose->DoInference(mat_vec,cpuCmapBuffer,cpuPafBuffer);
    } else{
        vector<float> cpuCmapBuffer;
        vector<float> cpuPafBuffer;
        openpose* mypose = new openpose(engine_path);
        mypose->DoInference(mat_vec,cpuCmapBuffer,cpuPafBuffer);
        getresult myresult;
        std::map<int,vector<BOX>>result;
        myresult.detect(cpuCmapBuffer,cpuPafBuffer,img,result);
        cv::imwrite("/disk2/my_trt/my_version/pose/images/11.jpg",img);

    }



    return 0;

}
