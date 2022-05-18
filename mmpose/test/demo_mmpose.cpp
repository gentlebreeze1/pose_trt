#include "mmpose.h"
#include "opencv2/opencv.hpp"
#include <vector>
#include <string>
#include "time.h"
#include <fstream>
#include <iostream>
using namespace cv;
using namespace std;
float point_thresh=0.5;
std::vector<std::vector<int> > skeleton={{15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11},
                                         {6, 12}, {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 2},
                                         {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}};
int main(int argc, char** argv) {
//    const std::string onnx_model =string(argv[1]);
//    const std::string onnx_path =onnx_model+".onnx";
//    const std::string engine_path=onnx_model+".engine";
//    int max_size= long (argv[2]);
    const std::string onnx_path ="/disk2/my_trt/my_version/pose/models/hrnet.onnx";
    const std::string engine_path="/disk2/my_trt/my_version/pose/models/hrnet.engine";
    const std::string pic_name="/disk2/my_trt/my_version/pose/images/00002.jpg";
    Mat img=cv::imread(pic_name);
    vector<cv::Mat> mat_vec;
    mat_vec.push_back(img);
    int max_size= 32;
    std::vector<std::string> outputBlobname{"2947"};
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
        mmpose* mypose = new mmpose(onnx_path,
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
        mmpose* mypose = new mmpose(engine_path);
        mypose->DoInference(mat_vec,cpuCmapBuffer,cpuPafBuffer);
        vector<vector<myKeyPoint> > key_points;
        int o_size=52224;
        mypose->postProcess(mat_vec,&cpuCmapBuffer[0],o_size,key_points);
        for (int i = 0; i < (int)mat_vec.size(); i++)
        {
            auto org_img = mat_vec[i];
            if (!org_img.data)
                continue;
            auto current_points = key_points[i];
            cv::cvtColor(org_img, org_img, cv::COLOR_BGR2RGB);
            for (const auto &bone : skeleton) {
                if (current_points[bone[0]].prob < point_thresh or current_points[bone[1]].prob < point_thresh)
                    continue;
                cv::Scalar color;
                if (bone[0] < 5 or bone[1] < 5)
                    color = cv::Scalar(0, 255, 0);
                else if (bone[0] > 12 or bone[1] > 12)
                    color = cv::Scalar(255, 0, 0);
                else if (bone[0] > 4 and bone[0] < 11 and bone[1] > 4 and bone[1] < 11)
                    color = cv::Scalar(0, 255, 255);
                else
                    color = cv::Scalar(255, 0, 255);
                cv::line(org_img, cv::Point(current_points[bone[0]].x, current_points[bone[0]].y),
                         cv::Point(current_points[bone[1]].x, current_points[bone[1]].y), color,
                         2);
            }
            for(const auto &point : current_points) {
                if (point.prob < point_thresh)
                    continue;
                cv::Scalar color;
                if (point.number < 5)
                    color = cv::Scalar(0, 255, 0);
                else if (point.number > 10)
                    color = cv::Scalar(255, 0, 0);
                else
                    color = cv::Scalar(0, 255, 255);
                cv::circle(org_img, cv::Point(point.x, point.y), 5, color, -1, cv::LINE_8, 0);
            }
            //int pos = vec_name[i].find_last_of(".");
            // std::string rst_name = vec_name[i].insert(pos, "_");
            // std::cout << rst_name << std::endl;
            cv::imwrite("/disk2/my_trt/my_version/pose/images/1.jpg", org_img);
        }

    }



    return 0;

}
