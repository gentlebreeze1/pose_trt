
#ifndef OPENPOSE_HPP
#define OPENPOSE_HPP
#include <string>
#include <vector>
#include <NvInfer.h>
#include <NvCaffeParser.h>
#include "opencv2/opencv.hpp"
class Trt;
struct mybox{
    int xmin;
    int ymin;
    int xmax;
    int ymax;
    float prob;
    int cla;
};
class yolo{
public:
    /**
     * @prototxt: NOTE: set input height and width in prototxt,
     * @calibratorData: create an empty instance, not support int8 now.
     * @maxBatchSize: set to 1.
     */

    yolo(
            const std::string& saveEngine
    );
    yolo(
            const std::string& onnxModel,
            const std::string& saveEngine,
            const std::vector<std::string>& outputBlobName,
            const std::vector<std::vector<float>>& calibratorData,
            int maxBatchSize
    );

    ~yolo();

    /**
     * @inputData: 1 * 3 * 480 * 640, or your favorite size, make sure modify it in prototxt.
     * @result: output keypoint, (x1,y1,score1, x2,y2,score2 ... x25, y25, scrore25) for one person and so on.
     */
    std::vector<float> prepareImage(std::vector<cv::Mat> &vec_img);
    void DoInference(std::vector<cv::Mat>& inputData,std::vector<float> &f1,std::vector<float> &f2);
    int postProcess( std::vector<cv::Mat> &vec_Mat, float *output,  std::map<int,std::vector<mybox>>&result);
    //std::vector<float> prepareImage(std::vector<cv::Mat> &vec_img);
private:
    void MallocExtraMemory();
    Trt* mNet;
    int mBatchSize;
    // input's device memory
    void* mpInputGpu;
    float * mfloatinput;
    nvinfer1::DataType mInputDataType;
    nvinfer1::Dims3 mInputDims;
    int64_t mInputSize;
    void *cudaFrame;
    void* cmap_vector;
    nvinfer1::Dims3 mout1Dims;
    int64_t mout1Size;
    int img_w_{};
    int img_h_{};
    int img_step1_{};
    int w{};
    int h{};
    int c{};
    float ratio;




};

#endif
