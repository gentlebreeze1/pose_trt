
#ifndef OPENPOSE_HPP
#define OPENPOSE_HPP
#include <string>
#include <vector>
#include <NvInfer.h>
#include <NvCaffeParser.h>
#include "opencv2/opencv.hpp"
class Trt;
class openpose{
public:
    /**
     * @prototxt: NOTE: set input height and width in prototxt,
     * @calibratorData: create an empty instance, not support int8 now.
     * @maxBatchSize: set to 1.
     */

    openpose(
            const std::string& saveEngine
    );
   openpose(
            const std::string& onnxModel,
            const std::string& saveEngine,
            const std::vector<std::string>& outputBlobName,
            const std::vector<std::vector<float>>& calibratorData,
            int maxBatchSize
    );

    ~openpose();

    /**
     * @inputData: 1 * 3 * 480 * 640, or your favorite size, make sure modify it in prototxt.
     * @result: output keypoint, (x1,y1,score1, x2,y2,score2 ... x25, y25, scrore25) for one person and so on.
     */
    void DoInference(std::vector<cv::Mat>& inputData,std::vector<float> &f1,std::vector<float> &f2);

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
    void* paf_vector;
    nvinfer1::Dims3 mout2Dims;
    int64_t mout2Size;



};

#endif
