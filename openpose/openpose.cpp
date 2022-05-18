//
// Created by zhangyong on 5/17/22.
//

#include "Trt.h"
#include "openpose.h"
#include "utils.h"
#include "resize.h"
//bool Trt::BuildEngineWithOnnx(const std::string& onnxModel,
//                              const std::string& engineFile,
//                              const std::vector<std::string>& customOutput,
//                              const std::vector<std::vector<float>>& calibratorData,
//
//                              int maxBatchSize)
using namespace std;
using namespace cv;

openpose::openpose(
        const std::string& saveEngine
        )
{
    mNet = new Trt();
    mNet->DeserializeEngine(saveEngine);
    mNet->InitEngine();
    MallocExtraMemory();
}
openpose::openpose(
                   const std::string& onnxModel,
                   const std::string& saveEngine,
                   const std::vector<std::string>& outputBlobName,
                   const std::vector<std::vector<float>>& calibratorData,
                   int maxBatchSize
                   )
                   {
    mNet = new Trt();

    mNet->BuildEngineWithOnnx(onnxModel,saveEngine,outputBlobName,calibratorData,maxBatchSize);
    MallocExtraMemory();


}

void openpose::DoInference(std::vector<cv::Mat>& inputData,vector<float> &f1,vector<float> &f2) {

    cudaFrame = safeCudaMalloc(1024 * 1024 * 3 * sizeof(uchar));
    size_t buff_size=c*w*h*sizeof(float);
    for (size_t i = 0; i < inputData.size(); ++i)
    {
        auto& mat = inputData[i];
        img_step1_=mat.step[0];
        img_h_=mat.rows;
        cudaMemcpy(cudaFrame, mat.data, img_step1_ * img_h_, cudaMemcpyHostToDevice);
        myresizeAndNorm(cudaFrame, mfloatinput+i*buff_size, mat.cols, mat.rows, w,h,nullptr);
    }
    mNet->Forward();
    std::vector<float> net_output;
    std::vector<float> net_output2;
    int out1_size=mBatchSize*18*56*56;
    int out2_size=mBatchSize*42*56*56;
    net_output.resize(out1_size);
    net_output2.resize(out2_size);
    f1.resize(out1_size);
    f2.resize(out2_size);
    mNet->CopyFromDeviceToHost(f1,1);
    mNet->CopyFromDeviceToHost(f2,2);
//    printf("ok1");
//    mNet->CopyFromDeviceToHost(net_output,1);
//    mNet->CopyFromDeviceToHost(net_output2,2);
////    f1=net_output;
//    f2=net_output2;
//    f1.insert(f1.begin(),net_output.begin(),net_output.end());
//    f2.insert(f2.begin(),net_output2.begin(),net_output2.end());
//    memcpy(&f1,net_output.data(),out1_size* sizeof(float));
//    memcpy(&f2,net_output2.data(),out2_size* sizeof(float));


}

void openpose::MallocExtraMemory() {
    mBatchSize = mNet->GetMaxBatchSize();

    mpInputGpu = mNet->GetBindingPtr(0);
    mfloatinput=(float*) mpInputGpu;
    mInputDataType = mNet->GetBindingDataType(0);
    nvinfer1::Dims inputDims = mNet->GetBindingDims(0);
    mInputDims = nvinfer1::Dims3(inputDims.d[0],inputDims.d[1],inputDims.d[2]);
    mInputSize = mNet->GetBindingSize(0);
    w=inputDims.d[2];
    h=inputDims.d[3];
    c=inputDims.d[1];
    cmap_vector = mNet->GetBindingPtr(1);
    nvinfer1::Dims  cmapDims = mNet->GetBindingDims(1);
    mout1Dims = nvinfer1::Dims3(cmapDims.d[0],cmapDims.d[1],cmapDims.d[2]);
    mout1Size = mNet->GetBindingSize(1);

    paf_vector = mNet->GetBindingPtr(2);
    nvinfer1::Dims pafDims = mNet->GetBindingDims(2);
    mout2Dims = nvinfer1::Dims3(pafDims.d[0],pafDims.d[1],pafDims.d[2]);
    mout2Size = mNet->GetBindingSize(2);

}
