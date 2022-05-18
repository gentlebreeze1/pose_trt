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
//    f1.resize(out1_size);
//    f2.resize(out2_size);
//    mNet->CopyFromDeviceToHost(f1,1);
//    mNet->CopyFromDeviceToHost(f2,2);
    printf("ok1");
    mNet->CopyFromDeviceToHost(net_output,1);
    mNet->CopyFromDeviceToHost(net_output2,2);
//    f1=net_output;
//    f2=net_output2;
//    f1.insert(f1.begin(),net_output.begin(),net_output.end());
//    f2.insert(f2.begin(),net_output2.begin(),net_output2.end());
    memcpy(&f1,net_output.data(),out1_size* sizeof(float));
    memcpy(&f2,net_output2.data(),out2_size* sizeof(float));

    printf("ok");
//    mNet->CopyFromHostToDevice(inputData, 0);
//    int numBlocks = (mInputSize/getElementSize(mInputDataType) + 512 - 1) / 512;
//    Normalize<<<numBlocks, 512 , 0>>>((float*)mpInputGpu);
//    mNet->Forward();
//    std::vector<float> net_output;
//    net_output.resize(78*60*80);
//    mNet->CopyFromDeviceToHost(net_output,1);
//    memcpy((void*)mpHeatMapCpu,(void*)(net_output.data()),mHeatMapSize);
//
//    if(mResizeScale > 1) {
//        int widthSouce = mHeatMapDims.d[2];
//        int heightSource = mHeatMapDims.d[1];
//        int widthTarget = mResizeMapDims.d[2];
//        int heightTarget = mResizeMapDims.d[1];
//        const dim3 threadsPerBlock{16, 16, 1};
//        const dim3 numBlocks{
//                op::getNumberCudaBlocks(widthTarget, threadsPerBlock.x),
//                op::getNumberCudaBlocks(heightTarget, threadsPerBlock.y),
//                op::getNumberCudaBlocks(mResizeMapDims.d[0], threadsPerBlock.z)};
//        op::resizeKernel<<<numBlocks, threadsPerBlock>>>((float*)mpResizeMapGpu,(float*)mpHeatMapGpu,widthSouce,heightSource,widthTarget,heightTarget);
//        CUDA_CHECK(cudaMemcpy(mpResizeMapCpu, mpResizeMapGpu,mResizeMapSize,cudaMemcpyDeviceToHost));
//    }
//
//    // pose nms
//    std::array<int,4> targetSize2{mBatchSize,mNumPeaks,mMaxPerson,mPeaksVector};
//    std::array<int,4> sourceSize2{mBatchSize,mResizeMapDims.d[0],mResizeMapDims.d[1],mResizeMapDims.d[2]};
//    op::Point<float> offset = op::Point<float>(0.5,0.5);
//    op::nmsGpu((float*)mpPeaksGpu, (int*)mpKernelGpu, (float*)mpResizeMapGpu, mThreshold, targetSize2, sourceSize2, offset);
//    CUDA_CHECK(cudaMemcpyAsync(mpPeaksCpu, mpPeaksGpu, mPeaksSize, cudaMemcpyDeviceToHost,0));
//
//    // bodypart connect
//    Array<float> poseKeypoints;
//    Array<float> poseScores;
//    op::Point<int> resizeMapSize = op::Point<int>(mResizeMapDims.d[2],mResizeMapDims.d[1]);
//    op::connectBodyPartsCpu(poseKeypoints, poseScores, mpResizeMapCpu, mpPeaksCpu, op::PoseModel::BODY_25, resizeMapSize, mMaxPerson, mInterMinAboveThreshold, mInterThreshold,
//                            mMinSubsetCnt, mMinSubsetScore, 1.f);
//
//    result.resize(poseKeypoints.getVolume());
//    // std::cout << "number of person: " << poseKeypoints.getVolume()/75 << std::endl;
//    for(int i = 0; i < poseKeypoints.getVolume(); i++) {
//        if((i+1)%3 == 0) {
//            result[i] = poseKeypoints[i];
//        } else {
//            result[i] = poseKeypoints[i] * (8/mResizeScale);
//        }
//
//    }

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
