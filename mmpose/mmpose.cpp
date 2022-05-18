//
// Created by zhangyong on 5/17/22.
//

#include "Trt.h"
#include "mmpose.h"
#include "utils.h"

//bool Trt::BuildEngineWithOnnx(const std::string& onnxModel,
//                              const std::string& engineFile,
//                              const std::vector<std::string>& customOutput,
//                              const std::vector<std::vector<float>>& calibratorData,
//
//                              int maxBatchSize)
using namespace std;
using namespace cv;
vector<float > img_mean={0.485, 0.456, 0.406};
vector<float > img_std={0.229, 0.224, 0.225};
int num_key_points=17;
std::vector<float> mmpose::prepareImage(std::vector<cv::Mat> &vec_img) {
    std::vector<float> result(mBatchSize * w * h * c);
    float *data = result.data();
    for (const cv::Mat &src_img : vec_img)
    {
        if (!src_img.data)
            continue;
        float ratio = std::min(float(w) / float(src_img.cols), float(h) / float(src_img.rows));
        cv::Mat flt_img = cv::Mat::zeros(cv::Size(w, h), CV_8UC3);
        cv::Mat rsz_img;
        cv::resize(src_img, rsz_img, cv::Size(), ratio, ratio);
        rsz_img.copyTo(flt_img(cv::Rect(0, 0, rsz_img.cols, rsz_img.rows)));
        flt_img.convertTo(flt_img, CV_32FC3, 1.0 / 255);

        //HWC TO CHW
        std::vector<cv::Mat> split_img(c);
        cv::split(flt_img, split_img);

        int channelLength = w * h;
        for (int i = 0; i < c; ++i)
        {
            split_img[i] = (split_img[i] - img_mean[i]) / img_std[i];
            memcpy(data, split_img[i].data, channelLength * sizeof(float));
            data += channelLength;
        }
    }
    return result;
}
int mmpose::postProcess( std::vector<cv::Mat> &vec_Mat, float *output,  int &outSize,std::vector<std::vector<myKeyPoint> > &vec_key_points) {

    int feature_size = w * h / 16;
    int index = 0;
    for (cv::Mat &src_img : vec_Mat) {
        std::vector<myKeyPoint> key_points = std::vector<myKeyPoint>(num_key_points);
        float ratio = std::max(float(src_img.cols) / float(w), float(src_img.rows) / float(h));
        float *current_person = output + index * outSize;
        for (int number = 0; number < num_key_points; number++) {
            float *current_point = current_person + feature_size * number;
            auto max_pos = std::max_element(current_point, current_point + feature_size);
            key_points[number].prob = *max_pos;
            float x = (max_pos - current_point) % (w / 4) + (*(max_pos + 1) > *(max_pos - 1) ? 0.25 : -0.25);
            float y = (max_pos - current_point) / (w / 4) + (*(max_pos + w / 4) > *(max_pos - w / 4) ? 0.25 : -0.25);
            key_points[number].x = int(x * ratio * 4);
            key_points[number].y = int(y * ratio * 4);
            key_points[number].number = number;
        }
        vec_key_points.push_back(key_points);
        index++;
    }


    return 0;
}
mmpose::mmpose(
        const std::string& saveEngine
)
{
    mNet = new Trt();
    mNet->DeserializeEngine(saveEngine);
    mNet->InitEngine();
    MallocExtraMemory();
}
mmpose::mmpose(
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

void mmpose::DoInference(std::vector<cv::Mat>& inputData,vector<float> &f1,vector<float> &f2) {

    int index = 0;
    std::vector<cv::Mat> vec_Mat(mBatchSize);
    int batch_id=0;
    for (auto& src_img :inputData)
    {
        cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);
        vec_Mat[batch_id] = src_img.clone();
        batch_id++;
        index++;

    }
    if (batch_id == mBatchSize or index == inputData.size())
    {
        std::vector<float> net_output;
        net_output.resize(mout1Size);
        std::vector<float>curInput = prepareImage(vec_Mat);
        printf("%d",curInput.size());
        mNet->CopyFromHostToDevice(curInput, 0);
        mNet->Forward();
        mNet->CopyFromDeviceToHost(net_output,1);
        f1.insert(f1.begin(),net_output.begin(),net_output.end());
    }



}

void mmpose::MallocExtraMemory() {
    mBatchSize = mNet->GetMaxBatchSize();

    mpInputGpu = mNet->GetBindingPtr(0);
    mfloatinput=(float*) mpInputGpu;
    mInputDataType = mNet->GetBindingDataType(0);
    nvinfer1::Dims inputDims = mNet->GetBindingDims(0);
    mInputDims = nvinfer1::Dims3(inputDims.d[0],inputDims.d[1],inputDims.d[2]);
    mInputSize = mNet->GetBindingSize(0);
    h=inputDims.d[2];
    w=inputDims.d[3];
    c=inputDims.d[1];
    cmap_vector = mNet->GetBindingPtr(1);
    nvinfer1::Dims  cmapDims = mNet->GetBindingDims(1);
    mout1Dims = nvinfer1::Dims3(cmapDims.d[0],cmapDims.d[1],cmapDims.d[2]);
    mout1Size = mNet->GetBindingSize(1);



}
