//
// Created by zhangyong on 5/17/22.
//

#include "Trt.h"
#include "yolo.h"
#include "utils.h"

//bool Trt::BuildEngineWithOnnx(const std::string& onnxModel,
//                              const std::string& engineFile,
//                              const std::vector<std::string>& customOutput,
//                              const std::vector<std::vector<float>>& calibratorData,
//
//                              int maxBatchSize)
using namespace std;
using namespace cv;
vector<float > img_mean={0, 0, 0};
vector<float > img_std={0, 0, 0};
int num_key_points=17;



std::vector<float> yolo::prepareImage(std::vector<cv::Mat> &vec_img) {
    std::vector<float> result(mBatchSize * w * h * c);
    float *data = result.data();
    for (const cv::Mat &src_img : vec_img)
    {
        if (!src_img.data)
            continue;
        ratio = std::min(float(w) / float(src_img.cols), float(h) / float(src_img.rows));
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
          //  split_img[i] = (split_img[i] - img_mean[i]) / img_std[i];
            memcpy(data, split_img[i].data, channelLength * sizeof(float));
            data += channelLength;
        }
    }
    return result;
}
int yolo::postProcess( std::vector<cv::Mat> &vec_Mat, float *output,map<int,vector<mybox>>&result) {
    int p_size=vec_Mat.size();
     int a=0;
    for(auto & vec:vec_Mat)
    {
        float*  temp_out=output+a*227430;
        vector<mybox> temp_boxvc;
        for (int i=0;i<227430;)
        {
            if (temp_out[4]>0.6)
            {
                mybox temp_box;
                float xc=temp_out[0]/ratio;
                float yc=temp_out[1]/ratio;
                float w=temp_out[2]/ratio;
                float h=temp_out[3]/ratio;
                int xmin=xc-w/2.0;
                int ymin=yc-h/2.0;
                int xmax=xc+w/2.0;
                int ymax=yc+h/2.0;
                xmin=xmin>0 ?xmin:0;
                ymin=ymin>0? ymin:0;
                int cla=int(temp_out[5]);
                float score=temp_out[6+cla];
                temp_box.xmin=xmin;
                temp_box.ymin=ymin;
                temp_box.xmax=xmax;
                temp_box.ymax=ymax;
                temp_box.cla=cla;
                temp_box.prob=score;
                temp_boxvc.push_back(temp_box);
            }

            temp_out=temp_out+10;
            i=i+10;
        }
        result[a]=temp_boxvc;
        a=a+1;
    }




    return 0;
}
yolo::yolo(
        const std::string& saveEngine
)
{
    mNet = new Trt();
    mNet->DeserializeEngine(saveEngine);
    mNet->InitEngine();
    MallocExtraMemory();
}
yolo::yolo(
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

void yolo::DoInference(std::vector<cv::Mat>& inputData,vector<float> &f1,vector<float> &f2) {

    int index = 0;
    std::vector<cv::Mat> vec_Mat(mBatchSize);
    int batch_id=0;
    for (auto& src_img :inputData)
    {
       // cv::cvtColor(src_img, src_img, cv::COLOR_BGR2RGB);
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

void yolo::MallocExtraMemory() {
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
