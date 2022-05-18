#include "decode.h"
#include "Trt.h"
#include "utils.h"
#include "math.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include "../plugin/YoloLayerPlugin/YoloLayerPlugin.hpp"
__device__ float Logist1(float data){ return 1./(1. + exp(-data)); };
__global__ void caldetection(const float* input,float* output,int noelements,int yolowidth,int yoloheight,
                             const float anchors[6],int classes,int outputElem)
{
//    printf("进入");
    int idx=threadIdx.x+blockDim.x*blockIdx.x;
    if(idx>noelements)return;
    int stride=yoloheight*yolowidth;
    int bnidx=idx/stride;
//    printf("input %f next is %f\n",input[0],input[1]);
    int curidx=idx-bnidx*stride;

//    printf("IDX is %d,bnidx is %d,curid is %d\n",idx,bnidx,curidx);
    const float* curinput=input+bnidx*(7)*stride*3;
    for(int k=0;k<3;k++)
    {
        int beginidx=(7*stride)*k+curidx;
        int objidx=beginidx+stride*4;
        float objprob=Logist1(curinput[objidx]);
//        ofstream out_txt;
//        out_txt.open("./22.txt",ios::app);
//
//        out_txt<<curidx<<" "<<beginidx<<" "<<curinput[objidx]<<" "<<objprob<<"\n";
//        printf("curidx is%d, begin id is%d, input is %f 得分是%f\n",curidx,beginidx,curinput[objidx],objprob);
        if(objprob <= 0.7)
            continue;
//        printf("走到这一步\n");
        int row = curidx / yolowidth;
        int cols = curidx % yolowidth;

        //classes
        int classId = -1;
        float maxProb = IGNORE_THRESH;
        for (int c = 0;c<2;++c){
            float cProb =  Logist1(curinput[beginidx + (5 + c) * stride]) * objprob;
            if(cProb > maxProb){
                maxProb = cProb;
                classId = c;
            }
        }
        if(classId >= 0) {
            float *curOutput = output + bnidx*outputElem;
            int resCount = (int)atomicAdd(curOutput,1);
            char* data = (char * )curOutput + sizeof(float) + resCount*sizeof(Detection);
            Detection* det =  (Detection*)(data);

            //Location
            det->bbox[0] = (cols + Logist1(curinput[beginidx]))/ yolowidth;
            det->bbox[1] = (row + Logist1(curinput[beginidx+stride]))/ yoloheight;
            det->bbox[2] = exp(curinput[beginidx+2*stride]) * anchors[2*k];
            det->bbox[3] = exp(curinput[beginidx+3*stride]) * anchors[2*k + 1];
//            printf("box value is%f ,%f,%f,%f",det->bbox[0],det->bbox[1],det->bbox[2],det->bbox[3]);
            float tem_cla=float(classId);
            det->classId = llround(double(tem_cla));
            det->prob = maxProb;
//            printf("det x is%f y is%f w is%f h is%f \n",det->bbox[0],det->bbox[1],det->bbox[2],det->bbox[3]);
//            printf("class id is %d,temp is %f score is %f\n",classId,tem_cla,maxProb);
        }
//        out_txt.close();
    }

}
cudaError_t decode_gpu(vector<float> input,YoloKernel yolo_kernel,vector<Detection>& output)
{
    float* temp_input;
    int input_num=input.size();
    CUDA_CHECK(cudaMalloc(&temp_input,input.size()*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(temp_input,&input[0],input.size()*sizeof(float),cudaMemcpyHostToDevice));
    float* output1;
    void* devAnchor;
    size_t AnchorLen = sizeof(float)* CHECK_COUNT*2;
    CUDA_CHECK(cudaMalloc(&devAnchor,AnchorLen));
    int outputElem = 1;
    outputElem+=yolo_kernel.width*yolo_kernel.height*3*sizeof(Detection)/sizeof(float);
    CUDA_CHECK(cudaMalloc(&output1,sizeof(float)*outputElem));

    int numelem=yolo_kernel.width*yolo_kernel.height;
    CUDA_CHECK(cudaMemcpyAsync(devAnchor,yolo_kernel.anchors,AnchorLen,cudaMemcpyHostToDevice));
    caldetection<<<(yolo_kernel.width*yolo_kernel.height+512-1)/512,512>>>
    (temp_input,output1,numelem,yolo_kernel.width,yolo_kernel.height,(float *)devAnchor,2,outputElem);

    cudaError_t cudaStatus;
    cudaFree(devAnchor);
    float* out_host{};
    CUDA_CHECK(cudaMallocHost(&out_host,sizeof(float)*outputElem));

    CUDA_CHECK(cudaMemcpy(out_host,output1,sizeof(float)*outputElem,cudaMemcpyDeviceToHost));
//    printf("第一个输出%f,第二个输出%f",out_host[0],out_host[1]);
    cudaFree(output1);
    for(int k=0;k<int(out_host[0]);k++)
    {
        Detection temp;
        temp.bbox[0]=out_host[6*k+1];
        temp.bbox[1]=out_host[6*k+2];
        temp.bbox[2]=out_host[6*k+3];
        temp.bbox[3]=out_host[6*k+4];
        temp.classId=out_host[6*k+5];
        temp.prob=out_host[6*k+6];
        output.push_back(temp);
//        printf("det x is %f,det y is %f,det w is %f,det h is %f,det cla is %d det scor is %f\n"
//                ,out_host[6*k+1],out_host[6*k+2],out_host[6*k+3],out_host[6*k+4],int(out_host[6*k+5]),out_host[6*k+6]);

    }
//
    cudaFreeHost(out_host);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA error " << cudaGetErrorString(cudaStatus) << " at " << __FILE__ << ":" << __LINE__ << std::endl;


    }



//    CUDA_CHECK(cudaFree(devAnchor));

    return cudaGetLastError();
}