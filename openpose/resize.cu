
#include <device_launch_parameters.h>
#include <stdio.h>

__forceinline__ __device__ float3 get(uchar3* src, int x, int y, int w, int h) {
	if (x < 0 || x >= w || y < 0 || y >= h) return make_float3(0.5, 0.5, 0.5);
	uchar3 temp = src[y*w + x];
	return make_float3(float(temp.x) / 255., float(temp.y) / 255., float(temp.z) / 255.);
}
__global__ void resizeNormKernel(uchar3* src, float *dst, int dstW, int dstH, int srcW, int srcH,
	float scaleX, float scaleY, float shiftX, float shiftY) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int x = idx % dstW;
	const int y = idx / dstW;
	if (x >= dstW || y >= dstH)
		return;
	float w = (x - shiftX + 0.5) * scaleX - 0.5;
	float h = (y - shiftY + 0.5) * scaleY - 0.5;
	int h_low = (int)h;
	int w_low = (int)w;
	int h_high = h_low + 1;
	int w_high = w_low + 1;
	float lh = h - h_low;
	float lw = w - w_low;
	float hh = 1 - lh, hw = 1 - lw;
	float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
	float3 v1 = get(src, w_low, h_low, srcW, srcH);
	float3 v2 = get(src, w_high, h_low, srcW, srcH);
	float3 v3 = get(src, w_low, h_high, srcW, srcH);
	float3 v4 = get(src, w_high, h_high, srcW, srcH);
	int stride = dstW * dstH;
	dst[y*dstW + x] = w1 * v1.x + w2 * v2.x + w3 * v3.x + w4 * v4.x;
	dst[stride + y * dstW + x] = w1 * v1.y + w2 * v2.y + w3 * v3.y + w4 * v4.y;
	dst[stride * 2 + y * dstW + x] = w1 * v1.z + w2 * v2.z + w3 * v3.z + w4 * v4.z;
}
__global__ void myresizeNormKernel(uchar3* src, float *dst, int dstW, int dstH, int srcW, int srcH,
	float scaleX, float scaleY, float shiftX, float shiftY) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;//cuda线程的x索引.x对应的comuln.这里的x y是相对输出来说的,是指输出的宽高,不是输入的.
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= dstW || y >= dstH)
		return;
	float w = (x - shiftX + 0.5) * scaleX - 0.5;
	float h = (y - shiftY + 0.5) * scaleY - 0.5;
	int h_low = (int)h;
	int w_low = (int)w;
	int h_high = h_low + 1;
	int w_high = w_low + 1;
	float lh = h - h_low;
	float lw = w - w_low;
	float hh = 1 - lh, hw = 1 - lw;
	float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
	float3 v1 = get(src, w_low, h_low, srcW, srcH);
	float3 v2 = get(src, w_high, h_low, srcW, srcH);
	float3 v3 = get(src, w_low, h_high, srcW, srcH);
	float3 v4 = get(src, w_high, h_high, srcW, srcH);
	int stride = dstW * dstH;
	dst[y*dstW + x] = w1 * v1.x + w2 * v2.x + w3 * v3.x + w4 * v4.x;
	dst[stride + y * dstW + x] = w1 * v1.y + w2 * v2.y + w3 * v3.y + w4 * v4.y;
	dst[stride * 2 + y * dstW + x] = w1 * v1.z + w2 * v2.z + w3 * v3.z + w4 * v4.z;
}
int resizeAndNorm(void * p, float *d, int w, int h, int in_w, int in_h, cudaStream_t stream) {
	float scaleX = (w*1.0f / in_w);
	float scaleY = (h*1.0f / in_h);
	float shiftX = 0.f, shiftY = 0.f;
	const int n = in_w * in_h;
	int blockSize = 1024;
	const int gridSize = (n + blockSize - 1) / blockSize;
	resizeNormKernel << <gridSize, blockSize, 0, stream >> > ((uchar3*)(p), d, in_w, in_h, w, h, scaleX, scaleY, shiftX, shiftY);
	return 0;
}
int myresizeAndNorm(void * p, float *d, int w, int h, int in_w, int in_h, cudaStream_t stream) {
	float scaleX = (w*1.0f / in_w);
	float scaleY = (h*1.0f / in_h);
	float shiftX = 0.f, shiftY = 0.f;
	const int n = in_w * in_h;
	const dim3 blockDim(32,32);
	const dim3 gridDim((in_w+32-1)/32,(in_h+32-1)/32);
	int blockSize = 1024;
	const int gridSize = (n + blockSize - 1) / blockSize;
	myresizeNormKernel << <gridDim, blockDim, 0, stream >> > ((uchar3*)(p), d, in_w, in_h, w, h, scaleX, scaleY, shiftX, shiftY);
	return 0;
}