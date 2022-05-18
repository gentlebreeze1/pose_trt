

#include "cuda_runtime.h"
typedef unsigned char uchar;
int resizeAndNorm(void * p, float *d, int w, int h, int in_w, int in_h, cudaStream_t stream);
int myresizeAndNorm(void * p, float *d, int w, int h, int in_w, int in_h, cudaStream_t stream);
