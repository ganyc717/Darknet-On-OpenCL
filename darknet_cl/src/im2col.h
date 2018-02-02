#ifndef IM2COL_H
#define IM2COL_H

void im2col_cpu(float* data_im,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_col);

#ifdef GPU
#include"ocl.h"
void im2col_gpu(CLArray im,
         int channels, int height, int width,
         int ksize, int stride, int pad, CLArray data_col);

#endif
#endif
