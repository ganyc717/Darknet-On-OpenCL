#ifndef BLAS_H
#define BLAS_H
#include "darknet.h"

void flatten(float *x, int size, int layers, int batch, int forward);
void pm(int M, int N, float *A);
float *random_matrix(int rows, int cols);
void time_random_matrix(int TA, int TB, int m, int k, int n);
void reorg_cpu(float *x, int w, int h, int c, int batch, int stride, int forward, float *out);

void test_blas();

void inter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
void deinter_cpu(int NX, float *X, int NY, float *Y, int B, float *OUT);
void mult_add_into_cpu(int N, float *X, float *Y, float *Z);

void const_cpu(int N, float ALPHA, float *X, int INCX);

void pow_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY);
void mul_cpu(int N, float *X, int INCX, float *Y, int INCY);

int test_gpu_blas();
void shortcut_cpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out);

void mean_cpu(float *x, int batch, int filters, int spatial, float *mean);
void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);

void scale_bias(float *output, float *scales, int batch, int n, int size);
void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates);
void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta);
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta);
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta);
void l2normalize_cpu(float *x, float *dx, int batch, int filters, int spatial);

void smooth_l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l2_cpu(int n, float *pred, float *truth, float *delta, float *error);
void l1_cpu(int n, float *pred, float *truth, float *delta, float *error);
void logistic_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error);
void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error);
void weighted_sum_cpu(float *a, float *b, float *s, int num, float *c);
void weighted_delta_cpu(float *a, float *b, float *s, float *da, float *db, float *ds, int n, float *dc);

void softmax(float *input, int n, float temp, int stride, float *output);
void softmax_cpu(float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, float *output);
void upsample_cpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);

#ifdef GPU
#include "ocl.h"
#include "tree.h"
void constrain_gpu(int N, float ALPHA, CLArray X, int INCX);
void axpy_gpu(int N, float ALPHA, CLArray X, int INCX, CLArray Y, int INCY);
void axpy_gpu_offset(int N, float ALPHA, CLArray X, int OFFX, int INCX, CLArray Y, int OFFY, int INCY);
void copy_gpu(int N, CLArray X, int INCX, CLArray Y, int INCY);
void copy_gpu_offset(int N, CLArray X, int OFFX, int INCX, CLArray Y, int OFFY, int INCY);
void add_gpu(int N, float ALPHA, CLArray X, int INCX);
void supp_gpu(int N, float ALPHA, CLArray X, int INCX);
void mask_gpu(int N, CLArray X, float mask_num, CLArray mask, float val);
void scale_mask_gpu(int N, CLArray X, float mask_num, CLArray mask, float scale);
void const_gpu(int N, float ALPHA, CLArray X, int INCX);
void pow_gpu(int N, float ALPHA, CLArray X, int INCX, CLArray Y, int INCY);
void mul_gpu(int N, CLArray X, int INCX, CLArray Y, int INCY);

void mean_gpu(CLArray x, int batch, int filters, int spatial, CLArray mean);
void variance_gpu(CLArray x, CLArray mean, int batch, int filters, int spatial, CLArray variance);
void normalize_gpu(CLArray x, CLArray mean, CLArray variance, int batch, int filters, int spatial);
void l2normalize_gpu(CLArray x, CLArray dx, int batch, int filters, int spatial);

void normalize_delta_gpu(CLArray x, CLArray mean, CLArray variance, CLArray mean_delta, CLArray variance_delta, int batch, int filters, int spatial, CLArray delta);
void fast_mean_delta_gpu(CLArray delta, CLArray variance, int batch, int filters, int spatial, CLArray mean_delta);
void fast_variance_delta_gpu(CLArray x, CLArray delta, CLArray mean, CLArray variance, int batch, int filters, int spatial, CLArray variance_delta);

void fast_variance_gpu(CLArray x, CLArray mean, int batch, int filters, int spatial, CLArray variance);
void fast_mean_gpu(CLArray x, int batch, int filters, int spatial, CLArray mean);
void shortcut_gpu(int batch, int w1, int h1, int c1, CLArray add, int w2, int h2, int c2, float s1, float s2, CLArray out);
void scale_bias_gpu(CLArray output, CLArray biases, int batch, int n, int size);
void backward_scale_gpu(CLArray x_norm, CLArray delta, int batch, int n, int size, CLArray scale_updates);
void scale_bias_gpu(CLArray output, CLArray  biases, int batch, int n, int size);
void add_bias_gpu(CLArray output, CLArray biases, int batch, int n, int size);
void backward_bias_gpu(CLArray bias_updates, CLArray delta, int batch, int n, int size);

void logistic_x_ent_gpu(int n, CLArray pred, CLArray truth, CLArray delta, CLArray error);
void softmax_x_ent_gpu(int n, CLArray pred, CLArray truth, CLArray delta, CLArray error);
void smooth_l1_gpu(int n, CLArray pred, CLArray truth, CLArray delta, CLArray error);
void l2_gpu(int n, CLArray pred, CLArray truth, CLArray delta, CLArray error);
void l1_gpu(int n, CLArray pred, CLArray truth, CLArray delta, CLArray error);
void wgan_gpu(int n, CLArray pred, CLArray truth, CLArray delta, CLArray error);
void weighted_delta_gpu(CLArray a, CLArray b, CLArray s, CLArray da, CLArray db, CLArray ds, int num, CLArray dc);
void weighted_sum_gpu(CLArray a, CLArray b, CLArray s, int num, CLArray c);
void mult_add_into_gpu(int num, CLArray a, CLArray b, CLArray c);
void inter_gpu(int NX, CLArray X, int NY, CLArray Y, int B, CLArray OUT_);
void deinter_gpu(int NX, CLArray X, int NY, CLArray Y, int B, CLArray OUT_);

void reorg_gpu(CLArray x, int w, int h, int c, int batch, int stride, int forward, CLArray out);

void softmax_gpu(CLArray input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, CLArray output);
void adam_update_gpu(CLArray w, CLArray d, CLArray m, CLArray v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
void adam_gpu(int n, CLArray x, CLArray m, CLArray v, float B1, float B2, float rate, float eps, int t);

void flatten_gpu(CLArray x, int spatial, int layers, int batch, int forward, CLArray out);
void softmax_tree(CLArray input, int spatial, int batch, int stride, float temp, CLArray output, tree hier);
void upsample_gpu(CLArray in, int w, int h, int c, int batch, int stride, int forward, float scale, CLArray out);
#endif
#endif
