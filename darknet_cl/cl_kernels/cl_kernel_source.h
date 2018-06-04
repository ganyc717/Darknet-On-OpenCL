#ifndef CL_KERNEL_SOURCE
#define CL_KERNEL_SOURCE
#include<map>
std::string activation_kernels = "\n\
typedef enum{\n\
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN\n\
} ACTIVATION;\n\
\n\
typedef enum{\n\
    MULT, ADD, SUB, DIV\n\
} BINARY_ACTIVATION;\n\
\n\
\n\
float lhtan_activate_kernel(float x)\n\
{\n\
    if(x < 0) return .001f*x;\n\
    if(x > 1) return .001f*(x-1.f) + 1.f;\n\
    return x;\n\
}\n\
\n\
float lhtan_gradient_kernel(float x)\n\
{\n\
    if(x > 0 && x < 1) return 1;\n\
    return .001;\n\
}\n\
\n\
float hardtan_activate_kernel(float x)\n\
{\n\
    if (x < -1) return -1;\n\
    if (x > 1) return 1;\n\
    return x;\n\
}\n\
float linear_activate_kernel(float x){return x;}\n\
float logistic_activate_kernel(float x){return 1.f/(1.f + exp(-x));}\n\
float loggy_activate_kernel(float x){return 2.f/(1.f + exp(-x)) - 1;}\n\
float relu_activate_kernel(float x){return x*(x>0);}\n\
float elu_activate_kernel(float x){return (x >= 0)*x + (x < 0)*(exp(x)-1);}\n\
float relie_activate_kernel(float x){return (x>0) ? x : .01f*x;}\n\
float ramp_activate_kernel(float x){return x*(x>0)+.1f*x;}\n\
float leaky_activate_kernel(float x){return (x>0) ? x : .1f*x;}\n\
float tanh_activate_kernel(float x){return (2.f/(1 + exp(-2*x)) - 1);}\n\
float plse_activate_kernel(float x)\n\
{\n\
    if(x < -4) return .01f * (x + 4);\n\
    if(x > 4)  return .01f * (x - 4) + 1;\n\
    return .125f*x + .5f;\n\
}\n\
\n\
float stair_activate_kernel(float x)\n\
{\n\
    int n = floor(x);\n\
    if (n%2 == 0) return floor(x/2);\n\
    else return (x - n) + floor(x/2);\n\
}\n\
 \n\
\n\
float hardtan_gradient_kernel(float x)\n\
{\n\
    if (x > -1 && x < 1) return 1;\n\
    return 0;\n\
}\n\
float linear_gradient_kernel(float x){return 1;}\n\
float logistic_gradient_kernel(float x){return (1-x)*x;}\n\
float loggy_gradient_kernel(float x)\n\
{\n\
    float y = (x+1)/2;\n\
    return 2*(1-y)*y;\n\
}\n\
\n\
float relu_gradient_kernel(float x){return (x>0);}\n\
float elu_gradient_kernel(float x){return (x >= 0) + (x < 0)*(x + 1);}\n\
float relie_gradient_kernel(float x){return (x>0) ? 1 : .01f;}\n\
float ramp_gradient_kernel(float x){return (x>0)+.1f;}\n\
float leaky_gradient_kernel(float x){return (x>0) ? 1 : .1f;}\n\
float tanh_gradient_kernel(float x){return 1-x*x;}\n\
float plse_gradient_kernel(float x){return (x < 0 || x > 1) ? .01f : .125f;}\n\
float stair_gradient_kernel(float x)\n\
{\n\
    if (floor(x) == x) return 0;\n\
    return 1;\n\
}\n\
\n\
float activate_kernel(float x, int a)\n\
{\n\
    switch(a){\n\
        case LINEAR:\n\
            return linear_activate_kernel(x);\n\
        case LOGISTIC:\n\
            return logistic_activate_kernel(x);\n\
        case LOGGY:\n\
            return loggy_activate_kernel(x);\n\
        case RELU:\n\
            return relu_activate_kernel(x);\n\
        case ELU:\n\
            return elu_activate_kernel(x);\n\
        case RELIE:\n\
            return relie_activate_kernel(x);\n\
        case RAMP:\n\
            return ramp_activate_kernel(x);\n\
        case LEAKY:\n\
            return leaky_activate_kernel(x);\n\
        case TANH:\n\
            return tanh_activate_kernel(x);\n\
        case PLSE:\n\
            return plse_activate_kernel(x);\n\
        case STAIR:\n\
            return stair_activate_kernel(x);\n\
        case HARDTAN:\n\
            return hardtan_activate_kernel(x);\n\
        case LHTAN:\n\
            return lhtan_activate_kernel(x);\n\
    }\n\
    return 0;\n\
}\n\
\n\
float gradient_kernel(float x, int a)\n\
{\n\
    switch(a){\n\
        case LINEAR:\n\
            return linear_gradient_kernel(x);\n\
        case LOGISTIC:\n\
            return logistic_gradient_kernel(x);\n\
        case LOGGY:\n\
            return loggy_gradient_kernel(x);\n\
        case RELU:\n\
            return relu_gradient_kernel(x);\n\
        case ELU:\n\
            return elu_gradient_kernel(x);\n\
        case RELIE:\n\
            return relie_gradient_kernel(x);\n\
        case RAMP:\n\
            return ramp_gradient_kernel(x);\n\
        case LEAKY:\n\
            return leaky_gradient_kernel(x);\n\
        case TANH:\n\
            return tanh_gradient_kernel(x);\n\
        case PLSE:\n\
            return plse_gradient_kernel(x);\n\
        case STAIR:\n\
            return stair_gradient_kernel(x);\n\
        case HARDTAN:\n\
            return hardtan_gradient_kernel(x);\n\
        case LHTAN:\n\
            return lhtan_gradient_kernel(x);\n\
    }\n\
    return 0;\n\
}\n\
\n\
__kernel void binary_gradient_array_kernel(__global float *x, __global float *dy, int n, int s, int a, __global float *dx)\n\
{\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    int i = id % s;\n\
    int b = id / s;\n\
    float x1 = x[b*s + i];\n\
    float x2 = x[b*s + s/2 + i];\n\
    if(id < n) {\n\
        float de = dy[id];\n\
        dx[b*s + i] = x2*de;\n\
        dx[b*s + s/2 + i] = x1*de; \n\
    }\n\
}\n\
\n\
__kernel void binary_activate_array_kernel(__global float *x, int n, int s, int a, __global float *y)\n\
{\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    int i = id % s;\n\
    int b = id / s;\n\
    float x1 = x[b*s + i];\n\
    float x2 = x[b*s + s/2 + i];\n\
    if(id < n) y[id] = x1*x2;\n\
}\n\
\n\
\n\
__kernel void activate_array_kernel(__global float *x, int n, int a)\n\
{\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(i < n) x[i] = activate_kernel(x[i], a);\n\
}\n\
\n\
__kernel void gradient_array_kernel(__global float *x, int n, int a, __global float *delta)\n\
{\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(i < n) delta[i] *= gradient_kernel(x[i], a);\n\
}\n\
"; 
std::string avgpool_layer_kernels = "\n\
__kernel void forward_avgpool_layer_kernel(int n, int w, int h, int c, __global float *input, __global float *output)\n\
{\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(id >= n) return;\n\
\n\
    int k = id % c;\n\
    id /= c;\n\
    int b = id;\n\
\n\
    int i;\n\
    int out_index = (k + c*b);\n\
    output[out_index] = 0;\n\
    for(i = 0; i < w*h; ++i){\n\
        int in_index = i + h*w*(k + b*c);\n\
        output[out_index] += input[in_index];\n\
    }\n\
    output[out_index] /= w*h;\n\
}\n\
\n\
__kernel void backward_avgpool_layer_kernel(int n, int w, int h, int c, __global float *in_delta, __global float *out_delta)\n\
{\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(id >= n) return;\n\
\n\
    int k = id % c;\n\
    id /= c;\n\
    int b = id;\n\
\n\
    int i;\n\
    int out_index = (k + c*b);\n\
    for(i = 0; i < w*h; ++i){\n\
        int in_index = i + h*w*(k + b*c);\n\
        in_delta[in_index] += out_delta[out_index] / (w*h);\n\
    }\n\
}\n\
"; 
std::string blas_kernels_1 = "\n\
#define BLOCK 512\n\
__kernel void scale_bias_kernel(__global float *output,__global float *biases, int n, int size)\n\
{\n\
	size_t global_x = get_global_id(0);\n\
	size_t global_y = get_global_id(1);\n\
	size_t global_z = get_global_id(2);\n\
	size_t filter = global_z - get_global_offset(2);\n\
	size_t x_dim_size = get_global_size(0);\n\
	size_t offset = x_dim_size * global_y + global_x;\n\
	if(offset < size) output[global_z*size + offset] *= biases[filter];\n\
}\n\
__kernel void backward_scale_kernel(__global float *x_norm, __global  float *delta, int batch, int n, int size, __global float *scale_updates)\n\
{\n\
    __local float part[BLOCK];\n\
    int i,b;\n\
	int filter = get_global_id(0);\n\
	int p = get_global_id(1);\n\
    float sum = 0;\n\
    for(b = 0; b < batch; ++b){\n\
        for(i = 0; i < size; i += BLOCK){\n\
            int index = p + i + size*(filter + n*b);\n\
            sum += (p+i < size) ? delta[index]*x_norm[index] : 0;\n\
        }\n\
    }\n\
    part[p] = sum;\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
    if (p == 0) {\n\
        for(i = 0; i < BLOCK; ++i) scale_updates[filter] += part[i];\n\
    }\n\
}\n\
__kernel void add_bias_kernel(__global float *output, __global float *biases, int batch, int n, int size)\n\
{\n\
	int index = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
	if (index >= n * size*batch) return;\n\
	int i = index % size;\n\
	index /= size;\n\
	int j = index % n;\n\
	index /= n;\n\
	int k = index;\n\
	output[(k*n + j)*size + i] += biases[j];\n\
}\n\
__kernel void backward_bias_conn_kernel(__global float *bias_updates, __global float *delta, int batch, int n)\n\
{\n\
	int index = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (index >= n) return;\n\
    int b;\n\
    float sum = 0;\n\
    for(b = 0; b < batch; ++b){\n\
        int i = b*n + index;\n\
        sum += delta[i];\n\
    }\n\
    bias_updates[index] += sum;\n\
}\n\
__kernel void backward_bias_kernel(__global float *bias_updates, __global float *delta, int batch, int n, int size)\n\
{\n\
	__local float part[BLOCK];\n\
	int i, b;\n\
	int filter = get_global_id(0);\n\
	int p = get_global_id(1);\n\
	float sum = 0;\n\
	for (b = 0; b < batch; ++b) {\n\
		for (i = 0; i < size; i += BLOCK) {\n\
			int index = p + i + size * (filter + n * b);\n\
			sum += (p + i < size) ? delta[index] : 0;\n\
		}\n\
	}\n\
	part[p] = sum;\n\
	barrier(CLK_LOCAL_MEM_FENCE);\n\
	if (p == 0) {\n\
		for (i = 0; i < BLOCK; ++i) bias_updates[filter] += part[i];\n\
	}\n\
}\n\
__kernel void adam_kernel(int N, __global float *x, __global float *m, __global float *v, float B1, float B2, float rate, float eps, int t)\n\
{\n\
	int index = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (index >= N) return;\n\
    float mhat = m[index] / (1.0 - pow(B1, t));\n\
    float vhat = v[index] / (1.0 - pow(B2, t));\n\
    x[index] = x[index] + rate * mhat / (sqrt(vhat) + eps);\n\
}\n\
__kernel void normalize_kernel(int N, __global float *x, __global float *mean, __global float *variance, int batch, int filters, int spatial)\n\
{\n\
	int index = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (index >= N) return;\n\
    int f = (index/spatial)%filters;\n\
    x[index] = (x[index] - mean[f])/(sqrt(variance[f] + .00001f));\n\
}\n\
__kernel void normalize_delta_kernel(int N, __global float *x, __global float *mean, __global float *variance, __global float *mean_delta, __global float *variance_delta, int batch, int filters, int spatial, __global float *delta)\n\
{\n\
	int index = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (index >= N) return;\n\
    int f = (index/spatial)%filters;\n\
    delta[index] = delta[index] * 1.f/(sqrt(variance[f] + .00001f)) + variance_delta[f] * 2.f * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);\n\
}\n\
__kernel void fast_mean_delta_kernel(__global float *delta, __global float *variance, int batch, int filters, int spatial, __global float *mean_delta)\n\
{\n\
	__local float part[BLOCK];\n\
	int id = get_global_id(1);\n\
	int filter = get_global_id(0);\n\
	float sum = 0;\n\
	int i, j;\n\
	for (j = 0; j < batch; ++j) {\n\
		for (i = 0; i < spatial; i += BLOCK) {\n\
			int index = j * spatial*filters + filter * spatial + i + id;\n\
			sum += (i + id < spatial) ? delta[index] : 0;\n\
		}\n\
	}\n\
	part[id] = sum;\n\
	barrier(CLK_LOCAL_MEM_FENCE);\n\
	if (id == 0) {\n\
		mean_delta[filter] = 0;\n\
		for (i = 0; i < BLOCK; ++i) {\n\
			mean_delta[filter] += part[i];\n\
		}\n\
		mean_delta[filter] *= (-1.f / sqrt(variance[filter] + .00001f));\n\
	}\n\
}\n\
__kernel void  fast_variance_delta_kernel(__global float *x, __global float *delta, __global float *mean, __global float *variance, int batch, int filters, int spatial, __global float *variance_delta)\n\
{\n\
	__local float part[BLOCK];\n\
	int id = get_global_id(1);\n\
	int filter = get_global_id(0);\n\
	float sum = 0;\n\
	int i, j;\n\
    for(j = 0; j < batch; ++j){\n\
        for(i = 0; i < spatial; i += BLOCK){\n\
            int index = j*spatial*filters + filter*spatial + i + id;\n\
			sum += (i+id < spatial) ? delta[index]*(x[index] - mean[filter]) : 0;\n\
        }\n\
    }\n\
	part[id] = sum;\n\
	barrier(CLK_LOCAL_MEM_FENCE);\n\
    if(id == 0){\n\
        variance_delta[filter] = 0;\n\
        for(i = 0; i < BLOCK; ++i){\n\
            variance_delta[filter] += part[i];\n\
        }\n\
        variance_delta[filter] *= -.5f * pow(variance[filter] + .00001f, (float)(-3.f/2.f));\n\
    }\n\
}\n\
__kernel void mean_delta_kernel(__global float *delta, __global float *variance, int batch, int filters, int spatial, __global float *mean_delta)\n\
{\n\
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (i >= filters) return;\n\
    int j,k;\n\
    mean_delta[i] = 0;\n\
    for (j = 0; j < batch; ++j) {\n\
        for (k = 0; k < spatial; ++k) {\n\
            int index = j*filters*spatial + i*spatial + k;\n\
            mean_delta[i] += delta[index];\n\
        }\n\
    }\n\
    mean_delta[i] *= (-1.f/sqrt(variance[i] + .00001f));\n\
}\n\
__kernel void  mean_kernel(__global float *x, int batch, int filters, int spatial, __global float *mean)\n\
{\n\
    float scale = 1.f/(batch * spatial);\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (i >= filters) return;\n\
    int j,k;\n\
    mean[i] = 0;\n\
    for(j = 0; j < batch; ++j){\n\
        for(k = 0; k < spatial; ++k){\n\
            int index = j*filters*spatial + i*spatial + k;\n\
            mean[i] += x[index];\n\
        }\n\
    }\n\
    mean[i] *= scale;\n\
}\n\
__kernel void variance_kernel(__global float *x, __global float *mean, int batch, int filters, int spatial, __global float *variance)\n\
{\n\
    float scale = 1.f/(batch * spatial - 1);\n\
    int j,k;\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (i >= filters) return;\n\
    variance[i] = 0;\n\
    for(j = 0; j < batch; ++j){\n\
        for(k = 0; k < spatial; ++k){\n\
            int index = j*filters*spatial + i*spatial + k;\n\
            variance[i] += pow((x[index] - mean[i]), 2);\n\
        }\n\
    }\n\
    variance[i] *= scale;\n\
}\n\
__kernel void reorg_kernel(int N, __global float *x, int w, int h, int c, int batch, int stride, int forward, __global float *out)\n\
{\n\
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(i >= N) return;\n\
    int in_index = i;\n\
    int in_w = i%w;\n\
    i = i/w;\n\
    int in_h = i%h;\n\
    i = i/h;\n\
    int in_c = i%c;\n\
    i = i/c;\n\
    int b = i%batch;\n\
    int out_c = c/(stride*stride);\n\
    int c2 = in_c % out_c;\n\
    int offset = in_c / out_c;\n\
    int w2 = in_w*stride + offset % stride;\n\
    int h2 = in_h*stride + offset / stride;\n\
    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));\n\
    if(forward) out[out_index] = x[in_index];\n\
    else out[in_index] = x[out_index];\n\
}\n\
__kernel void axpy_kernel(int N, float ALPHA, __global float *X, int OFFX, int INCX, __global float *Y, int OFFY, int INCY)\n\
{\n\
	size_t global_x = get_global_id(0);\n\
	size_t global_y = get_global_id(1);\n\
	size_t global_z = get_global_id(2);\n\
	size_t i = global_z * get_global_size(0) * get_global_size(1)\n\
		+ global_y * get_global_size(0) + global_x;\n\
    if(i < N) Y[OFFY+i*INCY] += ALPHA*X[OFFX+i*INCX];\n\
}\n\
__kernel void pow_kernel(int N, float ALPHA, __global float *X, int INCX, __global float *Y, int INCY)\n\
{\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(i < N) Y[i*INCY] = pow(X[i*INCX], ALPHA);\n\
}\n\
__kernel void const_kernel(int N, float ALPHA, __global float *X, int INCX)\n\
{\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(i < N) X[i*INCX] = ALPHA;\n\
}\n\
__kernel void constrain_kernel(int N, float ALPHA, __global float *X, int INCX)\n\
{\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(i < N) X[i*INCX] = fmin(ALPHA, fmax(-ALPHA, X[i*INCX]));\n\
}\n\
__kernel void supp_kernel(int N, float ALPHA, __global float *X, int INCX)\n\
{\n\
	size_t global_x = get_global_id(0);\n\
	size_t global_y = get_global_id(1);\n\
	size_t global_z = get_global_id(2);\n\
	size_t i = global_z * get_global_size(0) * get_global_size(1)\n\
		+ global_y * get_global_size(0) + global_x;\n\
    if(i < N) {\n\
        if((X[i*INCX] * X[i*INCX]) < (ALPHA * ALPHA)) X[i*INCX] = 0;\n\
    }\n\
}\n\
__kernel void add_kernel(int N, float ALPHA, __global float *X, int INCX)\n\
{\n\
	size_t global_x = get_global_id(0);\n\
	size_t global_y = get_global_id(1);\n\
	size_t global_z = get_global_id(2);\n\
	size_t i = global_z * get_global_size(0) * get_global_size(1)\n\
		+ global_y * get_global_size(0) + global_x;\n\
    if(i < N) X[i*INCX] += ALPHA;\n\
}\n\
__kernel void scal_kernel(int N, float ALPHA, __global float *X, int INCX)\n\
{\n\
	size_t global_x = get_global_id(0);\n\
	size_t global_y = get_global_id(1);\n\
	size_t global_z = get_global_id(2);\n\
	size_t i = global_z * get_global_size(0) * get_global_size(1)\n\
		+ global_y * get_global_size(0) + global_x;\n\
    if(i < N) X[i*INCX] *= ALPHA;\n\
}\n\
__kernel void fill_kernel(int N, float ALPHA, __global float *X, int INCX)\n\
{\n\
	size_t global_x = get_global_id(0);\n\
	size_t global_y = get_global_id(1);\n\
	size_t global_z = get_global_id(2);\n\
	size_t i = global_z * get_global_size(0) * get_global_size(1) \n\
	               + global_y * get_global_size(0) + global_x;\n\
	if(i < N) X[i*INCX] = ALPHA;\n\
}\n\
__kernel void copy_kernel(int N, __global float *X, int OFFX, int INCX, __global float *Y, int OFFY, int INCY)\n\
{\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];\n\
}\n\
__kernel void mul_kernel(int N, __global float *X, int INCX, __global float *Y, int INCY)\n\
{\n\
	size_t global_x = get_global_id(0);\n\
	size_t global_y = get_global_id(1);\n\
	size_t global_z = get_global_id(2);\n\
	size_t i = global_z * get_global_size(0) * get_global_size(1)\n\
		+ global_y * get_global_size(0) + global_x;\n\
    if(i < N) Y[i*INCY] *= X[i*INCX];\n\
}\n\
__kernel void  fast_mean_kernel(__global float *x, int batch, int filters, int spatial, __global float *mean)\n\
{\n\
	__local float part[BLOCK];\n\
	int id = get_global_id(1);\n\
	int filter = get_global_id(0);\n\
	float sum = 0;\n\
	int i, j;\n\
    for(j = 0; j < batch; ++j){\n\
        for(i = 0; i < spatial; i += BLOCK){\n\
            int index = j*spatial*filters + filter*spatial + i + id;\n\
			sum += (i+id < spatial) ? x[index] : 0;\n\
        }\n\
    }\n\
	part[id] = sum;\n\
	barrier(CLK_LOCAL_MEM_FENCE);\n\
    if(id == 0){\n\
        mean[filter] = 0;\n\
        for(i = 0; i < BLOCK; ++i){\n\
            mean[filter] += part[i];\n\
        }\n\
        mean[filter] /= spatial * batch;\n\
    }\n\
}\n\
__kernel void  fast_variance_kernel(__global float *x, __global float *mean, int batch, int filters, int spatial, __global float *variance)\n\
{\n\
	__local float part[BLOCK];\n\
	int id = get_global_id(1);\n\
	int filter = get_global_id(0);\n\
	float sum = 0;\n\
	int i, j;\n\
    for(j = 0; j < batch; ++j){\n\
        for(i = 0; i < spatial; i += BLOCK){\n\
            int index = j*spatial*filters + filter*spatial + i + id;\n\
\n\
			sum += (i+id < spatial) ? pow((x[index] - mean[filter]), 2) : 0;\n\
        }\n\
    }\n\
	part[id] = sum;\n\
	barrier(CLK_LOCAL_MEM_FENCE);\n\
    if(id == 0){\n\
        variance[filter] = 0;\n\
        for(i = 0; i < BLOCK; ++i){\n\
            variance[filter] += part[i];\n\
        }\n\
        variance[filter] /= (spatial * batch - 1);\n\
    }\n\
}\n\
\n\
__kernel void flatten_kernel(int N, __global float *x, int spatial, int layers, int batch, int forward, __global float *out)\n\
{\n\
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
	if (i >= N) return;\n\
	int in_s = i % spatial;\n\
	i = i / spatial;\n\
	int in_c = i % layers;\n\
	i = i / layers;\n\
	int b = i;\n\
	int i1 = b * layers*spatial + in_c * spatial + in_s;\n\
	int i2 = b * layers*spatial + in_s * layers + in_c;\n\
	if (forward) out[i2] = x[i1];\n\
	else out[i1] = x[i2];\n\
}\n\
__kernel void mask_kernel(int n, __global float *x, float mask_num, __global float *mask, float scale)\n\
{\n\
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
	if (i < n && mask[i] == mask_num) x[i] *= scale;\n\
}\n\
__kernel void shortcut_kernel(int size, int minw, int minh, int minc, int stride, int samples, int batch, int w1, int h1, int c1, __global float *add, int w2, int h2, int c2, float s1, float s2, __global float *out)\n\
{\n\
	int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
	if (id >= size) return;\n\
	int i = id % minw;\n\
	id /= minw;\n\
	int j = id % minh;\n\
	id /= minh;\n\
	int k = id % minc;\n\
	id /= minc;\n\
	int b = id % batch;\n\
	int out_index = i * samples + w2 * (j*samples + h2 * (k + c2 * b));\n\
	int add_index = i * stride + w1 * (j*stride + h1 * (k + c1 * b));\n\
	out[out_index] = s1*out[out_index] + s2*add[add_index];\n\
}\n\
__kernel void smooth_l1_kernel(int n, __global float *pred, __global float *truth, __global float *delta, __global float *error)\n\
{\n\
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
	if (i < n) {\n\
		float diff = truth[i] - pred[i];\n\
		float abs_val = fabs(diff);\n\
		if (abs_val < 1) {\n\
			error[i] = diff * diff;\n\
			delta[i] = diff;\n\
		}\n\
		else {\n\
			error[i] = 2 * abs_val - 1;\n\
			delta[i] = (diff > 0) ? 1 : -1;\n\
		}\n\
	}\n\
}\n\
__kernel void l2_kernel(int n, __global float *pred, __global float *truth, __global float *delta, __global float *error)\n\
{\n\
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
	if (i < n) {\n\
		float diff = truth[i] - pred[i];\n\
		error[i] = diff * diff;\n\
		delta[i] = diff;\n\
	}\n\
}\n\
__kernel void l1_kernel(int n, __global float *pred, __global float *truth, __global float *delta, __global float *error)\n\
{\n\
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
	if (i < n) {\n\
		float diff = truth[i] - pred[i];\n\
		error[i] = fabs(diff);\n\
		delta[i] = (diff > 0) ? 1 : -1;\n\
	}\n\
}\n\
__kernel void weighted_sum_kernel(int n, __global float *a, __global float *b, __global float *s, __global float *c)\n\
{\n\
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
	float m = 0.0;\n\
	if(b != 0)\n\
		m = b[i];\n\
	if (i < n) {\n\
		c[i] = s[i] * a[i] + (1 - s[i])*m;\n\
	}\n\
}\n\
"; 
std::string blas_kernels_2 = "\n\
__kernel void deinter_kernel(int NX, __global float *X, int NY, __global float *Y, int B, __global float *OUT_)\n\
{\n\
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
	if (i < (NX + NY)*B) {\n\
		int b = i / (NX + NY);\n\
		int j = i % (NX + NY);\n\
		if (j < NX) {\n\
			if (X) X[b*NX + j] += OUT_[i];\n\
		}\n\
		else {\n\
			if (Y) Y[b*NY + j - NX] += OUT_[i];\n\
		}\n\
	}\n\
}\n\
__kernel void inter_kernel(int NX, __global float *X, int NY, __global float *Y, int B, __global float *OUT_)\n\
{\n\
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
	if (i < (NX + NY)*B) {\n\
		int b = i / (NX + NY);\n\
		int j = i % (NX + NY);\n\
		if (j < NX) {\n\
			OUT_[i] = X[b*NX + j];\n\
		}\n\
		else {\n\
			OUT_[i] = Y[b*NY + j - NX];\n\
		}\n\
	}\n\
}\n\
__kernel void weighted_delta_kernel(int n, __global float *a, __global float *b, __global float *s, __global float *da, __global float *db, __global float *ds, __global float *dc)\n\
{\n\
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
	if (i < n) {\n\
		if (da) da[i] += dc[i] * s[i];\n\
		if (db) db[i] += dc[i] * (1 - s[i]);\n\
		ds[i] += dc[i] * (a[i] - b[i]);\n\
	}\n\
}\n\
__kernel void mult_add_into_kernel(int n, __global float *a, __global float *b, __global float *c)\n\
{\n\
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
	if (i < n) {\n\
		c[i] += a[i] * b[i];\n\
	}\n\
}\n\
void softmax_device(__global float *input, int n, float temp, int stride, __global float *output)\n\
{\n\
	int i;\n\
	float sum = 0;\n\
	float largest = -INFINITY;\n\
	for (i = 0; i < n; ++i) {\n\
		int val = input[i*stride];\n\
		largest = (val>largest) ? val : largest;\n\
	}\n\
	for (i = 0; i < n; ++i) {\n\
		float e = exp(input[i*stride] / temp - largest / temp);\n\
		sum += e;\n\
		output[i*stride] = e;\n\
	}\n\
	for (i = 0; i < n; ++i) {\n\
		output[i*stride] /= sum;\n\
	}\n\
}\n\
__kernel void softmax_tree_kernel(__global float *input, int spatial, int batch, int stride, float temp, __global float *output, int groups, __constant int *group_size, __constant int *group_offset)\n\
{\n\
	int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
	if (id >= spatial * batch*groups) return;\n\
	int s = id % spatial;\n\
	id = id / spatial;\n\
	int g = id % groups;\n\
	int b = id / groups;\n\
	int goff = group_offset[g] * spatial;\n\
	int boff = b * stride;\n\
	softmax_device(input + goff + boff + s, group_size[g], temp, spatial, output + goff + boff + s);\n\
}\n\
__kernel void softmax_kernel(__global float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, __global float *output)\n\
{\n\
	int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
	if (id >= batch * groups) return;\n\
	int b = id / groups;\n\
	int g = id % groups;\n\
	softmax_device(input + b * batch_offset + g * group_offset, n, temp, stride, output + b * batch_offset + g * group_offset);\n\
}\n\
__kernel void l2norm_kernel(int N, __global float *x, __global float *dx, int batch, int filters, int spatial)\n\
{\n\
    int index = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
                    get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (index >= N) return;\n\
    int b = index / spatial;\n\
    int i = index % spatial;\n\
    int f;\n\
    float sum = 0;\n\
    for(f = 0; f < filters; ++f){\n\
        int index = b*filters*spatial + f*spatial + i;\n\
        sum += pow(x[index], 2);\n\
    }\n\
    sum = sqrt(sum);\n\
    if(sum == 0) sum = 1;\n\
    for(f = 0; f < filters; ++f){\n\
        int index = b*filters*spatial + f*spatial + i;\n\
        x[index] /= sum;\n\
        dx[index] = (1 - x[index]) / sum;\n\
    }\n\
}\n\
__kernel void scale_mask_kernel(int n, __global float *x, float mask_num, __global float *mask, float scale)\n\
{\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
                        get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(i < n && mask[i] == mask_num) x[i] *= scale;\n\
}\n\
__kernel void softmax_x_ent_kernel(int n, __global float *pred, __global float *truth, __global float *delta, __global float *error)\n\
{\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
                            get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(i < n){\n\
        float t = truth[i];\n\
        float p = pred[i];\n\
        error[i] = (t) ? -log(p) : 0;\n\
        delta[i] = t-p;\n\
    }\n\
}\n\
__kernel void logistic_x_ent_kernel(int n, __global float *pred, __global float *truth, __global float *delta, __global float *error)\n\
{\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
                            get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(i < n){\n\
        float t = truth[i];\n\
        float p = pred[i];\n\
        error[i] = -t*log(p+.0000001) - (1-t)*log(1-p+.0000001);\n\
        delta[i] = t-p;\n\
    }\n\
}\n\
__kernel void wgan_kernel(int n, __global float *pred, __global float *truth, __global float *delta, __global float *error)\n\
{\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
                                get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(i < n){\n\
        error[i] = truth[i] ? -pred[i] : pred[i];\n\
        delta[i] = (truth[i] > 0) ? 1 : -1;\n\
    }\n\
}\n\
void atomic_float_add(volatile __global float *addr, float v)\n\
{\n\
    volatile __global int *p = (volatile __global int *)addr;\n\
    int last_value;\n\
    float result;\n\
    do\n\
    {\n\
        last_value = *p;\n\
        result = v + as_float(last_value);\n\
    }while(atomic_cmpxchg(p, last_value, as_int(result)) != last_value);\n\
}\n\
__kernel void upsample_kernel(int N, __global float *x, int w, int h, int c, int batch, int stride, int forward, float scale, __global float *out)\n\
{\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
                                    get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(i >= N) return;\n\
    int out_index = i;\n\
    int out_w = i%(w*stride);\n\
    i = i/(w*stride);\n\
    int out_h = i%(h*stride);\n\
    i = i/(h*stride);\n\
    int out_c = i%c;\n\
    i = i/c;\n\
    int b = i%batch;\n\
    int in_w = out_w / stride;\n\
    int in_h = out_h / stride;\n\
    int in_c = out_c;\n\
    int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;\n\
    if(forward) out[out_index] += scale * x[in_index];\n\
    else atomic_float_add(x+in_index, scale * out[out_index]);\n\
}\n\
"; 
std::string col2im_kernels = "\n\
// The comment-out cuda code is from the src, and I would like to port to \n\
// opencl kernel code as below.\n\
\n\
// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu\n\
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE\n\
\n\
/*\n\
__global__ void col2im_gpu_kernel(const int n, const float* data_col,\n\
        const int height, const int width, const int ksize,\n\
        const int pad,\n\
        const int stride,\n\
        const int height_col, const int width_col,\n\
        float *data_im) {\n\
    int index = blockIdx.x*blockDim.x+threadIdx.x;\n\
    for(; index < n; index += blockDim.x*gridDim.x){\n\
        float val = 0;\n\
        int w = index % width + pad;\n\
        int h = (index / width) % height + pad;\n\
        int c = index / (width * height);\n\
        // compute the start and end of the output\n\
        int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;\n\
        int w_col_end = min(w / stride + 1, width_col);\n\
        int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;\n\
        int h_col_end = min(h / stride + 1, height_col);\n\
        // equivalent implementation\n\
        int offset =\n\
            (c * ksize * ksize + h * ksize + w) * height_col * width_col;\n\
        int coeff_h_col = (1 - stride * ksize * height_col) * width_col;\n\
        int coeff_w_col = (1 - stride * height_col * width_col);\n\
        for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {\n\
            for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {\n\
                val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];\n\
            }\n\
        }\n\
        data_im[index] += val;\n\
    }\n\
}\n\
\n\
void col2im_gpu(float *data_col,\n\
        int channels, int height, int width,\n\
        int ksize, int stride, int pad, float *data_im){\n\
    // We are going to launch channels * height_col * width_col kernels, each\n\
    // kernel responsible for copying a single-channel grid.\n\
    int height_col = (height + 2 * pad - ksize) / stride + 1;\n\
    int width_col = (width + 2 * pad - ksize) / stride + 1;\n\
    int num_kernels = channels * height * width;\n\
    col2im_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,\n\
        BLOCK>>>(\n\
                num_kernels, data_col, height, width, ksize, pad,\n\
                stride, height_col,\n\
                width_col, data_im);\n\
}\n\
*/\n\
\n\
__kernel void col2im_gpu_kernel(const int n, __global const float* data_col,\n\
        const int height, const int width, const int ksize,\n\
        const int pad,\n\
        const int stride,\n\
        const int height_col, const int width_col,\n\
        __global float *data_im) {\n\
	int index = get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    for(; index < n; index += get_global_size(1) * get_global_size(0)){\n\
        float val = 0;\n\
        int w = index % width + pad;\n\
        int h = (index / width) % height + pad;\n\
        int c = index / (width * height);\n\
        // compute the start and end of the output\n\
        int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;\n\
        int w_col_end = min(w / stride + 1, width_col);\n\
        int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;\n\
        int h_col_end = min(h / stride + 1, height_col);\n\
        // equivalent implementation\n\
        int offset =\n\
            (c * ksize * ksize + h * ksize + w) * height_col * width_col;\n\
        int coeff_h_col = (1 - stride * ksize * height_col) * width_col;\n\
        int coeff_w_col = (1 - stride * height_col * width_col);\n\
        for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {\n\
            for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {\n\
                val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];\n\
            }\n\
        }\n\
        data_im[index] += val;\n\
    }\n\
}\n\
"; 
std::string convolutional_kernels = "\n\
__kernel void binarize_kernel(__global float *x, int n, __global float *binary)\n\
{\n\
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (i >= n) return;\n\
    binary[i] = (x[i] >= 0) ? 1 : -1;\n\
}\n\
\n\
__kernel void binarize_input_kernel(__global float *input, int n, int size, __global float *binary)\n\
{\n\
    int s = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (s >= size) return;\n\
    int i = 0;\n\
    float mean = 0;\n\
    for(i = 0; i < n; ++i){\n\
        mean += fabs(input[i*size + s]);\n\
    }\n\
    mean = mean / n;\n\
    for(i = 0; i < n; ++i){\n\
        binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;\n\
    }\n\
}\n\
\n\
\n\
__kernel void binarize_weights_kernel(__global float *weights, int n, int size, __global float *binary)\n\
{\n\
    int f = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if (f >= n) return;\n\
    int i = 0;\n\
    float mean = 0;\n\
    for(i = 0; i < size; ++i){\n\
        mean += fabs(weights[f*size + i]);\n\
    }\n\
    mean = mean / size;\n\
    for(i = 0; i < size; ++i){\n\
        binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;\n\
        //binary[f*size + i] = weights[f*size + i];\n\
    }\n\
}\n\
\n\
__kernel void smooth_kernel(__global float *x, int n, int w, int h, int c, int size, float rate, __global float *delta)\n\
{\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(id >= n) return;\n\
\n\
    int j = id % w;\n\
    id /= w;\n\
    int i = id % h;\n\
    id /= h;\n\
    int k = id % c;\n\
    id /= c;\n\
    int b = id;\n\
\n\
    int w_offset = -(size/2.f);\n\
    int h_offset = -(size/2.f);\n\
\n\
    int out_index = j + w*(i + h*(k + c*b));\n\
    int l, m;\n\
    for(l = 0; l < size; ++l){\n\
        for(m = 0; m < size; ++m){\n\
            int cur_h = h_offset + i + l;\n\
            int cur_w = w_offset + j + m;\n\
            int index = cur_w + w*(cur_h + h*(k + b*c));\n\
            int valid = (cur_h >= 0 && cur_h < h &&\n\
                    cur_w >= 0 && cur_w < w);\n\
            delta[out_index] += valid ? rate*(x[index] - x[out_index]) : 0;\n\
        }\n\
    }\n\
}\n\
\n\
\n\
"; 
std::string crop_layer_kernels = "\n\
float get_pixel_kernel(__global float *image, int w, int h, int x, int y, int c)\n\
{\n\
    if(x < 0 || x >= w || y < 0 || y >= h) return 0;\n\
    return image[x + w*(y + c*h)];\n\
}\n\
\n\
float3 rgb_to_hsv_kernel(float3 rgb)\n\
{\n\
    float r = rgb.x;\n\
    float g = rgb.y; \n\
    float b = rgb.z;\n\
\n\
    float h, s, v;\n\
    float max_ = (r > g) ? ( (r > b) ? r : b) : ( (g > b) ? g : b);\n\
    float min_ = (r < g) ? ( (r < b) ? r : b) : ( (g < b) ? g : b);\n\
    float delta = max_ - min_;\n\
    v = max_;\n\
    if(max_ == 0){\n\
        s = 0;\n\
        h = -1;\n\
    }else{\n\
        s = delta/max_;\n\
        if(r == max_){\n\
            h = (g - b) / delta;\n\
        } else if (g == max_) {\n\
            h = 2 + (b - r) / delta;\n\
        } else {\n\
            h = 4 + (r - g) / delta;\n\
        }\n\
        if (h < 0) h += 6;\n\
    }\n\
    return (float3)(h, s, v);\n\
}\n\
\n\
float3 hsv_to_rgb_kernel(float3 hsv)\n\
{\n\
    float h = hsv.x;\n\
    float s = hsv.y; \n\
    float v = hsv.z;\n\
\n\
    float r, g, b;\n\
    float f, p, q, t;\n\
\n\
    if (s == 0) {\n\
        r = g = b = v;\n\
    } else {\n\
        int index = (int) floor(h);\n\
        f = h - index;\n\
        p = v*(1-s);\n\
        q = v*(1-s*f);\n\
        t = v*(1-s*(1-f));\n\
        if(index == 0){\n\
            r = v; g = t; b = p;\n\
        } else if(index == 1){\n\
            r = q; g = v; b = p;\n\
        } else if(index == 2){\n\
            r = p; g = v; b = t;\n\
        } else if(index == 3){\n\
            r = p; g = q; b = v;\n\
        } else if(index == 4){\n\
            r = t; g = p; b = v;\n\
        } else {\n\
            r = v; g = p; b = q;\n\
        }\n\
    }\n\
    r = (r < 0) ? 0 : ((r > 1) ? 1 : r);\n\
    g = (g < 0) ? 0 : ((g > 1) ? 1 : g);\n\
    b = (b < 0) ? 0 : ((b > 1) ? 1 : b);\n\
    return (float3)(r, g, b);\n\
}\n\
\n\
float bilinear_interpolate_kernel(__global float *image, int w, int h, float x, float y, int c)\n\
{\n\
    int ix = (int) floor(x);\n\
    int iy = (int) floor(y);\n\
\n\
    float dx = x - ix;\n\
    float dy = y - iy;\n\
\n\
    float val = (1-dy) * (1-dx) * get_pixel_kernel(image, w, h, ix, iy, c) + \n\
        dy     * (1-dx) * get_pixel_kernel(image, w, h, ix, iy+1, c) + \n\
        (1-dy) *   dx   * get_pixel_kernel(image, w, h, ix+1, iy, c) +\n\
        dy     *   dx   * get_pixel_kernel(image, w, h, ix+1, iy+1, c);\n\
    return val;\n\
}\n\
\n\
__kernel void levels_image_kernel(__global float *image, __global float *rand, int batch, int w, int h, int train, float saturation, float exposure, float translate, float scale, float shift)\n\
{\n\
    int size = batch * w * h;\n\
\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
	if(id >= size) return;\n\
    int x = id % w;\n\
    id /= w;\n\
    int y = id % h;\n\
    id /= h;\n\
    float rshift = rand[0];\n\
    float gshift = rand[1];\n\
    float bshift = rand[2];\n\
    float r0 = rand[8*id + 0];\n\
    float r1 = rand[8*id + 1];\n\
    float r2 = rand[8*id + 2];\n\
    float r3 = rand[8*id + 3];\n\
\n\
    saturation = r0*(saturation - 1) + 1;\n\
    saturation = (r1 > .5f) ? 1.f/saturation : saturation;\n\
    exposure = r2*(exposure - 1) + 1;\n\
    exposure = (r3 > .5f) ? 1.f/exposure : exposure;\n\
\n\
    size_t offset = id * h * w * 3;\n\
    //image += offset;\n\
    float r = image[x + w*(y + h*0) + offset];\n\
    float g = image[x + w*(y + h*1) + offset];\n\
    float b = image[x + w*(y + h*2) + offset];\n\
    float3 rgb = (float3)(r,g,b);\n\
    if(train){\n\
        float3 hsv = rgb_to_hsv_kernel(rgb);\n\
        hsv.y *= saturation;\n\
        hsv.z *= exposure;\n\
        rgb = hsv_to_rgb_kernel(hsv);\n\
    } else {\n\
        shift = 0;\n\
    }\n\
    image[x + w*(y + h*0) + offset] = rgb.x*scale + translate + (rshift - .5f)*shift;\n\
    image[x + w*(y + h*1) + offset] = rgb.y*scale + translate + (gshift - .5f)*shift;\n\
    image[x + w*(y + h*2) + offset] = rgb.z*scale + translate + (bshift - .5f)*shift;\n\
}\n\
\n\
__kernel void forward_crop_layer_kernel(__global float *input, __global float *rand, int size, int c, int h, int w, int crop_height, int crop_width, int train, int flip, float angle, __global float *output)\n\
{\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(id >= size) return;\n\
\n\
    float cx = w/2.f;\n\
    float cy = h/2.f;\n\
\n\
    int count = id;\n\
    int j = id % crop_width;\n\
    id /= crop_width;\n\
    int i = id % crop_height;\n\
    id /= crop_height;\n\
    int k = id % c;\n\
    id /= c;\n\
    int b = id;\n\
\n\
    float r4 = rand[8*b + 4];\n\
    float r5 = rand[8*b + 5];\n\
    float r6 = rand[8*b + 6];\n\
    float r7 = rand[8*b + 7];\n\
\n\
    float dw = (w - crop_width)*r4;\n\
    float dh = (h - crop_height)*r5;\n\
    flip = (flip && (r6 > .5f));\n\
    angle = 2*angle*r7 - angle;\n\
    if(!train){\n\
        dw = (w - crop_width)/2.f;\n\
        dh = (h - crop_height)/2.f;\n\
        flip = 0;\n\
        angle = 0;\n\
    }\n\
\n\
    input += w*h*c*b;\n\
\n\
    float x = (flip) ? w - dw - j - 1 : j + dw;    \n\
    float y = i + dh;\n\
\n\
    float rx = cos(angle)*(x-cx) - sin(angle)*(y-cy) + cx;\n\
    float ry = sin(angle)*(x-cx) + cos(angle)*(y-cy) + cy;\n\
\n\
    output[count] = bilinear_interpolate_kernel(input, w, h, rx, ry, k);\n\
}\n\
"; 
std::string deconvolutional_kernels = "\n\
//This is empty file, may delete later \n\
"; 
std::string dropout_layer_kernels = "\n\
__kernel void yoloswag420blazeit360noscope(__global float *input, int size, __global float *rand, float prob, float scale)\n\
{\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;\n\
}\n\
"; 
std::string im2col_kernels = "\n\
// The comment-out cuda code is from the src, and I would like to port to \n\
// opencl kernel code as below.\n\
\n\
// src: https://github.com/BVLC/caffe/blob/master/src/caffe/util/im2col.cu\n\
// You may also want to read: https://github.com/BVLC/caffe/blob/master/LICENSE\n\
\n\
/*\n\
__global__ void im2col_gpu_kernel(const int n, const float* data_im,\n\
        const int height, const int width, const int ksize,\n\
        const int pad,\n\
        const int stride,\n\
        const int height_col, const int width_col,\n\
        float *data_col) {\n\
    int index = blockIdx.x*blockDim.x+threadIdx.x;\n\
    for(; index < n; index += blockDim.x*gridDim.x){\n\
        int w_out = index % width_col;\n\
        int h_index = index / width_col;\n\
        int h_out = h_index % height_col;\n\
        int channel_in = h_index / height_col;\n\
        int channel_out = channel_in * ksize * ksize;\n\
        int h_in = h_out * stride - pad;\n\
        int w_in = w_out * stride - pad;\n\
        float* data_col_ptr = data_col;\n\
        data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;\n\
        const float* data_im_ptr = data_im;\n\
        data_im_ptr += (channel_in * height + h_in) * width + w_in;\n\
        for (int i = 0; i < ksize; ++i) {\n\
            for (int j = 0; j < ksize; ++j) {\n\
                int h = h_in + i;\n\
                int w = w_in + j;\n\
\n\
                *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?\n\
                    data_im_ptr[i * width + j] : 0;\n\
\n\
                //*data_col_ptr = data_im_ptr[ii * width + jj];\n\
\n\
                data_col_ptr += height_col * width_col;\n\
            }\n\
        }\n\
    }\n\
}\n\
\n\
void im2col_gpu(float *im,\n\
         int channels, int height, int width,\n\
         int ksize, int stride, int pad, float *data_col){\n\
    // We are going to launch channels * height_col * width_col kernels, each\n\
    // kernel responsible for copying a single-channel grid.\n\
    int height_col = (height + 2 * pad - ksize) / stride + 1;\n\
    int width_col = (width + 2 * pad - ksize) / stride + 1;\n\
    int num_kernels = channels * height_col * width_col;\n\
    im2col_gpu_kernel<<<(num_kernels+BLOCK-1)/BLOCK,\n\
        BLOCK>>>(\n\
                num_kernels, im, height, width, ksize, pad,\n\
                stride, height_col,\n\
                width_col, data_col);\n\
}\n\
*/\n\
\n\
__kernel void im2col_gpu_kernel(const int n, __global const float* data_im,\n\
        const int height, const int width, const int ksize,\n\
        const int pad,\n\
        const int stride,\n\
        const int height_col, const int width_col,\n\
        __global float *data_col) {\n\
	int index = get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
	for(; index < n; index += get_global_size(1) * get_global_size(0)){\n\
        int w_out = index % width_col;\n\
        int h_index = index / width_col;\n\
        int h_out = h_index % height_col;\n\
        int channel_in = h_index / height_col;\n\
        int channel_out = channel_in * ksize * ksize;\n\
        int h_in = h_out * stride - pad;\n\
        int w_in = w_out * stride - pad;\n\
\n\
		int data_col_offset = (channel_out * height_col + h_out) * width_col + w_out;\n\
		int data_im_offset = (channel_in * height + h_in) * width + w_in;\n\
        for (int i = 0; i < ksize; ++i) {\n\
            for (int j = 0; j < ksize; ++j) {\n\
                int h = h_in + i;\n\
                int w = w_in + j;\n\
\n\
                data_col[data_col_offset] = (h >= 0 && w >= 0 && h < height && w < width) ?\n\
                    data_im[data_im_offset + i * width + j] : 0;\n\
\n\
                //*data_col_ptr = data_im_ptr[ii * width + jj];\n\
\n\
                data_col_offset += height_col * width_col;\n\
            }\n\
        }\n\
    }\n\
}\n\
"; 
std::string maxpool_layer_kernels = "\n\
__kernel void forward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, __global float *input, __global float *output, __global int *indexes)\n\
{\n\
    int h = (in_h + 2*pad)/stride;\n\
    int w = (in_w + 2*pad)/stride;\n\
    int c = in_c;\n\
\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(id >= n) return;\n\
\n\
    int j = id % w;\n\
    id /= w;\n\
    int i = id % h;\n\
    id /= h;\n\
    int k = id % c;\n\
    id /= c;\n\
    int b = id;\n\
\n\
    int w_offset = -pad;\n\
    int h_offset = -pad;\n\
\n\
    int out_index = j + w*(i + h*(k + c*b));\n\
    float max_value = -INFINITY;\n\
    int max_index = -1;\n\
    int l, m;\n\
    for(l = 0; l < size; ++l){\n\
        for(m = 0; m < size; ++m){\n\
            int cur_h = h_offset + i*stride + l;\n\
            int cur_w = w_offset + j*stride + m;\n\
            int index = cur_w + in_w*(cur_h + in_h*(k + b*in_c));\n\
            int valid = (cur_h >= 0 && cur_h < in_h &&\n\
                    cur_w >= 0 && cur_w < in_w);\n\
            float val = (valid != 0) ? input[index] : -INFINITY;\n\
            max_index = (val > max_value) ? index : max_index;\n\
            max_value   = (val > max_value) ? val   : max_value;\n\
        }\n\
    }\n\
    output[out_index] = max_value;\n\
    indexes[out_index] = max_index;\n\
}\n\
\n\
__kernel void backward_maxpool_layer_kernel(int n, int in_h, int in_w, int in_c, int stride, int size, int pad, __global float *delta, __global float *prev_delta, __global int *indexes)\n\
{\n\
    int h = (in_h + 2*pad)/stride;\n\
    int w = (in_w + 2*pad)/stride;\n\
    int c = in_c;\n\
    int area = (size-1)/stride;\n\
\n\
    int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +\n\
		get_global_id(1) * get_global_size(0) + get_global_id(0);\n\
    if(id >= n) return;\n\
\n\
    int index = id;\n\
    int j = id % in_w;\n\
    id /= in_w;\n\
    int i = id % in_h;\n\
    id /= in_h;\n\
    int k = id % in_c;\n\
    id /= in_c;\n\
    int b = id;\n\
\n\
    int w_offset = -pad;\n\
    int h_offset = -pad;\n\
\n\
    float d = 0;\n\
    int l, m;\n\
    for(l = -area; l < area+1; ++l){\n\
        for(m = -area; m < area+1; ++m){\n\
            int out_w = (j-w_offset)/stride + m;\n\
            int out_h = (i-h_offset)/stride + l;\n\
            int out_index = out_w + w*(out_h + h*(k + c*b));\n\
            int valid = (out_w >= 0 && out_w < w &&\n\
                     out_h >= 0 && out_h < h);\n\
            d += (valid && indexes[out_index] == index) ? delta[out_index] : 0;\n\
        }\n\
    }\n\
    prev_delta[index] += d;\n\
}\n\
"; 
std::map<std::string,std::string> source_map = {
std::make_pair("activation_kernels.cl",activation_kernels),
std::make_pair("avgpool_layer_kernels.cl",avgpool_layer_kernels),
std::make_pair("blas_kernels_1.cl",blas_kernels_1),
std::make_pair("blas_kernels_2.cl",blas_kernels_2),
std::make_pair("col2im_kernels.cl",col2im_kernels),
std::make_pair("convolutional_kernels.cl",convolutional_kernels),
std::make_pair("crop_layer_kernels.cl",crop_layer_kernels),
std::make_pair("deconvolutional_kernels.cl",deconvolutional_kernels),
std::make_pair("dropout_layer_kernels.cl",dropout_layer_kernels),
std::make_pair("im2col_kernels.cl",im2col_kernels),
std::make_pair("maxpool_layer_kernels.cl",maxpool_layer_kernels),
};
#endif