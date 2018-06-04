#define BLOCK 512
__kernel void scale_bias_kernel(__global float *output,__global float *biases, int n, int size)
{
	size_t global_x = get_global_id(0);
	size_t global_y = get_global_id(1);
	size_t global_z = get_global_id(2);
	size_t filter = global_z - get_global_offset(2);
	size_t x_dim_size = get_global_size(0);
	size_t offset = x_dim_size * global_y + global_x;
	if(offset < size) output[global_z*size + offset] *= biases[filter];
}
__kernel void backward_scale_kernel(__global float *x_norm, __global  float *delta, int batch, int n, int size, __global float *scale_updates)
{
    __local float part[BLOCK];
    int i,b;
	int filter = get_global_id(0);
	int p = get_global_id(1);
    float sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index]*x_norm[index] : 0;
        }
    }
    part[p] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) scale_updates[filter] += part[i];
    }
}
__kernel void add_bias_kernel(__global float *output, __global float *biases, int batch, int n, int size)
{
	int index = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
	if (index >= n * size*batch) return;
	int i = index % size;
	index /= size;
	int j = index % n;
	index /= n;
	int k = index;
	output[(k*n + j)*size + i] += biases[j];
}
__kernel void backward_bias_conn_kernel(__global float *bias_updates, __global float *delta, int batch, int n)
{
	int index = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (index >= n) return;
    int b;
    float sum = 0;
    for(b = 0; b < batch; ++b){
        int i = b*n + index;
        sum += delta[i];
    }
    bias_updates[index] += sum;
}
__kernel void backward_bias_kernel(__global float *bias_updates, __global float *delta, int batch, int n, int size)
{
	__local float part[BLOCK];
	int i, b;
	int filter = get_global_id(0);
	int p = get_global_id(1);
	float sum = 0;
	for (b = 0; b < batch; ++b) {
		for (i = 0; i < size; i += BLOCK) {
			int index = p + i + size * (filter + n * b);
			sum += (p + i < size) ? delta[index] : 0;
		}
	}
	part[p] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (p == 0) {
		for (i = 0; i < BLOCK; ++i) bias_updates[filter] += part[i];
	}
}
__kernel void adam_kernel(int N, __global float *x, __global float *m, __global float *v, float B1, float B2, float rate, float eps, int t)
{
	int index = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (index >= N) return;
    float mhat = m[index] / (1.0 - pow(B1, t));
    float vhat = v[index] / (1.0 - pow(B2, t));
    x[index] = x[index] + rate * mhat / (sqrt(vhat) + eps);
}
__kernel void normalize_kernel(int N, __global float *x, __global float *mean, __global float *variance, int batch, int filters, int spatial)
{
	int index = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (index >= N) return;
    int f = (index/spatial)%filters;
    x[index] = (x[index] - mean[f])/(sqrt(variance[f] + .00001f));
}
__kernel void normalize_delta_kernel(int N, __global float *x, __global float *mean, __global float *variance, __global float *mean_delta, __global float *variance_delta, int batch, int filters, int spatial, __global float *delta)
{
	int index = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (index >= N) return;
    int f = (index/spatial)%filters;
    delta[index] = delta[index] * 1.f/(sqrt(variance[f] + .00001f)) + variance_delta[f] * 2.f * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
}
__kernel void fast_mean_delta_kernel(__global float *delta, __global float *variance, int batch, int filters, int spatial, __global float *mean_delta)
{
	__local float part[BLOCK];
	int id = get_global_id(1);
	int filter = get_global_id(0);
	float sum = 0;
	int i, j;
	for (j = 0; j < batch; ++j) {
		for (i = 0; i < spatial; i += BLOCK) {
			int index = j * spatial*filters + filter * spatial + i + id;
			sum += (i + id < spatial) ? delta[index] : 0;
		}
	}
	part[id] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);
	if (id == 0) {
		mean_delta[filter] = 0;
		for (i = 0; i < BLOCK; ++i) {
			mean_delta[filter] += part[i];
		}
		mean_delta[filter] *= (-1.f / sqrt(variance[filter] + .00001f));
	}
}
__kernel void  fast_variance_delta_kernel(__global float *x, __global float *delta, __global float *mean, __global float *variance, int batch, int filters, int spatial, __global float *variance_delta)
{
	__local float part[BLOCK];
	int id = get_global_id(1);
	int filter = get_global_id(0);
	float sum = 0;
	int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += BLOCK){
            int index = j*spatial*filters + filter*spatial + i + id;
			sum += (i+id < spatial) ? delta[index]*(x[index] - mean[filter]) : 0;
        }
    }
	part[id] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);
    if(id == 0){
        variance_delta[filter] = 0;
        for(i = 0; i < BLOCK; ++i){
            variance_delta[filter] += part[i];
        }
        variance_delta[filter] *= -.5f * pow(variance[filter] + .00001f, (float)(-3.f/2.f));
    }
}
__kernel void mean_delta_kernel(__global float *delta, __global float *variance, int batch, int filters, int spatial, __global float *mean_delta)
{
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (i >= filters) return;
    int j,k;
    mean_delta[i] = 0;
    for (j = 0; j < batch; ++j) {
        for (k = 0; k < spatial; ++k) {
            int index = j*filters*spatial + i*spatial + k;
            mean_delta[i] += delta[index];
        }
    }
    mean_delta[i] *= (-1.f/sqrt(variance[i] + .00001f));
}
__kernel void  mean_kernel(__global float *x, int batch, int filters, int spatial, __global float *mean)
{
    float scale = 1.f/(batch * spatial);
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (i >= filters) return;
    int j,k;
    mean[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            mean[i] += x[index];
        }
    }
    mean[i] *= scale;
}
__kernel void variance_kernel(__global float *x, __global float *mean, int batch, int filters, int spatial, __global float *variance)
{
    float scale = 1.f/(batch * spatial - 1);
    int j,k;
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (i >= filters) return;
    variance[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            variance[i] += pow((x[index] - mean[i]), 2);
        }
    }
    variance[i] *= scale;
}
__kernel void reorg_kernel(int N, __global float *x, int w, int h, int c, int batch, int stride, int forward, __global float *out)
{
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
    if(i >= N) return;
    int in_index = i;
    int in_w = i%w;
    i = i/w;
    int in_h = i%h;
    i = i/h;
    int in_c = i%c;
    i = i/c;
    int b = i%batch;
    int out_c = c/(stride*stride);
    int c2 = in_c % out_c;
    int offset = in_c / out_c;
    int w2 = in_w*stride + offset % stride;
    int h2 = in_h*stride + offset / stride;
    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));
    if(forward) out[out_index] = x[in_index];
    else out[in_index] = x[out_index];
}
__kernel void axpy_kernel(int N, float ALPHA, __global float *X, int OFFX, int INCX, __global float *Y, int OFFY, int INCY)
{
	size_t global_x = get_global_id(0);
	size_t global_y = get_global_id(1);
	size_t global_z = get_global_id(2);
	size_t i = global_z * get_global_size(0) * get_global_size(1)
		+ global_y * get_global_size(0) + global_x;
    if(i < N) Y[OFFY+i*INCY] += ALPHA*X[OFFX+i*INCX];
}
__kernel void pow_kernel(int N, float ALPHA, __global float *X, int INCX, __global float *Y, int INCY)
{
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
    if(i < N) Y[i*INCY] = pow(X[i*INCX], ALPHA);
}
__kernel void const_kernel(int N, float ALPHA, __global float *X, int INCX)
{
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
    if(i < N) X[i*INCX] = ALPHA;
}
__kernel void constrain_kernel(int N, float ALPHA, __global float *X, int INCX)
{
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
    if(i < N) X[i*INCX] = fmin(ALPHA, fmax(-ALPHA, X[i*INCX]));
}
__kernel void supp_kernel(int N, float ALPHA, __global float *X, int INCX)
{
	size_t global_x = get_global_id(0);
	size_t global_y = get_global_id(1);
	size_t global_z = get_global_id(2);
	size_t i = global_z * get_global_size(0) * get_global_size(1)
		+ global_y * get_global_size(0) + global_x;
    if(i < N) {
        if((X[i*INCX] * X[i*INCX]) < (ALPHA * ALPHA)) X[i*INCX] = 0;
    }
}
__kernel void add_kernel(int N, float ALPHA, __global float *X, int INCX)
{
	size_t global_x = get_global_id(0);
	size_t global_y = get_global_id(1);
	size_t global_z = get_global_id(2);
	size_t i = global_z * get_global_size(0) * get_global_size(1)
		+ global_y * get_global_size(0) + global_x;
    if(i < N) X[i*INCX] += ALPHA;
}
__kernel void scal_kernel(int N, float ALPHA, __global float *X, int INCX)
{
	size_t global_x = get_global_id(0);
	size_t global_y = get_global_id(1);
	size_t global_z = get_global_id(2);
	size_t i = global_z * get_global_size(0) * get_global_size(1)
		+ global_y * get_global_size(0) + global_x;
    if(i < N) X[i*INCX] *= ALPHA;
}
__kernel void fill_kernel(int N, float ALPHA, __global float *X, int INCX)
{
	size_t global_x = get_global_id(0);
	size_t global_y = get_global_id(1);
	size_t global_z = get_global_id(2);
	size_t i = global_z * get_global_size(0) * get_global_size(1) 
	               + global_y * get_global_size(0) + global_x;
	if(i < N) X[i*INCX] = ALPHA;
}
__kernel void copy_kernel(int N, __global float *X, int OFFX, int INCX, __global float *Y, int OFFY, int INCY)
{
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
    if(i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];
}
__kernel void mul_kernel(int N, __global float *X, int INCX, __global float *Y, int INCY)
{
	size_t global_x = get_global_id(0);
	size_t global_y = get_global_id(1);
	size_t global_z = get_global_id(2);
	size_t i = global_z * get_global_size(0) * get_global_size(1)
		+ global_y * get_global_size(0) + global_x;
    if(i < N) Y[i*INCY] *= X[i*INCX];
}
__kernel void  fast_mean_kernel(__global float *x, int batch, int filters, int spatial, __global float *mean)
{
	__local float part[BLOCK];
	int id = get_global_id(1);
	int filter = get_global_id(0);
	float sum = 0;
	int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += BLOCK){
            int index = j*spatial*filters + filter*spatial + i + id;
			sum += (i+id < spatial) ? x[index] : 0;
        }
    }
	part[id] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);
    if(id == 0){
        mean[filter] = 0;
        for(i = 0; i < BLOCK; ++i){
            mean[filter] += part[i];
        }
        mean[filter] /= spatial * batch;
    }
}
__kernel void  fast_variance_kernel(__global float *x, __global float *mean, int batch, int filters, int spatial, __global float *variance)
{
	__local float part[BLOCK];
	int id = get_global_id(1);
	int filter = get_global_id(0);
	float sum = 0;
	int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += BLOCK){
            int index = j*spatial*filters + filter*spatial + i + id;

			sum += (i+id < spatial) ? pow((x[index] - mean[filter]), 2) : 0;
        }
    }
	part[id] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);
    if(id == 0){
        variance[filter] = 0;
        for(i = 0; i < BLOCK; ++i){
            variance[filter] += part[i];
        }
        variance[filter] /= (spatial * batch - 1);
    }
}

__kernel void flatten_kernel(int N, __global float *x, int spatial, int layers, int batch, int forward, __global float *out)
{
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
	if (i >= N) return;
	int in_s = i % spatial;
	i = i / spatial;
	int in_c = i % layers;
	i = i / layers;
	int b = i;
	int i1 = b * layers*spatial + in_c * spatial + in_s;
	int i2 = b * layers*spatial + in_s * layers + in_c;
	if (forward) out[i2] = x[i1];
	else out[i1] = x[i2];
}
__kernel void mask_kernel(int n, __global float *x, float mask_num, __global float *mask, float scale)
{
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
	if (i < n && mask[i] == mask_num) x[i] *= scale;
}
__kernel void shortcut_kernel(int size, int minw, int minh, int minc, int stride, int samples, int batch, int w1, int h1, int c1, __global float *add, int w2, int h2, int c2, float s1, float s2, __global float *out)
{
	int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
	if (id >= size) return;
	int i = id % minw;
	id /= minw;
	int j = id % minh;
	id /= minh;
	int k = id % minc;
	id /= minc;
	int b = id % batch;
	int out_index = i * samples + w2 * (j*samples + h2 * (k + c2 * b));
	int add_index = i * stride + w1 * (j*stride + h1 * (k + c1 * b));
	out[out_index] = s1*out[out_index] + s2*add[add_index];
}
__kernel void smooth_l1_kernel(int n, __global float *pred, __global float *truth, __global float *delta, __global float *error)
{
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
	if (i < n) {
		float diff = truth[i] - pred[i];
		float abs_val = fabs(diff);
		if (abs_val < 1) {
			error[i] = diff * diff;
			delta[i] = diff;
		}
		else {
			error[i] = 2 * abs_val - 1;
			delta[i] = (diff > 0) ? 1 : -1;
		}
	}
}
__kernel void l2_kernel(int n, __global float *pred, __global float *truth, __global float *delta, __global float *error)
{
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
	if (i < n) {
		float diff = truth[i] - pred[i];
		error[i] = diff * diff;
		delta[i] = diff;
	}
}
__kernel void l1_kernel(int n, __global float *pred, __global float *truth, __global float *delta, __global float *error)
{
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
	if (i < n) {
		float diff = truth[i] - pred[i];
		error[i] = fabs(diff);
		delta[i] = (diff > 0) ? 1 : -1;
	}
}
__kernel void weighted_sum_kernel(int n, __global float *a, __global float *b, __global float *s, __global float *c)
{
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
	float m = 0.0;
	if(b != 0)
		m = b[i];
	if (i < n) {
		c[i] = s[i] * a[i] + (1 - s[i])*m;
	}
}
