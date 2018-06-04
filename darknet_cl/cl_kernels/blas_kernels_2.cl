__kernel void deinter_kernel(int NX, __global float *X, int NY, __global float *Y, int B, __global float *OUT_)
{
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
	if (i < (NX + NY)*B) {
		int b = i / (NX + NY);
		int j = i % (NX + NY);
		if (j < NX) {
			if (X) X[b*NX + j] += OUT_[i];
		}
		else {
			if (Y) Y[b*NY + j - NX] += OUT_[i];
		}
	}
}
__kernel void inter_kernel(int NX, __global float *X, int NY, __global float *Y, int B, __global float *OUT_)
{
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
	if (i < (NX + NY)*B) {
		int b = i / (NX + NY);
		int j = i % (NX + NY);
		if (j < NX) {
			OUT_[i] = X[b*NX + j];
		}
		else {
			OUT_[i] = Y[b*NY + j - NX];
		}
	}
}
__kernel void weighted_delta_kernel(int n, __global float *a, __global float *b, __global float *s, __global float *da, __global float *db, __global float *ds, __global float *dc)
{
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
	if (i < n) {
		if (da) da[i] += dc[i] * s[i];
		if (db) db[i] += dc[i] * (1 - s[i]);
		ds[i] += dc[i] * (a[i] - b[i]);
	}
}
__kernel void mult_add_into_kernel(int n, __global float *a, __global float *b, __global float *c)
{
	int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
	if (i < n) {
		c[i] += a[i] * b[i];
	}
}
void softmax_device(__global float *input, int n, float temp, int stride, __global float *output)
{
	int i;
	float sum = 0;
	float largest = -INFINITY;
	for (i = 0; i < n; ++i) {
		int val = input[i*stride];
		largest = (val>largest) ? val : largest;
	}
	for (i = 0; i < n; ++i) {
		float e = exp(input[i*stride] / temp - largest / temp);
		sum += e;
		output[i*stride] = e;
	}
	for (i = 0; i < n; ++i) {
		output[i*stride] /= sum;
	}
}
__kernel void softmax_tree_kernel(__global float *input, int spatial, int batch, int stride, float temp, __global float *output, int groups, __constant int *group_size, __constant int *group_offset)
{
	int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
	if (id >= spatial * batch*groups) return;
	int s = id % spatial;
	id = id / spatial;
	int g = id % groups;
	int b = id / groups;
	int goff = group_offset[g] * spatial;
	int boff = b * stride;
	softmax_device(input + goff + boff + s, group_size[g], temp, spatial, output + goff + boff + s);
}
__kernel void softmax_kernel(__global float *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, __global float *output)
{
	int id = get_global_id(2) * get_global_size(0) * get_global_size(1) +
		get_global_id(1) * get_global_size(0) + get_global_id(0);
	if (id >= batch * groups) return;
	int b = id / groups;
	int g = id % groups;
	softmax_device(input + b * batch_offset + g * group_offset, n, temp, stride, output + b * batch_offset + g * group_offset);
}
__kernel void l2norm_kernel(int N, __global float *x, __global float *dx, int batch, int filters, int spatial)
{
    int index = get_global_id(2) * get_global_size(0) * get_global_size(1) +
                    get_global_id(1) * get_global_size(0) + get_global_id(0);
    if (index >= N) return;
    int b = index / spatial;
    int i = index % spatial;
    int f;
    float sum = 0;
    for(f = 0; f < filters; ++f){
        int index = b*filters*spatial + f*spatial + i;
        sum += pow(x[index], 2);
    }
    sum = sqrt(sum);
    if(sum == 0) sum = 1;
    for(f = 0; f < filters; ++f){
        int index = b*filters*spatial + f*spatial + i;
        x[index] /= sum;
        dx[index] = (1 - x[index]) / sum;
    }
}
__kernel void scale_mask_kernel(int n, __global float *x, float mask_num, __global float *mask, float scale)
{
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
                        get_global_id(1) * get_global_size(0) + get_global_id(0);
    if(i < n && mask[i] == mask_num) x[i] *= scale;
}
__kernel void softmax_x_ent_kernel(int n, __global float *pred, __global float *truth, __global float *delta, __global float *error)
{
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
                            get_global_id(1) * get_global_size(0) + get_global_id(0);
    if(i < n){
        float t = truth[i];
        float p = pred[i];
        error[i] = (t) ? -log(p) : 0;
        delta[i] = t-p;
    }
}
__kernel void logistic_x_ent_kernel(int n, __global float *pred, __global float *truth, __global float *delta, __global float *error)
{
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
                            get_global_id(1) * get_global_size(0) + get_global_id(0);
    if(i < n){
        float t = truth[i];
        float p = pred[i];
        error[i] = -t*log(p+.0000001) - (1-t)*log(1-p+.0000001);
        delta[i] = t-p;
    }
}
__kernel void wgan_kernel(int n, __global float *pred, __global float *truth, __global float *delta, __global float *error)
{
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
                                get_global_id(1) * get_global_size(0) + get_global_id(0);
    if(i < n){
        error[i] = truth[i] ? -pred[i] : pred[i];
        delta[i] = (truth[i] > 0) ? 1 : -1;
    }
}
void atomic_float_add(volatile __global float *addr, float v)
{
    volatile __global int *p = (volatile __global int *)addr;
    int last_value;
    float result;
    do
    {
        last_value = *p;
        result = v + as_float(last_value);
    }while(atomic_cmpxchg(p, last_value, as_int(result)) != last_value);
}
__kernel void upsample_kernel(int N, __global float *x, int w, int h, int c, int batch, int stride, int forward, float scale, __global float *out)
{
    int i = get_global_id(2) * get_global_size(0) * get_global_size(1) +
                                    get_global_id(1) * get_global_size(0) + get_global_id(0);
    if(i >= N) return;
    int out_index = i;
    int out_w = i%(w*stride);
    i = i/(w*stride);
    int out_h = i%(h*stride);
    i = i/(h*stride);
    int out_c = i%c;
    i = i/c;
    int b = i%batch;
    int in_w = out_w / stride;
    int in_h = out_h / stride;
    int in_c = out_c;
    int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;
    if(forward) out[out_index] += scale * x[in_index];
    else atomic_float_add(x+in_index, scale * out[out_index]);
}
