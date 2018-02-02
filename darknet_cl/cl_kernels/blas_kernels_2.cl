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