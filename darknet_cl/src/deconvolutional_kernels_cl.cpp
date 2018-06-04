#ifdef GPU
#include "convolutional_layer.h"
#include "deconvolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "ocl.h"

void forward_deconvolutional_layer_gpu(layer l, network net)
{
	int i;

	int m = l.size*l.size*l.n;
	int n = l.h*l.w;
	int k = l.c;

	fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);

	for (i = 0; i < l.batch; ++i) {
		CLArray gpu_a = l.weights_gpu;
		CLArray gpu_b = net.input_gpu + i * l.c*l.h*l.w;
		CLArray gpu_c = net.workspace_gpu;
		CLArray output_gpu = l.output_gpu + i * l.outputs;

		gemm_gpu(1, 0, m, n, k, 1, gpu_a, m, gpu_b, n, 0, gpu_c, n);
		col2im_gpu(net.workspace_gpu, l.out_c, l.out_h, l.out_w, l.size, l.stride, l.pad, output_gpu);
		cl_free(gpu_b);
		cl_free(output_gpu);
	}
	if (l.batch_normalize) {
		forward_batchnorm_layer_gpu(l, net);
	}
	else {
		add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
	}
	activate_array_gpu(l.output_gpu, l.batch*l.n*l.out_w*l.out_h, l.activation);
}

void backward_deconvolutional_layer_gpu(layer l, network net)
{
	int i;

	//constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
	gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);

	if (l.batch_normalize) {
		backward_batchnorm_layer_gpu(l, net);
	}
	else {
		backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
	}

	for (i = 0; i < l.batch; ++i) {
		int m = l.c;
		int n = l.size*l.size*l.n;
		int k = l.h*l.w;

		CLArray gpu_a = net.input_gpu + i * m*k;
		CLArray gpu_b = net.workspace_gpu;
		CLArray gpu_c = l.weight_updates_gpu;
		CLArray delta_gpu = l.delta_gpu + i * l.outputs;


		im2col_gpu(delta_gpu, l.out_c, l.out_h, l.out_w,
			l.size, l.stride, l.pad, gpu_b);
		gemm_gpu(0, 1, m, n, k, 1, gpu_a, k, gpu_b, k, 1, gpu_c, n);

		cl_free(gpu_a);
		cl_free(delta_gpu);

		if (net.delta_gpu.buffer && net.delta_gpu.size > 0) {
			int m = l.c;
			int n = l.h*l.w;
			int k = l.size*l.size*l.n;

			CLArray gpu_a = l.weights_gpu;
			CLArray gpu_b = net.workspace_gpu;
			CLArray gpu_c = net.delta_gpu + i * n*m;

			gemm_gpu(0, 0, m, n, k, 1, gpu_a, k, gpu_b, n, 1, gpu_c, n);
			cl_free(gpu_c);
		}
	}
}

void pull_deconvolutional_layer(layer l)
{
	cl_pull_array(l.weights_gpu, l.weights, l.c*l.n*l.size*l.size);
	cl_pull_array(l.biases_gpu, l.biases, l.n);
	cl_pull_array(l.weight_updates_gpu, l.weight_updates, l.c*l.n*l.size*l.size);
	cl_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
	if (l.batch_normalize) {
		cl_pull_array(l.scales_gpu, l.scales, l.n);
		cl_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
		cl_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
	}
}

void push_deconvolutional_layer(layer l)
{
	cl_push_array(l.weights_gpu, l.weights, l.c*l.n*l.size*l.size);
	cl_push_array(l.biases_gpu, l.biases, l.n);
	cl_push_array(l.weight_updates_gpu, l.weight_updates, l.c*l.n*l.size*l.size);
	cl_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
	if (l.batch_normalize) {
		cl_push_array(l.scales_gpu, l.scales, l.n);
		cl_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
		cl_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
	}
}

void update_deconvolutional_layer_gpu(layer l, update_args a)
{
	float learning_rate = a.learning_rate*l.learning_rate_scale;
	float momentum = a.momentum;
	float decay = a.decay;
	int batch = a.batch;

	//int size = l.size*l.size*l.c*l.n;

	if (a.adam) {
		adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
		adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
		if (l.scales_gpu.buffer && l.scales_gpu.size > 0) {
			adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
		}
	}
	else {
		//axpy_gpu(size, -decay * batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
		//axpy_gpu(size, learning_rate / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
		//scal_gpu(size, momentum, l.weight_updates_gpu, 1);
		axpy_gpu(l.nweights, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
		axpy_gpu(l.nweights, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
		scal_gpu(l.nweights, momentum, l.weight_updates_gpu, 1);

		axpy_gpu(l.n, learning_rate / batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
		scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);

		if (l.scales_gpu.buffer && l.scales_gpu.size > 0) {
			axpy_gpu(l.n, learning_rate / batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
			scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
		}
	}
}

#endif
