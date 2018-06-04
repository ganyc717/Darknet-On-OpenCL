#ifdef GPU
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "ocl.h"

const static std::string kernel_file = "convolutional_kernels.cl";
static std::shared_ptr<CLProgram> program = NULL;
void binarize_gpu(CLArray x, int n, CLArray binary)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("binarize_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&x.buffer));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(int), (void*)&n));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&binary.buffer));

	dim2 dim = cl_gridsize(n);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void binarize_input_gpu(CLArray input, int n, int size, CLArray binary)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("binarize_input_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&input.buffer));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(int), (void*)&n));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&size));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&binary.buffer));

	dim2 dim = cl_gridsize(size);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void binarize_weights_gpu(CLArray weights, int n, int size, CLArray binary)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("binarize_weights_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&weights.buffer));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(int), (void*)&n));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&size));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&binary.buffer));

	dim2 dim = cl_gridsize(n);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void forward_convolutional_layer_gpu(convolutional_layer l, network net)
{
	fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
	if (l.binary) {
		binarize_weights_gpu(l.weights_gpu, l.n, l.c / l.groups*l.size*l.size, l.binary_weights_gpu);
		swap_binary(&l);
	}

	if (l.xnor) {
		binarize_weights_gpu(l.weights_gpu, l.n, l.c / l.groups*l.size*l.size, l.binary_weights_gpu);
		swap_binary(&l);
		binarize_gpu(net.input_gpu, l.c*l.h*l.w*l.batch, l.binary_input_gpu);
		net.input_gpu = l.binary_input_gpu;
	}

#ifdef CUDNN
	float one = 1;
	cudnnConvolutionForward(cudnn_handle(),
		&one,
		l.srcTensorDesc,
		net.input_gpu,
		l.weightDesc,
		l.weights_gpu,
		l.convDesc,
		l.fw_algo,
		net.workspace,
		l.workspace_size,
		&one,
		l.dstTensorDesc,
		l.output_gpu);

#else
	int i, j;
	int m = l.n / l.groups;
	int k = l.size*l.size*l.c / l.groups;
	int n = l.out_w*l.out_h;
	for (i = 0; i < l.batch; ++i) {
		for (j = 0; j < l.groups; ++j) {
			CLArray gpu_a = l.weights_gpu + j * l.nweights / l.groups;
			CLArray gpu_b = net.workspace_gpu;
			CLArray gpu_c = l.output_gpu + (i*l.groups + j)*n*m;
			CLArray input_gpu = net.input_gpu + (i*l.groups + j)*l.c / l.groups*l.h*l.w;

			if (l.size == 1)
				gpu_b = input_gpu;
			else
				im2col_gpu(input_gpu,l.c / l.groups, l.h, l.w, l.size, l.stride, l.pad, gpu_b);
			gemm_gpu(0, 0, m, n, k, 1, gpu_a, k, gpu_b, n, 1, gpu_c, n);
			cl_free(gpu_a);
			cl_free(gpu_c);
			cl_free(input_gpu);
		}
	}
#endif

	if (l.batch_normalize) {
		forward_batchnorm_layer_gpu(l, net);
	}
	else {
		add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
	}

	activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
	if (l.binary || l.xnor) swap_binary(&l);
}

void smooth_layer(layer l, int size, float rate)
{
	int h = l.out_h;
	int w = l.out_w;
	int c = l.out_c;

	size_t n = h * w*c*l.batch;

	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("smooth_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&l.output_gpu.buffer));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(int), (void*)&n));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&l.w));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&l.h));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&l.c));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(int), (void*)&size));
	cl->checkError(clSetKernelArg(kernel, 6, sizeof(float), (void*)&rate));
	cl->checkError(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&l.delta_gpu.buffer));

	dim2 dim = cl_gridsize(n);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void backward_convolutional_layer_gpu(convolutional_layer l, network net)
{
	if (l.smooth) {
		smooth_layer(l, 5, l.smooth);
	}
	gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);


	if (l.batch_normalize) {
		backward_batchnorm_layer_gpu(l, net);
	}
	else {
		backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
	}
	CLArray original_input = net.input_gpu;

	if (l.xnor) net.input_gpu = l.binary_input_gpu;
#ifdef CUDNN
	float one = 1;
	cudnnConvolutionBackwardFilter(cudnn_handle(),
		&one,
		l.srcTensorDesc,
		net.input_gpu,
		l.ddstTensorDesc,
		l.delta_gpu,
		l.convDesc,
		l.bf_algo,
		net.workspace,
		l.workspace_size,
		&one,
		l.dweightDesc,
		l.weight_updates_gpu);

	if (net.delta_gpu) {
		if (l.binary || l.xnor) swap_binary(&l);
		cudnnConvolutionBackwardData(cudnn_handle(),
			&one,
			l.weightDesc,
			l.weights_gpu,
			l.ddstTensorDesc,
			l.delta_gpu,
			l.convDesc,
			l.bd_algo,
			net.workspace,
			l.workspace_size,
			&one,
			l.dsrcTensorDesc,
			net.delta_gpu);
		if (l.binary || l.xnor) swap_binary(&l);
		if (l.xnor) gradient_array_gpu(original_input, l.batch*l.c*l.h*l.w, HARDTAN, net.delta_gpu);
	}

#else
	int m = l.n / l.groups;
	int n = l.size*l.size*l.c / l.groups;
	int k = l.out_w*l.out_h;

	int i, j;
	for (i = 0; i < l.batch; ++i) {
		CLArray original_input_gpu = original_input + i * l.c*l.h*l.w;

		for (j = 0; j < l.groups; ++j) {
			CLArray gpu_a = l.delta_gpu + (i*l.groups + j)*m*k;
			CLArray gpu_b = net.workspace_gpu;
			CLArray gpu_c = l.weight_updates_gpu + j * l.nweights / l.groups;
			CLArray im = net.input_gpu + (i*l.groups + j)*l.c / l.groups*l.h*l.w;
			im2col_gpu(im, l.c / l.groups, l.h, l.w,
				l.size, l.stride, l.pad, gpu_b);
			gemm_gpu(0, 1, m, n, k, 1, gpu_a, k, gpu_b, k, 1, gpu_c, n);

			cl_free(gpu_a);
			cl_free(gpu_c);
			cl_free(im);

			if (net.delta_gpu.size > 0 && net.delta_gpu.buffer) {
				if (l.binary || l.xnor) swap_binary(&l);

				CLArray a = l.weights_gpu + j * l.nweights / l.groups;
				CLArray b = l.delta_gpu + (i*l.groups + j)*m*k;
				CLArray c = net.workspace_gpu;
				CLArray img = net.delta_gpu + (i*l.groups + j)*l.c / l.groups*l.h*l.w;

				if (l.size == 1)
					c = img;
				gemm_gpu(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);
				if (l.size != 1)
					col2im_gpu(net.workspace_gpu, l.c / l.groups, l.h, l.w, l.size, l.stride,
						l.pad, img);

				cl_free(a);
				cl_free(b);
				cl_free(img);

				if (l.binary || l.xnor) {
					swap_binary(&l);
				}
			}

			if (l.xnor)
			{
				CLArray delta_gpu = net.delta_gpu + i * l.c*l.h*l.w;
				gradient_array_gpu(original_input_gpu, l.c*l.h*l.w, HARDTAN, delta_gpu);
				cl_free(delta_gpu);
			}
			cl_free(original_input_gpu);
		}
	}
#endif
}

void pull_convolutional_layer(layer l)
{
	cl_pull_array(l.weights_gpu, l.weights, l.nweights);
	cl_pull_array(l.biases_gpu, l.biases, l.n);
	cl_pull_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
	cl_pull_array(l.bias_updates_gpu, l.bias_updates, l.n);
	if (l.batch_normalize) {
		cl_pull_array(l.scales_gpu, l.scales, l.n);
		cl_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
		cl_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
	}
}

void push_convolutional_layer(layer l)
{
	cl_push_array(l.weights_gpu, l.weights, l.nweights);
	cl_push_array(l.biases_gpu, l.biases, l.n);
	cl_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
	cl_push_array(l.bias_updates_gpu, l.bias_updates, l.n);
	if (l.batch_normalize) {
		cl_push_array(l.scales_gpu, l.scales, l.n);
		cl_push_array(l.rolling_mean_gpu, l.rolling_mean, l.n);
		cl_push_array(l.rolling_variance_gpu, l.rolling_variance, l.n);
	}
}

void update_convolutional_layer_gpu(layer l, update_args a)
{
	float learning_rate = a.learning_rate*l.learning_rate_scale;
	float momentum = a.momentum;
	float decay = a.decay;
	int batch = a.batch;

	if (a.adam) {
		adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.nweights, batch, a.t);
		adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
		if (l.scales_gpu.buffer && l.scales_gpu.size > 0) {
			adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
		}
	}
	else {
		axpy_gpu(l.nweights, -decay * batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
		axpy_gpu(l.nweights, learning_rate / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
		scal_gpu(l.nweights, momentum, l.weight_updates_gpu, 1);

		axpy_gpu(l.n, learning_rate / batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
		scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);

		if (l.scales_gpu.buffer && l.scales_gpu.size > 0) {
			axpy_gpu(l.n, learning_rate / batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
			scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
		}
	}
	if(l.clip)
		constrain_gpu(l.nweights, l.clip, l.weights_gpu, 1);
}

#endif
