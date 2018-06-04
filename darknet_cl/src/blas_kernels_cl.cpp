#ifdef GPU

#include "ocl.h"
#include "blas.h"
#include "utils.h"
const static std::string kernel_file = "blas_kernels_1.cl";
static std::shared_ptr<CLProgram> program = NULL;

//Visual Studio compiler has limitaion on string size, so that I need to split this file into 2 parts

const static std::string kernel_file_2 = "blas_kernels_2.cl";
static std::shared_ptr<CLProgram> program_2 = NULL;

void scale_bias_gpu(CLArray output, CLArray biases, int batch, int n, int size)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("scale_bias_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&output.buffer));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&biases.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&n));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&size));

	size_t global_size[] = { BLOCK ,(size - 1) / BLOCK + 1 ,n };
	cl_event* events = new cl_event[batch];

	for (int i = 0; i < batch; i++)
	{
		size_t offset[] = { 0,0,i * n };
		cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, offset, global_size, NULL, NULL, NULL, &events[i]);
		cl->checkError(error);
	}

	cl->checkError(clWaitForEvents(batch, events));
	for (int i = 0; i < batch; i++)
		clReleaseEvent(events[i]);
	delete[] events;
}

void backward_scale_gpu(CLArray x_norm, CLArray delta, int batch, int n, int size, CLArray scale_updates)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("backward_scale_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&x_norm.buffer));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&delta.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&batch));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&n));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&size));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&scale_updates.buffer));

	size_t global_size[] = { n,BLOCK };
	size_t group_size[] = { 1,BLOCK };
	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 2, NULL, global_size, group_size, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void add_bias_gpu(CLArray output, CLArray biases, int batch, int n, int size)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("add_bias_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&output.buffer));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&biases.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&batch));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&n));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&size));


	int num = n * size*batch;
	dim2 dim = cl_gridsize(num);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void backward_bias_gpu(CLArray bias_updates, CLArray delta, int batch, int n, int size)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_event e;
	if (size == 1) {
		cl_kernel kernel = program->getKernel("backward_bias_conn_kernel");

		cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bias_updates.buffer));
		cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&delta.buffer));
		cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&batch));
		cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&n));

		int num = n * size*batch;
		dim2 dim = cl_gridsize(num);
		size_t global_size[] = { dim.x,dim.y,BLOCK };
		cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
		cl->checkError(error);
	}
	else {
		cl_kernel kernel = program->getKernel("backward_bias_kernel");

		cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&bias_updates.buffer));
		cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&delta.buffer));
		cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&batch));
		cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&n));
		cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&size));

		size_t global_size[] = { n,BLOCK };
		size_t group_size[] = { 1,BLOCK };
		cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 2, NULL, global_size, group_size, NULL, NULL, &e);
		cl->checkError(error);
	}
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void adam_gpu(int n, CLArray x, CLArray m, CLArray v, float B1, float B2, float rate, float eps, int t)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("adam_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&n));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&x.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&m.buffer));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&v.buffer));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(float), (void*)&B1));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(float), (void*)&B1));
	cl->checkError(clSetKernelArg(kernel, 6, sizeof(float), (void*)&rate));
	cl->checkError(clSetKernelArg(kernel, 7, sizeof(float), (void*)&eps));
	cl->checkError(clSetKernelArg(kernel, 8, sizeof(int), (void*)&t));

	dim2 dim = cl_gridsize(n);
	size_t global_size[] = { dim.x,dim.y,BLOCK };
	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);

	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void adam_update_gpu(CLArray w, CLArray d, CLArray m, CLArray v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t)
{
	scal_gpu(n, B1, m, 1);
	scal_gpu(n, B2, v, 1);
	axpy_gpu(n, -decay * batch, w, 1, d, 1);

	axpy_gpu(n, (1 - B1), d, 1, m, 1);
	mul_gpu(n, d, 1, d, 1);
	axpy_gpu(n, (1 - B2), d, 1, v, 1);

	adam_gpu(n, w, m, v, B1, B2, rate, eps, t);
	fill_gpu(n, 0, d, 1);
}

void normalize_delta_gpu(CLArray x, CLArray mean, CLArray variance, CLArray mean_delta, CLArray variance_delta, int batch, int filters, int spatial, CLArray delta)
{
	size_t N = batch * filters*spatial;
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("normalize_delta_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&N));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&x.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&mean.buffer));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&variance.buffer));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&mean_delta.buffer));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&variance_delta.buffer));
	cl->checkError(clSetKernelArg(kernel, 6, sizeof(int), (void*)&batch));
	cl->checkError(clSetKernelArg(kernel, 7, sizeof(int), (void*)&filters));
	cl->checkError(clSetKernelArg(kernel, 8, sizeof(int), (void*)&spatial));
	cl->checkError(clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&delta.buffer));

	dim2 dim = cl_gridsize(N);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void mean_delta_gpu(CLArray delta, CLArray variance, int batch, int filters, int spatial, CLArray mean_delta)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("mean_delta_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&delta.buffer));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&variance.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&batch));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&filters));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&spatial));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&mean_delta.buffer));

	dim2 dim = cl_gridsize(filters);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void fast_mean_delta_gpu(CLArray delta, CLArray variance, int batch, int filters, int spatial, CLArray mean_delta)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("fast_mean_delta_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&delta.buffer));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&variance.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&batch));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&filters));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&spatial));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&mean_delta.buffer));

	size_t global_size[] = { filters,BLOCK };
	size_t group_size[] = { 1,BLOCK };
	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 2, NULL, global_size, group_size, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void fast_variance_delta_gpu(CLArray x, CLArray delta, CLArray mean, CLArray variance, int batch, int filters, int spatial, CLArray variance_delta)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("fast_variance_delta_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&x.buffer));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&delta.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&mean.buffer));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&variance.buffer));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&batch));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(int), (void*)&filters));
	cl->checkError(clSetKernelArg(kernel, 6, sizeof(int), (void*)&spatial));
	cl->checkError(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&variance_delta.buffer));

	size_t global_size[] = { filters,BLOCK };
	size_t group_size[] = { 1,BLOCK };
	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 2, NULL, global_size, group_size, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void normalize_gpu(CLArray x, CLArray mean, CLArray variance, int batch, int filters, int spatial)
{
	size_t N = batch * filters*spatial;

	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("normalize_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&N));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&x.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&mean.buffer));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&variance.buffer));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&batch));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(int), (void*)&filters));
	cl->checkError(clSetKernelArg(kernel, 6, sizeof(int), (void*)&spatial));

	dim2 dim = cl_gridsize(N);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void fast_mean_gpu(CLArray x, int batch, int filters, int spatial, CLArray mean)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("fast_mean_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&x.buffer));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(int), (void*)&batch));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&filters));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&spatial));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&mean.buffer));

	size_t global_size[] = { filters,BLOCK };
	size_t group_size[] = { 1,BLOCK };
	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 2, NULL, global_size, group_size, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void fast_variance_gpu(CLArray x, CLArray mean, int batch, int filters, int spatial, CLArray variance)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("fast_variance_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&x.buffer));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mean.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&batch));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&filters));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&spatial));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&variance.buffer));

	size_t global_size[] = { filters,BLOCK };
	size_t group_size[] = { 1,BLOCK };
	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 2, NULL, global_size, group_size, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void mean_gpu(CLArray x, int batch, int filters, int spatial, CLArray mean)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("mean_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&x.buffer));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(int), (void*)&batch));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&filters));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&spatial));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&mean.buffer));

	dim2 dim = cl_gridsize(filters);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void variance_gpu(CLArray x, CLArray mean, int batch, int filters, int spatial, CLArray variance)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("variance_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&x.buffer));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mean.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&batch));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&filters));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&spatial));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&variance.buffer));

	dim2 dim = cl_gridsize(filters);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void axpy_gpu(int N, float ALPHA, CLArray X, int INCX, CLArray Y, int INCY)
{
	axpy_gpu_offset(N, ALPHA, X, 0, INCX, Y, 0, INCY);
}

void pow_gpu(int N, float ALPHA, CLArray X, int INCX, CLArray Y, int INCY)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("pow_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&N));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(float), (void*)&ALPHA));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&X.buffer));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&INCX));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&Y.buffer));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(int), (void*)&INCY));

	dim2 dim = cl_gridsize(N);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void axpy_gpu_offset(int N, float ALPHA, CLArray X, int OFFX, int INCX, CLArray Y, int OFFY, int INCY)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("axpy_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&N));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(float), (void*)&ALPHA));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&X.buffer));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&OFFX));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&INCX));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&Y.buffer));
	cl->checkError(clSetKernelArg(kernel, 6, sizeof(int), (void*)&OFFY));
	cl->checkError(clSetKernelArg(kernel, 7, sizeof(int), (void*)&INCY));

	dim2 dim = cl_gridsize(N);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void copy_gpu(int N, CLArray X, int INCX, CLArray Y, int INCY)
{
	copy_gpu_offset(N, X, 0, INCX, Y, 0, INCY);
}

void mul_gpu(int N, CLArray X, int INCX, CLArray Y, int INCY)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("mul_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&N));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&X.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&INCX));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&Y.buffer));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&INCY));

	dim2 dim = cl_gridsize(N);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void copy_gpu_offset(int N, CLArray X, int OFFX, int INCX, CLArray Y, int OFFY, int INCY)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("copy_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&N));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&X.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&OFFX));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&INCX));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&Y.buffer));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(int), (void*)&OFFY));
	cl->checkError(clSetKernelArg(kernel, 6, sizeof(int), (void*)&INCY));

	dim2 dim = cl_gridsize(N);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void flatten_gpu(CLArray x, int spatial, int layers, int batch, int forward, CLArray out)
{
	int size = spatial * batch*layers;
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("flatten_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&size));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&x.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&spatial));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&layers));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&batch));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(int), (void*)&forward));
	cl->checkError(clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&out.buffer));

	dim2 dim = cl_gridsize(size);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void reorg_gpu(CLArray x, int w, int h, int c, int batch, int stride, int forward, CLArray out)
{
	int size = w * h*c*batch;
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("reorg_kernel");
	
	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&size));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&x.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&w));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&h));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&c));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(int), (void*)&batch));
	cl->checkError(clSetKernelArg(kernel, 6, sizeof(int), (void*)&stride));
	cl->checkError(clSetKernelArg(kernel, 7, sizeof(int), (void*)&forward));
	cl->checkError(clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&out.buffer));

	dim2 dim = cl_gridsize(size);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void mask_gpu(int N, CLArray X, float mask_num, CLArray mask, float scale)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("mask_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&N));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&X.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(float), (void*)&mask_num));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&mask.buffer));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(float), (void*)&scale));

	dim2 dim = cl_gridsize(N);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void mask_gpu(int N, CLArray X, float mask_num, CLArray mask)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("mask_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&N));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&X.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(float), (void*)&mask_num));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&mask.buffer));

	dim2 dim = cl_gridsize(N);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void const_gpu(int N, float ALPHA, CLArray X, int INCX)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("const_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&N));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(float), (void*)&ALPHA));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&X.buffer));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&INCX));

	dim2 dim = cl_gridsize(N);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void constrain_gpu(int N, float ALPHA, CLArray X, int INCX)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("constrain_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&N));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(float), (void*)&ALPHA));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&X.buffer));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&INCX));

	dim2 dim = cl_gridsize(N);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void add_gpu(int N, float ALPHA, CLArray X, int INCX)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("add_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&N));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(float), (void*)&ALPHA));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&X.buffer));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&INCX));

	dim2 dim = cl_gridsize(N);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void scal_gpu(int N, float ALPHA, CLArray X, int INCX)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("scal_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&N));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(float), (void*)&ALPHA));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&X.buffer));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&INCX));

	dim2 dim = cl_gridsize(N);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void supp_gpu(int N, float ALPHA, CLArray X, int INCX)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("supp_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&N));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(float), (void*)&ALPHA));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&X.buffer));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&INCX));

	dim2 work_size = cl_gridsize(N);
	size_t global_size[] = { work_size.x,work_size.y,BLOCK };
	size_t offset[] = { 0,0,0 };
	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, offset, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);

	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void fill_gpu(int N, float ALPHA, CLArray X, int INCX)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("fill_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&N));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(float), (void*)&ALPHA));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&X.buffer));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&INCX));

	dim2 work_size = cl_gridsize(N);
	size_t global_size[] = { work_size.x,work_size.y,BLOCK };
	size_t offset[] = { 0,0,0 };
	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, offset, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);

	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void shortcut_gpu(int batch, int w1, int h1, int c1, CLArray add, int w2, int h2, int c2, float s1, float s2, CLArray out)
{
	int minw = (w1 < w2) ? w1 : w2;
	int minh = (h1 < h2) ? h1 : h2;
	int minc = (c1 < c2) ? c1 : c2;

	int stride = w1 / w2;
	int sample = w2 / w1;
	assert(stride == h1 / h2);
	assert(sample == h2 / h1);
	if (stride < 1) stride = 1;
	if (sample < 1) sample = 1;

	int size = batch * minw * minh * minc;
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("shortcut_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&size));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(int), (void*)&minw));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&minh));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&minc));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&stride));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(int), (void*)&sample));
	cl->checkError(clSetKernelArg(kernel, 6, sizeof(int), (void*)&batch));
	cl->checkError(clSetKernelArg(kernel, 7, sizeof(int), (void*)&w1));
	cl->checkError(clSetKernelArg(kernel, 8, sizeof(int), (void*)&h1));
	cl->checkError(clSetKernelArg(kernel, 9, sizeof(int), (void*)&c1));
	cl->checkError(clSetKernelArg(kernel, 10, sizeof(cl_mem), (void*)&add.buffer));
	cl->checkError(clSetKernelArg(kernel, 11, sizeof(int), (void*)&w2));
	cl->checkError(clSetKernelArg(kernel, 12, sizeof(int), (void*)&h2));
	cl->checkError(clSetKernelArg(kernel, 13, sizeof(int), (void*)&c2));
	cl->checkError(clSetKernelArg(kernel, 14, sizeof(float), (void*)&s1));
	cl->checkError(clSetKernelArg(kernel, 15, sizeof(float), (void*)&s2));
	cl->checkError(clSetKernelArg(kernel, 16, sizeof(cl_mem), (void*)&out.buffer));

	dim2 dim = cl_gridsize(size);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void smooth_l1_gpu(int n, CLArray pred, CLArray truth, CLArray delta, CLArray error)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("smooth_l1_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&n));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&pred.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&truth.buffer));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&delta.buffer));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&error.buffer));

	dim2 dim = cl_gridsize(n);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int status = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(status);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void l2_gpu(int n, CLArray pred, CLArray truth, CLArray delta, CLArray error)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("l2_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&n));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&pred.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&truth.buffer));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&delta.buffer));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&error.buffer));

	dim2 dim = cl_gridsize(n);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int status = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(status);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void l1_gpu(int n, CLArray pred, CLArray truth, CLArray delta, CLArray error)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("l1_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&n));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&pred.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&truth.buffer));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&delta.buffer));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&error.buffer));

	dim2 dim = cl_gridsize(n);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int status = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(status);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void deinter_gpu(int NX, CLArray X, int NY, CLArray Y, int B, CLArray OUT_)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program_2 == NULL)
		program_2 = cl->buildProgramFromFile(kernel_file_2, "");
	cl_kernel kernel = program_2->getKernel("deinter_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&NX));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&X.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&NY));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&Y.buffer));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&B));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&OUT_.buffer));

	dim2 dim = cl_gridsize((NX + NY)*B);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int status = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(status);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void inter_gpu(int NX, CLArray X, int NY, CLArray Y, int B, CLArray OUT_)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program_2 == NULL)
		program_2 = cl->buildProgramFromFile(kernel_file_2, "");
	cl_kernel kernel = program_2->getKernel("inter_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&NX));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&X.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&NY));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&Y.buffer));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&B));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&OUT_.buffer));

	dim2 dim = cl_gridsize((NX + NY)*B);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int status = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(status);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void weighted_sum_gpu(CLArray a, CLArray b, CLArray s, int num, CLArray c)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("weighted_sum_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&num));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&a.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&b.buffer));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&s.buffer));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&c.buffer));

	dim2 dim = cl_gridsize(num);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int status = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(status);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void weighted_delta_gpu(CLArray a, CLArray b, CLArray s, CLArray da, CLArray db, CLArray ds, int num, CLArray dc)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program_2 == NULL)
		program_2 = cl->buildProgramFromFile(kernel_file_2, "");
	cl_kernel kernel = program_2->getKernel("weighted_delta_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&num));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&a.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&b.buffer));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&s.buffer));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&da.buffer));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&db.buffer));
	cl->checkError(clSetKernelArg(kernel, 6, sizeof(cl_mem), (void*)&ds.buffer));
	cl->checkError(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&dc.buffer));

	dim2 dim = cl_gridsize(num);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int status = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(status);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void mult_add_into_gpu(int num, CLArray a, CLArray b, CLArray c)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program_2 == NULL)
		program_2 = cl->buildProgramFromFile(kernel_file_2, "");
	cl_kernel kernel = program_2->getKernel("mult_add_into_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&num));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&a.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&b.buffer));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&c.buffer));

	dim2 dim = cl_gridsize(num);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int status = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(status);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void softmax_tree(CLArray input, int spatial, int batch, int stride, float temp, CLArray output, tree hier)
{
	CLArray tree_groups_size = cl_make_int_array(hier.group_size, hier.groups);
	CLArray tree_groups_offset = cl_make_int_array(hier.group_offset, hier.groups);
    
	int num = spatial * batch*hier.groups;

	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program_2 == NULL)
		program_2 = cl->buildProgramFromFile(kernel_file_2, "");
	cl_kernel kernel = program_2->getKernel("softmax_tree_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&input.buffer));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(int), (void*)&spatial));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&batch));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&stride));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(float), (void*)&temp));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&output.buffer));
	cl->checkError(clSetKernelArg(kernel, 6, sizeof(int), (void*)&hier.groups));
	cl->checkError(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&tree_groups_size.buffer));
	cl->checkError(clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&tree_groups_offset.buffer));

	dim2 dim = cl_gridsize(num);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int status = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(status);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
	cl_free(tree_groups_size);
	cl_free(tree_groups_offset);
}

void softmax_gpu(CLArray input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, CLArray output)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program_2 == NULL)
		program_2 = cl->buildProgramFromFile(kernel_file_2, "");
	cl_kernel kernel = program_2->getKernel("softmax_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&input.buffer));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(int), (void*)&n));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&batch));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&batch_offset));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&groups));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(int), (void*)&group_offset));
	cl->checkError(clSetKernelArg(kernel, 6, sizeof(int), (void*)&stride));
	cl->checkError(clSetKernelArg(kernel, 7, sizeof(float), (void*)&temp));
	cl->checkError(clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&output.buffer));

	dim2 dim = cl_gridsize(batch*groups);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int status = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(status);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void l2normalize_gpu(CLArray x, CLArray dx, int batch, int filters, int spatial)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program_2 == NULL)
	program_2 = cl->buildProgramFromFile(kernel_file_2, "");
	cl_kernel kernel = program_2->getKernel("l2normalize_gpu");

	int N = batch*spatial;
	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&N));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&x.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&dx.buffer));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&batch));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&filters));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(int), (void*)&spatial));

	dim2 dim = cl_gridsize(N);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int status = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(status);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void scale_mask_gpu(int N, CLArray X, float mask_num, CLArray mask, float scale)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program_2 == NULL)
	program_2 = cl->buildProgramFromFile(kernel_file_2, "");
	cl_kernel kernel = program_2->getKernel("scale_mask_kernel");
	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&N));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&X.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(float), (void*)&mask_num));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&mask.buffer));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(float), (void*)&scale));

	dim2 dim = cl_gridsize(N);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int status = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(status);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void softmax_x_ent_gpu(int n, CLArray pred, CLArray truth, CLArray delta, CLArray error)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program_2 == NULL)
	program_2 = cl->buildProgramFromFile(kernel_file_2, "");
	cl_kernel kernel = program_2->getKernel("softmax_x_ent_kernel");
	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&n));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&pred.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&truth.buffer));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&delta.buffer));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&error.buffer));

	dim2 dim = cl_gridsize(n);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int status = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(status);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void logistic_x_ent_gpu(int n, CLArray pred, CLArray truth, CLArray delta, CLArray error)
{
        std::shared_ptr<CLWarpper> cl = getCLWarpper();
        if (program_2 == NULL)
        program_2 = cl->buildProgramFromFile(kernel_file_2, "");
        cl_kernel kernel = program_2->getKernel("logistic_x_ent_kernel");
        cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&n));
        cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&pred.buffer));
        cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&truth.buffer));
        cl->checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&delta.buffer));
        cl->checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&error.buffer));

        dim2 dim = cl_gridsize(n);
        size_t global_size[] = { dim.x,dim.y,BLOCK };

        cl_event e;
        cl_int status = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
        cl->checkError(status);
        cl->checkError(clWaitForEvents(1, &e));
        clReleaseEvent(e);
}

void wgan_gpu(int n, CLArray pred, CLArray truth, CLArray delta, CLArray error)
{
        std::shared_ptr<CLWarpper> cl = getCLWarpper();
        if (program_2 == NULL)
        program_2 = cl->buildProgramFromFile(kernel_file_2, "");
        cl_kernel kernel = program_2->getKernel("wgan_kernel");
        cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&n));
        cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&pred.buffer));
        cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&truth.buffer));
        cl->checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&delta.buffer));
        cl->checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&error.buffer));

        dim2 dim = cl_gridsize(n);
        size_t global_size[] = { dim.x,dim.y,BLOCK };

        cl_event e;
        cl_int status = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
        cl->checkError(status);
        cl->checkError(clWaitForEvents(1, &e));
        clReleaseEvent(e);
}

void upsample_gpu(CLArray in, int w, int h, int c, int batch, int stride, int forward, float scale, CLArray out)
{
        std::shared_ptr<CLWarpper> cl = getCLWarpper();
        if (program_2 == NULL)
        program_2 = cl->buildProgramFromFile(kernel_file_2, "");
        cl_kernel kernel = program_2->getKernel("upsample_kernel");
		int size = w*h*c*batch*stride*stride;
        cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&size));
        cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&in.buffer));
        cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&w));
        cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&h));
        cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&c));
        cl->checkError(clSetKernelArg(kernel, 5, sizeof(int), (void*)&batch));
        cl->checkError(clSetKernelArg(kernel, 6, sizeof(int), (void*)&stride));
        cl->checkError(clSetKernelArg(kernel, 7, sizeof(int), (void*)&forward));
        cl->checkError(clSetKernelArg(kernel, 8, sizeof(float), (void*)&scale));
        cl->checkError(clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&out.buffer));

        dim2 dim = cl_gridsize(size);
        size_t global_size[] = { dim.x,dim.y,BLOCK };

        cl_event e;
        cl_int status = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
        cl->checkError(status);
        cl->checkError(clWaitForEvents(1, &e));
        clReleaseEvent(e);
}
#endif
