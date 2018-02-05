#ifdef GPU
#include "maxpool_layer.h"
#include "ocl.h"

const static std::string kernel_file = "maxpool_layer_kernels.cl";
static std::shared_ptr<CLProgram> program = NULL;

void forward_maxpool_layer_gpu(maxpool_layer layer, network net)
{
	int h = layer.out_h;
	int w = layer.out_w;
	int c = layer.c;

	size_t n = h * w*c*layer.batch;


	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("forward_maxpool_layer_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&n));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(int), (void*)&layer.h));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&layer.w));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&layer.c));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&layer.stride));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(int), (void*)&layer.size));
	cl->checkError(clSetKernelArg(kernel, 6, sizeof(int), (void*)&layer.pad));
	cl->checkError(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&net.input_gpu.buffer));
	cl->checkError(clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&layer.output_gpu.buffer));
	cl->checkError(clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&layer.indexes_gpu.buffer));

	dim2 dim = cl_gridsize(n);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void backward_maxpool_layer_gpu(maxpool_layer layer, network net)
{
	size_t n = layer.h*layer.w*layer.c*layer.batch;

	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("backward_maxpool_layer_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&n));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(int), (void*)&layer.h));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&layer.w));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&layer.c));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&layer.stride));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(int), (void*)&layer.size));
	cl->checkError(clSetKernelArg(kernel, 6, sizeof(int), (void*)&layer.pad));
	cl->checkError(clSetKernelArg(kernel, 7, sizeof(cl_mem), (void*)&layer.delta_gpu.buffer));
	cl->checkError(clSetKernelArg(kernel, 8, sizeof(cl_mem), (void*)&net.delta_gpu.buffer));
	cl->checkError(clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&layer.indexes_gpu.buffer));

	dim2 dim = cl_gridsize(n);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}


#endif