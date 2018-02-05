#ifdef GPU

#include "ocl.h"
#include "avgpool_layer.h"
const static std::string kernel_file = "avgpool_layer_kernels.cl";
static std::shared_ptr<CLProgram> program = NULL;


void forward_avgpool_layer_gpu(avgpool_layer layer, network net)
{
	size_t n = layer.c*layer.batch;

	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("forward_avgpool_layer_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&n));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(int), (void*)&layer.w));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&layer.h));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&layer.c));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&net.input_gpu.buffer));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&layer.output_gpu.buffer));

	dim2 dim = cl_gridsize(n);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void backward_avgpool_layer_gpu(avgpool_layer layer, network net)
{
	size_t n = layer.c*layer.batch;

	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("backward_avgpool_layer_kernel");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&n));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(int), (void*)&layer.w));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&layer.h));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&layer.c));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&net.delta_gpu.buffer));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&layer.delta_gpu.buffer));

	dim2 dim = cl_gridsize(n);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}
#endif