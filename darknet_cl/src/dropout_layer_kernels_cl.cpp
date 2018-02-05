#ifdef GPU
#include "dropout_layer.h"
#include "ocl.h"
#include "utils.h"

const static std::string kernel_file = "dropout_layer_kernels.cl";
static std::shared_ptr<CLProgram> program = NULL;

void forward_dropout_layer_gpu(dropout_layer layer, network net)
{
	if (!net.train) return;
	int size = layer.inputs*layer.batch;
	cl_random(layer.rand_gpu, size);

	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("yoloswag420blazeit360noscope");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&net.input_gpu.buffer));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(int), (void*)&size));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&layer.rand_gpu.buffer));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(float), (void*)&layer.probability));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(float), (void*)&layer.scale));

	dim2 dim = cl_gridsize(size);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void backward_dropout_layer_gpu(dropout_layer layer, network net)
{
	if (!net.delta_gpu.buffer || net.delta_gpu.size <= 0) return;
	int size = layer.inputs*layer.batch;

	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("yoloswag420blazeit360noscope");

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&net.delta_gpu.buffer));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(int), (void*)&size));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&layer.rand_gpu.buffer));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(float), (void*)&layer.probability));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(float), (void*)&layer.scale));

	dim2 dim = cl_gridsize(size);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}
#endif