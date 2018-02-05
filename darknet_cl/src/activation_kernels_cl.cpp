#ifdef GPU

#include "ocl.h"
#include "activations.h"

const static std::string kernel_file = "activation_kernels.cl";
static std::shared_ptr<CLProgram> program = NULL;


void binary_gradient_array_gpu(CLArray x, CLArray dx, int n, int size, BINARY_ACTIVATION a, CLArray y)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("binary_gradient_array_kernel");

	size_t t = n / 2;
	int binary_activation = (int)a;

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&x.buffer));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&dx.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&t));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&size));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&binary_activation));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&y.buffer));

	dim2 dim = cl_gridsize(t);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void binary_activate_array_gpu(CLArray x, int n, int size, BINARY_ACTIVATION a, CLArray y)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("binary_activate_array_kernel");

	size_t t = n / 2;
	int binary_activation = (int)a;

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&x.buffer));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(int), (void*)&t));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&size));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&binary_activation));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&y.buffer));

	dim2 dim = cl_gridsize(t);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void activate_array_gpu(CLArray x, int n, ACTIVATION a)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("activate_array_kernel");

	int activation = (int)a;

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&x.buffer));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(int), (void*)&n));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&activation));

	dim2 dim = cl_gridsize(n);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void gradient_array_gpu(CLArray x, int n, ACTIVATION a, CLArray delta)
{
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("gradient_array_kernel");

	int activation = (int)a;

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&x.buffer));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(int), (void*)&n));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(ACTIVATION), (void*)&activation));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&delta.buffer));

	dim2 dim = cl_gridsize(n);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}
#endif