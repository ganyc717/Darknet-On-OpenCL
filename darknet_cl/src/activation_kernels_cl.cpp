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

	int t = n / 2;
	int binary_activation = (int)a;

	CLKernel clkernel = CLKernel(kernel);
	cl->checkError(clkernel.setArgs(&x.buffer));
	cl->checkError(clkernel.setArgs(&dx.buffer));
	cl->checkError(clkernel.setArgs(&t));
	cl->checkError(clkernel.setArgs(&size));
	cl->checkError(clkernel.setArgs(&binary_activation));
	cl->checkError(clkernel.setArgs(&y.buffer));

	dim2 dim = cl_gridsize(t);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clkernel.run(*cl->queue, 3, NULL, global_size, NULL, NULL, NULL, &e);
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

	int t = n / 2;
	int binary_activation = (int)a;

	CLKernel clkernel = CLKernel(kernel);
	cl->checkError(clkernel.setArgs(&x.buffer));
	cl->checkError(clkernel.setArgs(&t));
	cl->checkError(clkernel.setArgs(&size));
	cl->checkError(clkernel.setArgs(&binary_activation));
	cl->checkError(clkernel.setArgs(&y.buffer));

	dim2 dim = cl_gridsize(t);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clkernel.run(*cl->queue, 3, NULL, global_size, NULL, NULL, NULL, &e);
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

	CLKernel clkernel = CLKernel(kernel);
	cl->checkError(clkernel.setArgs(&x.buffer));
	cl->checkError(clkernel.setArgs(&n));
	cl->checkError(clkernel.setArgs(&activation));

	dim2 dim = cl_gridsize(n);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clkernel.run(*cl->queue, 3, NULL, global_size, NULL, NULL, NULL, &e);
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

	CLKernel clkernel = CLKernel(kernel);
	cl->checkError(clkernel.setArgs(&x.buffer));
	cl->checkError(clkernel.setArgs(&n));
	cl->checkError(clkernel.setArgs(&activation));
	cl->checkError(clkernel.setArgs(&delta.buffer));

	dim2 dim = cl_gridsize(n);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clkernel.run(*cl->queue, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}
#endif