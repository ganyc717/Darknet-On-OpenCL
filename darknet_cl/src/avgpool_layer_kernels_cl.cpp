#ifdef GPU

#include "ocl.h"
#include "avgpool_layer.h"
const static std::string kernel_file = "avgpool_layer_kernels.cl";
static std::shared_ptr<CLProgram> program = NULL;


void forward_avgpool_layer_gpu(avgpool_layer layer, network net)
{
	int n = layer.c*layer.batch;

	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("forward_avgpool_layer_kernel");

	CLKernel clkernel = CLKernel(kernel);
	cl->checkError(clkernel.setArgs(&n));
	cl->checkError(clkernel.setArgs(&layer.w));
	cl->checkError(clkernel.setArgs(&layer.h));
	cl->checkError(clkernel.setArgs(&layer.c));
	cl->checkError(clkernel.setArgs(&net.input_gpu.buffer));
	cl->checkError(clkernel.setArgs(&layer.output_gpu.buffer));

	dim2 dim = cl_gridsize(n);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clkernel.run(*cl->queue, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void backward_avgpool_layer_gpu(avgpool_layer layer, network net)
{
	int n = layer.c*layer.batch;

	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("backward_avgpool_layer_kernel");

	CLKernel clkernel = CLKernel(kernel);
	cl->checkError(clkernel.setArgs(&n));
	cl->checkError(clkernel.setArgs(&layer.w));
	cl->checkError(clkernel.setArgs(&layer.h));
	cl->checkError(clkernel.setArgs(&layer.c));
	cl->checkError(clkernel.setArgs(&net.delta_gpu.buffer));
	cl->checkError(clkernel.setArgs(&layer.delta_gpu.buffer));

	dim2 dim = cl_gridsize(n);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clkernel.run(*cl->queue, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}
#endif