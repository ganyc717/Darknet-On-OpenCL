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

	int n = h * w*c*layer.batch;


	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("forward_maxpool_layer_kernel");

	CLKernel clkernel = CLKernel(kernel);
	cl->checkError(clkernel.setArgs(&n));
	cl->checkError(clkernel.setArgs(&layer.h));
	cl->checkError(clkernel.setArgs(&layer.w));
	cl->checkError(clkernel.setArgs(&layer.c));
	cl->checkError(clkernel.setArgs(&layer.stride));
	cl->checkError(clkernel.setArgs(&layer.size));
	cl->checkError(clkernel.setArgs(&layer.pad));
	cl->checkError(clkernel.setArgs(&net.input_gpu.buffer));
	cl->checkError(clkernel.setArgs(&layer.output_gpu.buffer));
	cl->checkError(clkernel.setArgs(&layer.indexes_gpu.buffer));

	dim2 dim = cl_gridsize(n);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clkernel.run(*cl->queue, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void backward_maxpool_layer_gpu(maxpool_layer layer, network net)
{
	int n = layer.h*layer.w*layer.c*layer.batch;

	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("backward_maxpool_layer_kernel");

	CLKernel clkernel = CLKernel(kernel);
	cl->checkError(clkernel.setArgs(&n));
	cl->checkError(clkernel.setArgs(&layer.h));
	cl->checkError(clkernel.setArgs(&layer.w));
	cl->checkError(clkernel.setArgs(&layer.c));
	cl->checkError(clkernel.setArgs(&layer.stride));
	cl->checkError(clkernel.setArgs(&layer.size));
	cl->checkError(clkernel.setArgs(&layer.pad));
	cl->checkError(clkernel.setArgs(&layer.delta_gpu.buffer));
	cl->checkError(clkernel.setArgs(&net.delta_gpu.buffer));
	cl->checkError(clkernel.setArgs(&layer.indexes_gpu.buffer));

	dim2 dim = cl_gridsize(n);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e;
	cl_int error = clkernel.run(*cl->queue, 3, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}
#endif