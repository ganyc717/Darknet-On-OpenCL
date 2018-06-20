#ifdef GPU
#include "col2im.h"
#include "ocl.h"
const static std::string kernel_file = "col2im_kernels.cl";
static std::shared_ptr<CLProgram> program = NULL;

void col2im_gpu(CLArray data_col,
	int channels, int height, int width,
	int ksize, int stride, int pad, CLArray data_im) {
	// We are going to launch channels * height_col * width_col kernels, each
	// kernel responsible for copying a single-channel grid.
	int height_col = (height + 2 * pad - ksize) / stride + 1;
	int width_col = (width + 2 * pad - ksize) / stride + 1;
	int num_kernels = channels * height * width;

	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel kernel = program->getKernel("col2im_gpu_kernel");

	CLKernel clkernel = CLKernel(kernel);
	cl->checkError(clkernel.setArgs(&num_kernels));
	cl->checkError(clkernel.setArgs(&data_col.buffer));
	cl->checkError(clkernel.setArgs(&height));
	cl->checkError(clkernel.setArgs(&width));
	cl->checkError(clkernel.setArgs(&ksize));
	cl->checkError(clkernel.setArgs(&pad));
	cl->checkError(clkernel.setArgs(&stride));
	cl->checkError(clkernel.setArgs(&height_col));
	cl->checkError(clkernel.setArgs(&width_col));
	cl->checkError(clkernel.setArgs(&data_im.buffer));

	size_t global_size[] = { (num_kernels + BLOCK - 1) / BLOCK,BLOCK };

	cl_event e;
	cl_int error = clkernel.run(*cl->queue, 2, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}
#endif