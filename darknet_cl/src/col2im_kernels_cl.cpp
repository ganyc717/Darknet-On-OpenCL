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

	cl->checkError(clSetKernelArg(kernel, 0, sizeof(int), (void*)&num_kernels));
	cl->checkError(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&data_col.buffer));
	cl->checkError(clSetKernelArg(kernel, 2, sizeof(int), (void*)&height));
	cl->checkError(clSetKernelArg(kernel, 3, sizeof(int), (void*)&width));
	cl->checkError(clSetKernelArg(kernel, 4, sizeof(int), (void*)&ksize));
	cl->checkError(clSetKernelArg(kernel, 5, sizeof(int), (void*)&pad));
	cl->checkError(clSetKernelArg(kernel, 6, sizeof(int), (void*)&stride));
	cl->checkError(clSetKernelArg(kernel, 7, sizeof(int), (void*)&height_col));
	cl->checkError(clSetKernelArg(kernel, 8, sizeof(int), (void*)&width_col));
	cl->checkError(clSetKernelArg(kernel, 9, sizeof(cl_mem), (void*)&data_im.buffer));

	size_t global_size[] = { (num_kernels + BLOCK - 1) / BLOCK,BLOCK };

	cl_event e;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, kernel, 2, NULL, global_size, NULL, NULL, NULL, &e);
	cl->checkError(error);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}
#endif