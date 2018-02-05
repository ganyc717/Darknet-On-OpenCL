#ifdef GPU
#include "crop_layer.h"
#include "utils.h"
#include "ocl.h"
#include "image.h"


const static std::string kernel_file = "crop_layer_kernels.cl";
static std::shared_ptr<CLProgram> program = NULL;

void forward_crop_layer_gpu(crop_layer layer, network net)
{
	cl_random(layer.rand_gpu, layer.batch * 8);

	float radians = layer.angle*3.14159265f / 180.f;

	float scale = 2;
	float translate = -1;
	if (layer.noadjust) {
		scale = 1;
		translate = 0;
	}

	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	if (program == NULL)
		program = cl->buildProgramFromFile(kernel_file, "");
	cl_kernel levels_image_kernel = program->getKernel("levels_image_kernel");
	cl_kernel forward_crop_layer_kernel = program->getKernel("forward_crop_layer_kernel");

	cl->checkError(clSetKernelArg(levels_image_kernel, 0, sizeof(cl_mem), (void*)&net.input_gpu.buffer));
	cl->checkError(clSetKernelArg(levels_image_kernel, 1, sizeof(cl_mem), (void*)&layer.rand_gpu.buffer));
	cl->checkError(clSetKernelArg(levels_image_kernel, 2, sizeof(int), (void*)&layer.batch));
	cl->checkError(clSetKernelArg(levels_image_kernel, 3, sizeof(int), (void*)&layer.w));
	cl->checkError(clSetKernelArg(levels_image_kernel, 4, sizeof(int), (void*)&layer.h));
	cl->checkError(clSetKernelArg(levels_image_kernel, 5, sizeof(int), (void*)&net.train));
	cl->checkError(clSetKernelArg(levels_image_kernel, 6, sizeof(float), (void*)&layer.saturation));
	cl->checkError(clSetKernelArg(levels_image_kernel, 7, sizeof(float), (void*)&layer.exposure));
	cl->checkError(clSetKernelArg(levels_image_kernel, 8, sizeof(float), (void*)&translate));
	cl->checkError(clSetKernelArg(levels_image_kernel, 9, sizeof(float), (void*)&scale));
	cl->checkError(clSetKernelArg(levels_image_kernel, 10, sizeof(float), (void*)&layer.shift));


	int size = layer.batch * layer.w * layer.h;
	dim2 dim = cl_gridsize(size);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e1;
	cl_int error = clEnqueueNDRangeKernel(*cl->queue, levels_image_kernel, 3, NULL, global_size, NULL, NULL, NULL, &e1);
	cl->checkError(error);

	cl->checkError(clSetKernelArg(forward_crop_layer_kernel, 0, sizeof(cl_mem), (void*)&net.input_gpu.buffer));
	cl->checkError(clSetKernelArg(forward_crop_layer_kernel, 1, sizeof(cl_mem), (void*)&layer.rand_gpu.buffer));
	cl->checkError(clSetKernelArg(forward_crop_layer_kernel, 2, sizeof(int), (void*)&size));
	cl->checkError(clSetKernelArg(forward_crop_layer_kernel, 3, sizeof(int), (void*)&layer.c));
	cl->checkError(clSetKernelArg(forward_crop_layer_kernel, 4, sizeof(int), (void*)&layer.h));
	cl->checkError(clSetKernelArg(forward_crop_layer_kernel, 5, sizeof(int), (void*)&layer.w));
	cl->checkError(clSetKernelArg(forward_crop_layer_kernel, 6, sizeof(int), (void*)&layer.out_h));
	cl->checkError(clSetKernelArg(forward_crop_layer_kernel, 7, sizeof(int), (void*)&layer.out_w));
	cl->checkError(clSetKernelArg(forward_crop_layer_kernel, 8, sizeof(int), (void*)&net.train));
	cl->checkError(clSetKernelArg(forward_crop_layer_kernel, 9, sizeof(int), (void*)&layer.flip));
	cl->checkError(clSetKernelArg(forward_crop_layer_kernel, 10, sizeof(float), (void*)&radians));
	cl->checkError(clSetKernelArg(forward_crop_layer_kernel, 11, sizeof(cl_mem), (void*)&layer.output_gpu.buffer));

	size = layer.batch*layer.c*layer.out_w*layer.out_h;
	dim = cl_gridsize(size);
	global_size[0] = dim.x;
	global_size[1] = dim.y;
	global_size[2] = BLOCK;

	cl_event e2;
	error = clEnqueueNDRangeKernel(*cl->queue, forward_crop_layer_kernel, 3, NULL, global_size, NULL, 1, &e1, &e2);
	cl->checkError(error);
	clReleaseEvent(e1);

	cl->checkError(clWaitForEvents(1, &e2));
	clReleaseEvent(e2);
}
#endif