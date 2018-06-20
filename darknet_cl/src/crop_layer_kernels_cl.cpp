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

	CLKernel clkernel_levels_image_kernel = CLKernel(levels_image_kernel);
	cl->checkError(clkernel_levels_image_kernel.setArgs(&net.input_gpu.buffer));
	cl->checkError(clkernel_levels_image_kernel.setArgs(&layer.rand_gpu.buffer));
	cl->checkError(clkernel_levels_image_kernel.setArgs(&layer.batch));
	cl->checkError(clkernel_levels_image_kernel.setArgs(&layer.w));
	cl->checkError(clkernel_levels_image_kernel.setArgs(&layer.h));
	cl->checkError(clkernel_levels_image_kernel.setArgs(&net.train));
	cl->checkError(clkernel_levels_image_kernel.setArgs(&layer.saturation));
	cl->checkError(clkernel_levels_image_kernel.setArgs(&layer.exposure));
	cl->checkError(clkernel_levels_image_kernel.setArgs(&translate));
	cl->checkError(clkernel_levels_image_kernel.setArgs(&scale));
	cl->checkError(clkernel_levels_image_kernel.setArgs(&layer.shift));


	int size = layer.batch * layer.w * layer.h;
	dim2 dim = cl_gridsize(size);
	size_t global_size[] = { dim.x,dim.y,BLOCK };

	cl_event e1;
	cl_int error = clkernel_levels_image_kernel.run(*cl->queue, 3, NULL, global_size, NULL, NULL, NULL, &e1);
	cl->checkError(error);

	CLKernel clkernel_forward_crop_layer_kernel = CLKernel(forward_crop_layer_kernel);
	cl->checkError(clkernel_forward_crop_layer_kernel.setArgs(&net.input_gpu.buffer));
	cl->checkError(clkernel_forward_crop_layer_kernel.setArgs(&layer.rand_gpu.buffer));
	cl->checkError(clkernel_forward_crop_layer_kernel.setArgs(&size));
	cl->checkError(clkernel_forward_crop_layer_kernel.setArgs(&layer.c));
	cl->checkError(clkernel_forward_crop_layer_kernel.setArgs(&layer.h));
	cl->checkError(clkernel_forward_crop_layer_kernel.setArgs(&layer.w));
	cl->checkError(clkernel_forward_crop_layer_kernel.setArgs(&layer.out_h));
	cl->checkError(clkernel_forward_crop_layer_kernel.setArgs(&layer.out_w));
	cl->checkError(clkernel_forward_crop_layer_kernel.setArgs(&net.train));
	cl->checkError(clkernel_forward_crop_layer_kernel.setArgs(&layer.flip));
	cl->checkError(clkernel_forward_crop_layer_kernel.setArgs(&radians));
	cl->checkError(clkernel_forward_crop_layer_kernel.setArgs(&layer.output_gpu.buffer));

	size = layer.batch*layer.c*layer.out_w*layer.out_h;
	dim = cl_gridsize(size);
	global_size[0] = dim.x;
	global_size[1] = dim.y;
	global_size[2] = BLOCK;

	cl_event e2;
	error = clkernel_forward_crop_layer_kernel.run(*cl->queue, 3, NULL, global_size, NULL, 1, &e1, &e2);
	cl->checkError(error);
	clReleaseEvent(e1);

	cl->checkError(clWaitForEvents(1, &e2));
	clReleaseEvent(e2);
}
#endif