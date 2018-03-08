int gpu_index = -1;

#ifdef GPU

#include "ocl.h"
#include "utils.h"
#include "blas.h"
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <random>

static std::shared_ptr<CLWarpper> cl = CLWarpper::createForIndexedGpu(0);;
std::shared_ptr<CLWarpper> getCLWarpper()
{
	return cl;
}

CLArray operator+(CLArray buffer, size_t offset)
{
	assert(buffer.ptr != NULL);
	cl_mem sub;
	cl_int err;
	cl_buffer_region region;

	size_t origin = buffer.origin + offset;
	assert(origin >= 0);
	size_t sub_buffer_size = buffer.major_size - origin;
	assert(sub_buffer_size >= 0);
	float* host_ptr = (float*)buffer.ptr;

	return CLArray(host_ptr + offset, buffer.major_size, origin, sub_buffer_size);
}

CLArray operator-(CLArray buffer, size_t offset)
{
	int size = -1 * offset;
	return buffer + size;
}

CLArray &CLArray::operator+= (size_t offset)
{
	*this = *this + offset;
	return *this;
}

CLArray &CLArray::operator-= (size_t offset)
{
	*this = *this - offset;
	return *this;
}

CLArray::CLArray()
{
	ptr = 0;
    origin = 0;
    major_size = 0;
    size = 0;
};

CLArray::CLArray(void* map_ptr, size_t major_mem_size, size_t origin_offset, size_t buf_size)
{
	ptr = map_ptr;
    major_size = major_mem_size;
    origin = origin_offset;
    size = buf_size;
};

CLArray::~CLArray()
{};//Do nothing here, explicitly call cl_free to release cl_mem


void cl_set_device(int n)
{
	if (gpu_index == -1)
	{
		gpu_index = n;
		cl = CLWarpper::createForIndexedGpu(gpu_index);
	}
	else if (n != gpu_index)
	{
		throw std::runtime_error("don't support temporary\n");
	}
}

int cl_get_device()
{
    return gpu_index;
}

void check_error(cl_int status)
{
	cl->checkError(status);
}

dim2 cl_gridsize(size_t n){
    size_t k = (n-1) / BLOCK + 1;
    size_t x = k;
    size_t y = 1;
    if(x > 65535){
        x = ceil(sqrt(k));
        y = (n-1)/(x*BLOCK) + 1;
    }
    dim2 d = {x, y};
    //printf("%ld %ld %ld %ld\n", n, x, y, x*y*BLOCK);
    return d;
}

CLArray cl_make_array(float *x, size_t n)
{
	void* x_gpu;
	cl_int error;
	size_t size = sizeof(float)*n;

	x_gpu = clSVMAlloc(*(cl->context), CL_MEM_READ_WRITE, size, 0);
	if (!x_gpu) throw std::runtime_error("OpenCL SVM malloc failed, didn't support this feature\n");

	CLArray buf = CLArray(x_gpu, n, 0, n);
	if (x) {
		cl_event e;
		//check_error(clEnqueueSVMMap(*(cl->queue), CL_TRUE, CL_MAP_WRITE, x_gpu, size, 0, NULL, NULL));
		int a = clEnqueueSVMMap(*(cl->queue), CL_TRUE, CL_MAP_WRITE, x_gpu, size, 0, NULL, NULL);
		
		memcpy(x_gpu, x, size);
		check_error(clEnqueueSVMUnmap(*(cl->queue), x_gpu, 0, NULL, &e));
		check_error(clWaitForEvents(1, &e));
		clReleaseEvent(e);
	}
	else {
		fill_gpu(n, 0, buf, 1);
	}
	
	return buf;

}

void cl_random(CLArray x_gpu, size_t n)
{//Any good idea to generate random float using opencl?
	std::default_random_engine e;
	std::uniform_real_distribution<float> u(0.0, 1.0);
	cl_event event;
	size_t size = n * sizeof(float);
	
	check_error(clEnqueueSVMMap(*(cl->queue), CL_TRUE, CL_MAP_WRITE, x_gpu.ptr, size, 0, NULL, NULL));
	float* buffer = (float*)x_gpu.ptr;
	for (int i = 0; i < n; i++)
		buffer[i] = u(e);
	check_error(clEnqueueSVMUnmap(*(cl->queue), x_gpu.ptr, 0, NULL, &event));
	check_error(clWaitForEvents(1, &event));
	clReleaseEvent(event);
}

float cl_compare(CLArray x_gpu, float *x, size_t n, char *s)
{
	float *tmp = new float(n);
    cl_pull_array(x_gpu, tmp, n);
    //int i;
    //for(i = 0; i < n; ++i) printf("%f %f\n", tmp[i], x[i]);
    axpy_cpu(n, -1, x, 1, tmp, 1);
    float err = dot_cpu(n, tmp, 1, tmp, 1);
    printf("Error %s: %f\n", s, sqrt(err/n));
    free(tmp);
    return err;
}

CLArray cl_make_int_array(int *x, size_t n)
{/*
	cl_mem x_gpu;
	cl_int error;
	size_t size = sizeof(int)*n;
	x_gpu = clCreateBuffer(*(cl->context), CL_MEM_READ_WRITE, size, 0, &error);
	cl->checkError(error);

	CLArray buf = CLArray(x_gpu, x_gpu, n, 0, n);
	if (x) {
		error = clEnqueueWriteBuffer(*(cl->queue), x_gpu, CL_TRUE, 0, size, x, 0, NULL, NULL);
		check_error(error);
	}
	else {
		fill_gpu(n, 0, buf, 1);
	}
	if (!x_gpu) throw std::runtime_error("opencl malloc failed\n");
	return buf;*/
	return cl_make_array((float*)x, n);
}

void cl_free(CLArray x_gpu)
{
	clSVMFree(*(cl->context), x_gpu.ptr);
}

void cl_push_array(CLArray x_gpu, float *x, size_t n)
{
	cl_event e;
	size_t size = sizeof(float)*n;
	check_error(clEnqueueSVMMap(*(cl->queue), CL_TRUE, CL_MAP_WRITE, x_gpu.ptr, size, 0, NULL, NULL));
	memcpy(x_gpu.ptr, x, size);
	check_error(clEnqueueSVMUnmap(*(cl->queue), x_gpu.ptr, 0, NULL, &e));
	check_error(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

void cl_pull_array(CLArray x_gpu, float *x, size_t n)
{
	cl_event e;
	size_t size = sizeof(float)*n;
	check_error(clEnqueueSVMMap(*(cl->queue), CL_TRUE, CL_MAP_READ, x_gpu.ptr, size, 0, NULL, NULL));
	memcpy(x, x_gpu.ptr, size);
	check_error(clEnqueueSVMUnmap(*(cl->queue), x_gpu.ptr, 0, NULL, &e));
	check_error(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}

float cl_mag_array(CLArray x_gpu, size_t n)
{
	float *temp = new float[n];
    cl_pull_array(x_gpu, temp, n);
    float m = mag_array(temp, n);
	delete[] temp;
    return m;
}
#else
void cuda_set_device(int n){}

#endif
