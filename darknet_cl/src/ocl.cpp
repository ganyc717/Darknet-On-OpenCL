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
	assert(buffer.major != NULL);
	cl_mem sub;
	cl_int err;
	cl_buffer_region region;

	size_t origin = buffer.origin + offset;
	assert(origin >= 0);
	size_t sub_buffer_size = buffer.major_size - origin;
	assert(sub_buffer_size >= 0);

	region.origin = origin * sizeof(float);
	region.size = sub_buffer_size * sizeof(float);

	sub = clCreateSubBuffer(buffer.major, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
	check_error(err);
	return CLArray(buffer.major, sub, buffer.major_size, origin, sub_buffer_size);
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
    major = 0;
    buffer = 0;
    origin = 0;
    major_size = 0;
    size = 0;
};

CLArray::CLArray(cl_mem major_mem, cl_mem mem, size_t major_mem_size, size_t origin_offset, size_t buf_size)
{
    major = major_mem;
    buffer = mem;
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

#ifdef CUDNN
cudnnHandle_t cudnn_handle()
{
    static int init[16] = {0};
    static cudnnHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cudnnCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
}
#endif

CLArray cl_make_array(float *x, size_t n)
{
    cl_mem x_gpu;
	cl_int error;
    size_t size = sizeof(float)*n;
	x_gpu = clCreateBuffer(*(cl->context), CL_MEM_READ_WRITE, size, 0, &error);
	cl->checkError(error);
	CLArray buf = CLArray(x_gpu, x_gpu, n, 0, n);

    if(x){
        error = clEnqueueWriteBuffer(*(cl->queue), x_gpu, CL_TRUE, 0, size, x, 0, NULL, NULL);
        check_error(error);
    } else {
        fill_gpu(n, 0, buf, 1);
    }
    if(!x_gpu) throw std::runtime_error("opencl malloc failed\n");
	return buf;
}

void cl_random(CLArray x_gpu, size_t n)
{//Any good idea to generate random float using opencl?
	std::default_random_engine e;
	std::uniform_real_distribution<float> u(0.0, 1.0);
	float* buffer = new float[n];
	for (int i = 0; i < n; i++)
		buffer[i] = u(e);
	cl_push_array(x_gpu, buffer, n);
	delete[] buffer;
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
{
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
	return buf;
}

void cl_free(CLArray x_gpu)
{
    cl_int status = clReleaseMemObject(x_gpu.buffer);
    check_error(status);
}

void cl_push_array(CLArray x_gpu, float *x, size_t n)
{
    size_t size = sizeof(float)*n;
    cl_int status = clEnqueueWriteBuffer(*(cl->queue), x_gpu.buffer, CL_TRUE, 0, size, x, 0, NULL, NULL);
    check_error(status);
}

void cl_pull_array(CLArray x_gpu, float *x, size_t n)
{
	cl_event event = NULL;
	cl_int status;
	size_t size = sizeof(float) * n;
	status = clEnqueueReadBuffer(*(cl->queue), x_gpu.buffer, CL_TRUE, 0, size, x, 0, NULL, &event);
	cl->checkError(status);
	cl_int err = clWaitForEvents(1, &event);
	clReleaseEvent(event);
	if (err != CL_SUCCESS) {
		throw std::runtime_error("wait for event on copytohost failed with " + CLWarpper::toString(err));
	}
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
