#ifndef OCL_H
#define OCL_H

#include "darknet.h"

#ifdef GPU
#include "cl_warpper.h"
#include <memory>
std::shared_ptr<CLWarpper> getCLWarpper();

typedef struct dim2 {
	size_t x;
	size_t y;
}dim2;

void check_error(cl_int status);
CLArray cl_make_int_array(int *x, size_t n);

void cl_random(CLArray x_gpu, size_t n);
float cl_compare(CLArray x_gpu, float *x, size_t n, char *s);
dim2 cl_gridsize(size_t n);


#ifdef CUDNN
cudnnHandle_t cudnn_handle();
#endif

#endif
#endif
