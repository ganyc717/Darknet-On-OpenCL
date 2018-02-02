#include "gemm.h"
#include "utils.h"
#include "ocl.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            char A_PART = A[i*lda+k];
            if(A_PART){
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] += B[k*ldb+j];
                }
            } else {
                for(j = 0; j < N; ++j){
                    C[i*ldc+j] -= B[k*ldb+j];
                }
            }
        }
    }
}

float *random_matrix(int rows, int cols)
{
    int i;
    float *m = (float*)calloc(rows*cols, sizeof(float));
    for(i = 0; i < rows*cols; ++i){
        m[i] = (float)rand()/RAND_MAX;
    }
    return m;
}

void time_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
    for(i = 0; i<10; ++i){
        gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    }
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf ms\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
}


void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

void gemm_nn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_nt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float *C, int ldc)
{
    int i,j,k;
    #pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float *A, int lda, 
        float *B, int ldb,
        float BETA,
        float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

#ifdef GPU

#include <math.h>
void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
	CLArray A_gpu, int lda,
	CLArray B_gpu, int ldb,
	float BETA,
	CLArray C_gpu, int ldc)
{
	//cl_int status = clblasSetup();
	//check_error(status);
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	cl_event e;

	clblasTranspose transA = (TA ? clblasTrans : clblasNoTrans);
	clblasTranspose transB = (TB ? clblasTrans : clblasNoTrans);

	clblasSgemm(
		clblasColumnMajor,//Make column major the same with cublasSgemm
		transB, transA, N, M, K, ALPHA, B_gpu.buffer, 0, ldb, A_gpu.buffer, 0, lda, BETA,
		C_gpu.buffer, 0, ldc,
		1,//cl_uint numCommandQueues,
		cl->queue,//cl_command_queue *commandQueues,
		0,//cl_uint numEventsInWaitList,
		NULL,//const cl_event *eventWaitList,
		&e//cl_event *events
	);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
	//clblasTeardown();
}
/*
void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        float* A, int lda, 
	    float* B, int ldb,
        float BETA,
		float* C, int ldc)
{
    //cublasHandle_t handle = blas_handle();
    //cudaError_t status = cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N), 
    //        (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K, &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu, ldc);
    //check_error(status);
	///////////////////////////////////////////
	cl_mem A_gpu = cl_make_array(A, M*K);
	cl_mem B_gpu = cl_make_array(B, N*K);
	cl_mem C_gpu = cl_make_array(C, M*N);

	cl_int status = clblasSetup();
	check_error(status);
	std::shared_ptr<CLWarpper> cl = getCLWarpper();
	cl_event e;

	clblasTranspose transA = (TA ? clblasTrans : clblasNoTrans);
	clblasTranspose transB = (TB ? clblasTrans : clblasNoTrans);

	clblasSgemm(
		clblasColumnMajor,//Make column major the same with cublasSgemm
		transB,transA,N,M,K,ALPHA,B_gpu,0,ldb,A_gpu,0,lda,BETA,
		C_gpu,0,ldc,
		1,//cl_uint numCommandQueues,
		cl->queue,//cl_command_queue *commandQueues,
		0,//cl_uint numEventsInWaitList,
		NULL,//const cl_event *eventWaitList,
		&e//cl_event *events
	);
	cl->checkError(clWaitForEvents(1, &e));
	clReleaseEvent(e);
}*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void time_gpu_random_matrix(int TA, int TB, int m, int k, int n)
{
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    int i;
    clock_t start = clock(), end;
	CLArray gpu_a = cl_make_array(a, m*k);
	CLArray gpu_b = cl_make_array(b, n*k);
	CLArray gpu_c = cl_make_array(c, m*n);

    for(i = 0; i<32; ++i){
        gemm_gpu(TA,TB,m,n,k,1, gpu_a,lda, gpu_b,ldb,1, gpu_c,n);
    }
	cl_pull_array(gpu_c, c, m*n);
    end = clock();
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s\n",m,k,k,n, TA, TB, (float)(end-start)/CLOCKS_PER_SEC);
    free(a);
    free(b);
    free(c);
	cl_free(gpu_a);
	cl_free(gpu_b);
	cl_free(gpu_c);
}

void time_gpu(int TA, int TB, int m, int k, int n)
{
    int iter = 10;
    float *a = random_matrix(m,k);
    float *b = random_matrix(k,n);

    int lda = (!TA)?k:m;
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);

    int i;
    clock_t start = clock(), end;
	CLArray gpu_a = cl_make_array(a, m*k);
	CLArray gpu_b = cl_make_array(b, n*k);
	CLArray gpu_c = cl_make_array(c, m*n);


    for(i = 0; i<iter; ++i){
        gemm_gpu(TA,TB,m,n,k,1, gpu_a,lda, gpu_b,ldb,1, gpu_c,n);
    }
    double flop = ((double)m)*n*(2.*k + 2.)*iter;
    double gflop = flop/pow(10., 9);
	cl_pull_array(gpu_c, c, m*n);
    end = clock();
    double seconds = sec(end-start);
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %lf s, %lf GFLOPS\n",m,k,k,n, TA, TB, seconds, gflop/seconds);
    free(a);
    free(b);
    free(c);
	cl_free(gpu_a);
	cl_free(gpu_b);
	cl_free(gpu_c);
}


void test_gpu_accuracy(int TA, int TB, int m, int k, int n)
{
    srand(0);
    float *a;
    if(!TA) a = random_matrix(m,k);
    else a = random_matrix(k,m);
    int lda = (!TA)?k:m;
    float *b;
    if(!TB) b = random_matrix(k,n);
    else b = random_matrix(n,k);
    int ldb = (!TB)?n:k;

    float *c = random_matrix(m,n);
    float *c_gpu = random_matrix(m,n);
    memset(c, 0, m*n*sizeof(float));
    memset(c_gpu, 0, m*n*sizeof(float));

	CLArray gpu_a = cl_make_array(a, m*k);
	CLArray gpu_b = cl_make_array(b, n*k);
	CLArray gpu_c = cl_make_array(c_gpu, m*n);

    int i;
    //pm(m,k,b);
    gemm_gpu(TA,TB,m,n,k,1, gpu_a,lda, gpu_b,ldb,1, gpu_c,n);
	cl_pull_array(gpu_c, c_gpu, m*n);
    //printf("GPU\n");
    //pm(m, n, c_gpu);

    gemm_cpu(TA,TB,m,n,k,1,a,lda,b,ldb,1,c,n);
    //printf("\n\nCPU\n");
    //pm(m, n, c);
    double sse = 0;
    for(i = 0; i < m*n; ++i) {
        //printf("%f %f\n", c[i], c_gpu[i]);
        sse += pow(c[i]-c_gpu[i], 2);
    }
    printf("Matrix Multiplication %dx%d * %dx%d, TA=%d, TB=%d: %g SSE\n",m,k,k,n, TA, TB, sse/(m*n));
    free(a);
    free(b);
    free(c);
    free(c_gpu);
	cl_free(gpu_a);
	cl_free(gpu_b);
	cl_free(gpu_c);
}

int test_gpu_blas()
{
    /*
       test_gpu_accuracy(0,0,10,576,75); 

       test_gpu_accuracy(0,0,17,10,10); 
       test_gpu_accuracy(1,0,17,10,10); 
       test_gpu_accuracy(0,1,17,10,10); 
       test_gpu_accuracy(1,1,17,10,10); 

       test_gpu_accuracy(0,0,1000,10,100); 
       test_gpu_accuracy(1,0,1000,10,100); 
       test_gpu_accuracy(0,1,1000,10,100); 
       test_gpu_accuracy(1,1,1000,10,100); 

       test_gpu_accuracy(0,0,10,10,10); 

       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,64,2916,363); 
       time_gpu(0,0,192,729,1600); 
       time_gpu(0,0,384,196,1728); 
       time_gpu(0,0,256,196,3456); 
       time_gpu(0,0,256,196,2304); 
       time_gpu(0,0,128,4096,12544); 
       time_gpu(0,0,128,4096,4096); 
     */
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,75,12544); 
    time_gpu(0,0,64,576,12544); 
    time_gpu(0,0,256,2304,784); 
    time_gpu(1,1,2304,256,784); 
    time_gpu(0,0,512,4608,196); 
    time_gpu(1,1,4608,512,196); 

    return 0;
}
#endif

