#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>

#include "Util.h"




float vecAdd(float* A, float* B, float* C, int N);
float dot(int* A, int* B, int* C, int N);
float mmult(float* A, float* B, float* S, int N);


/* lab0 */

__global__ void kernel(void) { }

int lab0_cpu()
{
	printf("Hello World!\n");
	return EXIT_SUCCESS;
}

int lab0_gpu()
{
	kernel<<<1, 1>>>();
	printf("Hello World!\n");

	return 0;
}

/* lab1 */

void vecAddCPU(int N, float* A, float* B, float* C)
{ 
	int i;

	for (i = 0; i < N; i++)
	{
		C[i] = A[i] + B[i]; 
	}
}

int lab1_cpu()
{
	int i, N = 720896;  /* default vector size */
	float *A, *B, *C; 

	clock_t t1, t2;
	float rt;

	/* check for user-supplied vector size */
	/*if (argc > 1) 
	N = atoi(argv[1]);*/

	printf("Running CPU vecAdd for %i elements\n", N);

	/* allocate memory */ 
	A = (float*)malloc(N * sizeof(float));
	B = (float*)malloc(N * sizeof(float));
	C = (float*)malloc(N * sizeof(float));

	for (i = 0; i < N; i++)  /* generate random data */
	{
		A[i] = (float)rand();
		B[i] = (float)RAND_MAX - A[i];
	}

	/* timing */
	t1 = clock();

	/* call compute kernel */
	vecAddCPU(N, A, B, C);

	/* timing */
	t2 = clock();

	//rt = (float)(msec2-msec1) / 1E6;
	rt = (double)(t2 - t1)/CLOCKS_PER_SEC;
	printf("time=%.4f seconds, MFLOPS=%.1f\n", rt, (float)N/rt/1E6);	

	/* print out first 10 results */
	for (i = 0; i < 10; i++)
		printf("C[%i]=%.2f\n", i, C[i]);

	/* free allocated memory */
	free(A);
	free(B);
	free(C);

	return 0;
}

int lab1_gpu()
{
	int i, N = 720896;  /* default vector size */
	float *A, *B, *C;
	float rt;

	/* check for user-supplied vector size */
	/*if (argc > 1) 
	N = atoi(argv[1]);*/

	/* allocate memory */ 
	A = (float*)malloc(N * sizeof(float));
	B = (float*)malloc(N * sizeof(float));
	C = (float*)malloc(N * sizeof(float));

	printf("Running CPU vecAdd for %i elements\n\n", N);

	//rt = vecAdd(A, B, C, N);

	printf("time=%.4f seconds, MFLOPS=%.1f\n", rt, (float)N/rt/1E6);

	/* print out first 10 results */
	for (i = 0; i < 10; i++)
		printf("C[%i]=%.2f\n", i, C[i]);

	/* free allocated memory */
	free(A);
	free(B);
	free(C);

	return 0;
}

/* lab2 */

int lab2()
{
	cudaDeviceInit(1, 3);

	return 0;
}

/* lab3 */

float dot_cpu(float* A, float* B, int N)
{ 
	int i;
	float s = 0.0f;

	for (i = 0; i < N; i++)
		s += A[i] * B[i]; 

	return s;
}

int lab3_cpu()
{
	long long int i, N = 2097152;  // vector size
	float *A, *B;
	float s = 0.0f;

	clock_t t1, t2;
	float gflop = 0;

	//if (argc > 1) N = atoi(argv[1]);  // get size of the vectors

	A = (float*)malloc(N * sizeof(float));  // allocate memory 
	B = (float*)malloc(N * sizeof(float));  // allocate memory 

	//srand(1);
	for (i = 0; i < N; i++)  // generate random data
	{
		A[i] = (float)rand()/RAND_MAX;
		B[i] = (float)rand()/RAND_MAX;
	}

	printf("Running CPU sum for %d elements\n", N);

	t1 = clock();

	s = dot_cpu(A, B, N);  // call compute kernel

	t2 = clock();

	printf("sum=%.2f\n", s);

	int t = (t2-t1)/CLOCKS_PER_SEC;
	if (t != 0)
		gflop = ((N-1) / t) / 1000.0f;
	printf("sec = %f   GFLOPS = %.3f\n", t, gflop);

	free(A);  // free allocated memory

	return 0;
}

int lab3_gpu()
{
	int i, N = 2097152;  // default vector size
	int *A, *B, *S, *devPtrA, *devPtrB, *devPtrS;

	clock_t t1, t2;
	float gflop = 0;

	//if (argc > 1) N = atoi(argv[1]);  // get size of the vectors

	// allocate memory 
	A = (int*)malloc(N * sizeof(int));
	B = (int*)malloc(N * sizeof(int));
	S = (int*)malloc(sizeof(int));

	//srand(1);
	for (i = 0; i < N; i++)  // generate random data
	{
		A[i] = (int)rand();
		B[i] = (int)rand();
	}

	printf("Running GPU sum for %i elements\n", N);

	t1 = clock();

	// wait until GPU kernel is done
	cudaThreadSynchronize();

	t2 = clock();

	printf("sum=%.2f\n", *S);

	int t = (t2-t1)/CLOCKS_PER_SEC;
	if (t != 0)
		gflop = ((N-1) / t) / 1000.0f;
	printf("sec = %f   GFLOPS = %.3f\n", t, gflop);

	// free allocated memory
	free(A);
	free(B);
	free(S);

	return 0;
}

/* lab4 */

// a = b * c
void mmult_cpu(float *a, float *b, float *c, int N)
{
    int i, j, k;
    
    for (j = 0; j < N; j++)
		for (k = 0; k < N; k++)
			for (i = 0; i < N; i++)
				a[i+j*N] += b[i+k*N]*c[k+j*N];
}

// a[][] = 0, b[][] = 1; c[][] = 1
void minit(float *a, float *b, float *c, int N)
{
    int i, j;
    
    for (j = 0; j < N; j++)
		for (i = 0; i < N; i++)
		{
			a[i+N*j] = 0.0f;
			b[i+N*j] = 1.0f;
			c[i+N*j] = 1.0f;
		}
}

void mprint(float *a, int N, int M)
{
    int i, j;
    
    for (j = 0; j < M; j++)
    {
        for (i = 0; i < M; i++)
            printf("%.2f ", a[i+N*j]);
			
        printf("...\n");
    }
    printf("...\n");
}

int lab4_cpu()
{
	int N = 1024;
    
    clock_t t1, t2, ta, tb;
    float flop, gflop;
    
    float *a = (float *)malloc(N*N*sizeof(float));
    float *b = (float *)malloc(N*N*sizeof(float));
    float *c = (float *)malloc(N*N*sizeof(float));

    minit(a, b, c, N);

    t1 = clock();

    mmult_cpu(a, b, c, N);

    t2 = clock();

    mprint(a, N, 5);
    
    free(a);
    free(b);
    free(c);

	double t = (double)(t2 - t1) / CLOCKS_PER_SEC;
    flop = N*N*N*2.0f;
    gflop = flop / t / 1000.0f;
    printf("msec = %f   GFLOPS = %.3f\n", t / 1000, gflop);

    return 0;
}

int lab4_gpu()
{
	int N = 1024;
    
    clock_t t1, t2, ta, tb;
    float flop, gflop;
    
    int *a = (int *)malloc(N*N*sizeof(int));
    int *b = (int *)malloc(N*N*sizeof(int));
    int *c = (int *)malloc(N*N*sizeof(int));

    minit(a, b, c, N);

    t1 = clock();

	float x = mmult(a, b, c, N);

    t2 = clock();

    mprint(a, N, 5);
    
    free(a);
    free(b);
    free(c);

	double t = (double)(t2 - t1) / CLOCKS_PER_SEC;
    flop = N*N*N*2.0f;
    gflop = flop / t / 1000.0f;
    printf("msec = %f   GFLOPS = %.3f\n", t / 1000, gflop);

    return 0;
}


/* Default template */

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}


int test()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

//Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}


/* Generic */

int main()
{
	lab4_cpu();

	printf("\n\n");

	lab4_gpu();

	getchar();
}