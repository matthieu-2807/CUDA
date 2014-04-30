#include "Util.h"

__global__ void vecAdd_kernel(float* A, float* B, float* C)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	C[index] = A[index] + B[index];
}

extern "C" float vecAdd(float* A, float* B, float* C, int N) 
{
	int i;
	float *dev_A, *dev_B, *dev_C; /* create devices variables to use it in device memory */

	clock_t t1, t2;
	float rt;

	/* allocate device memory  */
	cudaSafeCall( cudaMalloc((void**)&dev_A, N * sizeof(float)) );
	cudaSafeCall( cudaMalloc((void**)&dev_B, N * sizeof(float)) );
	cudaSafeCall( cudaMalloc((void**)&dev_C, N * sizeof(float)) );

	for (i = 0; i < N; i++)  /* generate random data */
	{
		A[i] = (float)rand();
		B[i] = (float)RAND_MAX - A[i];
	}

	/* timing */
	t1 = clock();

	/* Copy variables into device memory variables */
	cudaSafeCall( cudaMemcpy(dev_A, A, N * sizeof(float), cudaMemcpyHostToDevice) );
	cudaSafeCall( cudaMemcpy(dev_B, B, N * sizeof(float), cudaMemcpyHostToDevice) );

	/* call compute kernel */
	vecAdd_kernel<<< N/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(dev_A, dev_B, dev_C);

	/* Copy device memory variables into variables */
	cudaSafeCall( cudaMemcpy(C, dev_C, N * sizeof(float), cudaMemcpyDeviceToHost) );

	/* timing */
	t2 = clock();

	rt = (double)(t2 - t1)/CLOCKS_PER_SEC;
	
	/* free allocated device memory */
	cudaSafeCall( cudaFree(dev_A) );
	cudaSafeCall( cudaFree(dev_B) );
	cudaSafeCall( cudaFree(dev_C) );

	return rt;
}