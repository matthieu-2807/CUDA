#include "Util.h"

__global__ void mmult_kernel(float* A, float* B, float* C, int N)
{
	float sum = 0;

	int i = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	int j = blockIdx.y;

	for (int k = 0; k < N*N; ++k)
		sum += B[i + N*k] * C[k + N*j];

	A[i + N*j] = sum;
}

extern "C" float mmult(float* A, float* B, float* C, int N)
{
	float *devPtrA, *devPtrB, *devPtrC;

	// Allocate
	cudaSafeCall( cudaMalloc((void**)&devPtrA, N * N * sizeof(float)) ); 
	cudaSafeCall( cudaMalloc((void**)&devPtrB, N * N * sizeof(float)) );
	cudaSafeCall( cudaMalloc((void**)&devPtrC, N * N * sizeof(float)) );

	cudaSafeCall( cudaMemcpy(devPtrB, B, N * N * sizeof(float), cudaMemcpyHostToDevice) ); 
	cudaSafeCall( cudaMemcpy(devPtrC, C, N * N * sizeof(float), cudaMemcpyHostToDevice) );

	// define grid and thread block sizes
	dim3 threads(THREADS_PER_BLOCK);
    dim3 grid(N/THREADS_PER_BLOCK, N);

	//mmult_kernel<<< grid, threads >>>(devPtrA, devPtrB, devPtrC, N);

	// copy from device to host
	cudaSafeCall( cudaMemcpy(A, devPtrA, sizeof(float), cudaMemcpyDeviceToHost) ); 

	// free device memory
	cudaSafeCall( cudaFree(devPtrA) );
	cudaSafeCall( cudaFree(devPtrB) );
	cudaSafeCall( cudaFree(devPtrC) );

	return 0;
}