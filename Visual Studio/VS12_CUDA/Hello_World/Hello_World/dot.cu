#include "Util.h"

__global__ void dot_kernel(int* A, int* B, int* S)
{
	__shared__ int temp[THREADS_PER_BLOCK];
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	temp[threadIdx.x] = A[index] * B[index];

	//_syncThreads();

	if (0 == threadIdx.x)
	{
		int sum = 0;
		for (int i = 0; i < THREADS_PER_BLOCK; i++)
			sum += temp[i];

		//atomicAdd(S, sum);
	}
}

extern "C" int dot(int* A, int* B, int* S, int N)
{
	int *devPtrA, *devPtrB, *devPtrS;

	// Alocate
	cudaSafeCall( cudaMalloc((void**)&devPtrA, N * sizeof(int)) ); 
	cudaSafeCall( cudaMalloc((void**)&devPtrB, N * sizeof(int)) );
	cudaSafeCall( cudaMalloc((void**)&devPtrS, sizeof(int)) );

	cudaSafeCall( cudaMemcpy(devPtrA, A, N * sizeof(int), cudaMemcpyHostToDevice) ); 
	cudaSafeCall( cudaMemcpy(devPtrB, B, N * sizeof(int), cudaMemcpyHostToDevice) );

	//dot_kernel<<< N/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(devPtrA, devPtrB, devPtrS);

	// copy from device to host
	cudaSafeCall( cudaMemcpy(S, devPtrS, sizeof(int), cudaMemcpyDeviceToHost) ); 

	// free device memory
	cudaSafeCall( cudaFree(devPtrA) ); 
	cudaSafeCall( cudaFree(devPtrB) ); 
	cudaSafeCall( cudaFree(devPtrS) ); 

	return 0;
}