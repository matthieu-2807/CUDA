#include <stdio.h> 
#include <stdlib.h>

#include "utils.h"

__global__ void vecAdd_kernel(real* A, real* B, real* C) 
{ 
    // threadIdx.x is a built-in variable provided by CUDA at runtime 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    C[i] = A[i] + B[i]; 
}

extern "C" float vecAdd(real* A, real* B, real* C, int N) 
{
    real *devPtrA;
    real *devPtrB; 
    real *devPtrC; 

    cudaEvent_t start, stop;

    float rt;

    cudaSafeCall( cudaMalloc((void**)&devPtrA, N * sizeof(real)) ); 
    cudaSafeCall( cudaMalloc((void**)&devPtrB, N * sizeof(real)) ); 
    cudaSafeCall( cudaMalloc((void**)&devPtrC, N * sizeof(real)) ); 

    /* timing code */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);

    cudaSafeCall( cudaMemcpy(devPtrA, A, N * sizeof(real), cudaMemcpyHostToDevice) ); 
    cudaSafeCall( cudaMemcpy(devPtrB, B, N * sizeof(real), cudaMemcpyHostToDevice) ); 

    /* call compute kernel */
    /* vecAdd(N, A, B, C); */
    vecAdd_kernel<<<N/512, 512>>>(devPtrA, devPtrB, devPtrC);
    
    cudaSafeCall( cudaMemcpy(C, devPtrC, N * sizeof(real),  cudaMemcpyDeviceToHost) ); 

    /* timing */
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&rt, start, stop);  /* in milliseconds */
    rt /= 1E3;  /* convert to seconds */

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaSafeCall( cudaFree(devPtrA) ); 
    cudaSafeCall( cudaFree(devPtrB) ); 
    cudaSafeCall( cudaFree(devPtrC) ); 

    return rt;
}

