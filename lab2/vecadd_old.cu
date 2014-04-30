#include <stdio.h> 
#include <stdlib.h>
#include <sys/time.h>


__global__  void vecAdd(float* A, float* B, float* C) 
{ 
    // threadIdx.x is a built-in variable provided by CUDA at runtime 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    C[i] = A[i] + B[i]; 
}

#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)

void __cudaSafeCall(cudaError_t err, char *file, int line) 
{
    if ((err) != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in file %s at line %i: %s.\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


int  main(int argc, char **argv)
{
    int i, N = 720896;  /* default vector size */
    float *A, *devPtrA;
    float *B, *devPtrB; 
    float *C, *devPtrC; 

    cudaEvent_t start, stop;

    float rt;

    /* check for user-supplied vector size */
    if (argc > 1) 
        N = atoi(argv[1]);

    printf("Running GPU vecAdd for %i elements\n", N);

     /* allocate host memory */
    A = (float*)malloc(N * sizeof(float));
    B = (float*)malloc(N * sizeof(float));
    C = (float*)malloc(N * sizeof(float));

    /* generate random data */
    for (i = 0; i < N; i++)
    {
        A[i] = (float)random();
        B[i] = (float)RAND_MAX - A[i];
    }

    /* ----- GPU add-on ------- */
    cudaSafeCall( cudaMalloc((void**)&devPtrA, N * sizeof(float)) ); 
    cudaSafeCall( cudaMalloc((void**)&devPtrB, N * sizeof(float)) ); 
    cudaSafeCall( cudaMalloc((void**)&devPtrC, N * sizeof(float)) ); 

    /* timing code */
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, 0);

    cudaSafeCall( cudaMemcpy(devPtrA, A, N * sizeof(float), cudaMemcpyHostToDevice) ); 
    cudaSafeCall( cudaMemcpy(devPtrB, B, N * sizeof(float), cudaMemcpyHostToDevice) ); 

    /* call compute kernel */
	/* vecAdd(N, A, B, C); */
    vecAdd<<<N/512, 512>>>(devPtrA, devPtrB, devPtrC);
    
    cudaSafeCall( cudaMemcpy(C, devPtrC, N * sizeof(float),  cudaMemcpyDeviceToHost) ); 

	/* timing */
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&rt, start, stop);  /* in milliseconds */
    rt /= 1E3;  /* convert to seconds */

    printf("time=%.4f seconds, MFLOPS=%.1f\n", rt, (float)N/rt/1E6);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaSafeCall( cudaFree(devPtrA) ); 
    cudaSafeCall( cudaFree(devPtrB) ); 
    cudaSafeCall( cudaFree(devPtrC) ); 
    /* ------------ */
        
    /* print out first 10 results */
    for (i = 0; i < 10; i++)
        printf("C[%i]=%.2f\n", i, C[i]);
  
    /* free allocated host memory */
    free(A); 
    free(B);
    free(C);
	
	return EXIT_SUCCESS;
}

