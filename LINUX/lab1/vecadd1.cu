#include <stdio.h> 
#include <stdlib.h>
#include <sys/time.h>


__global__  void vecAdd(float* A, float* B, float* C) ;


int  main(int argc, char **argv)
{
    int i, N = 720896;  /* default vector size */
    float *A, *dev_a;
    float *B, *dev_b; 
    float *C, *dev_c; 

	cudaEvent_t begin, stop;
    float rt;

    /* check for user-supplied vector size */
    if (argc > 1) 
        N = atoi(argv[1]);

    printf("Running GPU vecAdd for %i elements\n", N);

    /* allocate memory - host */ 
    A = (float*)malloc(N * sizeof(float));
    B = (float*)malloc(N * sizeof(float));
    C = (float*)malloc(N * sizeof(float));
	
	
	for (i = 0; i < N; i++)  /* generate random data */
    {
        A[i] = (float)random();
        B[i] = (float)RAND_MAX - A[i];
    }
	
	/* allocate memory - GPU */
    cudaError_t err;
    err = cudaMalloc((void**)&dev_a, N * sizeof(float));
	if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc ERROR : , %s.\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
    err = cudaMalloc((void**)&dev_b, N * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc ERROR : , %s.\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	err = cudaMalloc((void**)&dev_c, N * sizeof(float));
	if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc ERROR : , %s.\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	// Copie les donnees HOST -> GPU
    cudaMemcpy(dev_a, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, B, N * sizeof(float), cudaMemcpyHostToDevice);

	/* On cree les timer et on lance begin */
    cudaEventCreate(&begin);
    cudaEventCreate(&stop);
    cudaEventRecord(begin, 0);
	
    /* On appelle la methode */
	vecAdd<<<N/512, 512>>>(dev_a, dev_b, dev_c);
	
    // Copie les donnees de GPU -> HOST
    cudaMemcpy(C, dev_c, N * sizeof(float), cudaMemcpyDeviceToHost);

	/* On arrete le chrono et on compare begin et stop */
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&rt, begin, stop);  /* in milliseconds */
	rt /= 1E3;
	
    printf("time=%.4f seconds, MFLOPS=%.1f\n", rt, (float)N/rt/1E6);
	
	/* On supprime les timers */
    cudaEventDestroy(begin);
    cudaEventDestroy(stop);
	
    /* Affiche les 10 premiers resultats */
    for (i = 0; i < 10; i++)
        printf("C[%i]=%.2f\n", i, C[i]);
	
	/* Libere la memoire du host */
    free(A); 
    free(B);
    free(C);

    /* Libere la memoire GPU */
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
	
	return EXIT_SUCCESS;
}

__global__  void vecAdd(float* A, float* B, float* C) 
{  
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    C[i] = A[i] + B[i]; 
}
