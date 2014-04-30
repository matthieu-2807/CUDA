#include <stdio.h> 
#include <stdlib.h>
#include "utils.h"

float vecAdd(real* A, real* B, real* C, int N);

int  main(int argc, char **argv)
{
    int i, N = 720896;  /* default vector size */
    real *A;
    real *B; 
    real *C; 
    float rt;

    /* check for user-supplied vector size */
    if (argc > 1)
        N = atoi(argv[1]);

    /* select an appropriate device */
    cudaDeviceInit(1,3);

    printf("Running GPU vecAdd for %i elements\n", N);

     /* allocate host memory */
    A = (real*)malloc(N * sizeof(real));
    B = (real*)malloc(N * sizeof(real));
    C = (real*)malloc(N * sizeof(real));

    /* generate random data */
    for (i = 0; i < N; i++)
    {
        A[i] = (real)random();
        B[i] = (real)RAND_MAX - A[i];
    }

    rt = vecAdd(A, B, C, N);

    printf("time=%.4f seconds, MFLOPS=%.1f\n", rt, (real)N/rt/1E6);

    /* print out first 10 results */
    for (i = 0; i < 10; i++)
        printf("C[%i]=%.2f\n", i, C[i]);
  
    /* free allocated host memory */
    free(A); 
    free(B);
    free(C);
	
    return EXIT_SUCCESS;
}

