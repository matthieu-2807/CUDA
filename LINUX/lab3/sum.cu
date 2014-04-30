#include <stdio.h> 
#include <stdlib.h>
#include <sys/time.h>

#define THREADS_PER_BLOCK 512

__global__ void dot(int *a, int *b, int *c) {
	__shared__ int temp[THREADS_PER_BLOCK];
	int index = threadIdx.x+blockIdx.x*blockDim.x;
	temp [threadIdx.x]=a[index]*b[index];
	__syncthreads();
	if(0==threadIdx.x){ 
		int sum = 0 ;
		for(int i =0; i<THREADS_PER_BLOCK;i++) 
			sum+=temp[i];
		atomicAdd(c,sum);
	}
}

int main(void) {
	int i, N = 2097152;
	int *a, *b ,*c;
	int *da, *db, *dc;
	int size = N*sizeof(int);

	// On alloue la memoire CPU et GPU
	a=(int *)malloc(size);
	b=(int *)malloc(size);
	c=(int *)malloc(sizeof(int));
	cudaMalloc((void**)&da,size); 
	cudaMalloc((void**)&db,size); 
	cudaMalloc((void**)&dc,sizeof(int));
	
	// On donne des donnees aleatoire
	for (i = 0; i < N; i++) {
		  a[i] = rand(); 
		  b[i] = rand(); 
	}
	
	// On copie les donnees du CPU vers le GPU
	cudaMemcpy(da, a, N * sizeof(int),  cudaMemcpyHostToDevice); 
	cudaMemcpy(db, b, N * sizeof(int),  cudaMemcpyHostToDevice); 
	
	// On fait le produit scalaire
	dot<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(da,db,dc);
	
	// On copie le resultat du GPU sur le CPU
	cudaMemcpy(c, dc, sizeof(int), cudaMemcpyDeviceToHost); 

	printf("sum=%d \n", *c);
	
	// On libere la memoire
	free(a);  
	free(b);  
	free(c);
	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);
	

	return EXIT_SUCCESS;
}