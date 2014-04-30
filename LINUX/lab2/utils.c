#include <stdlib.h>
#include <stdio.h>
#include "utils.h"

void __cudaSafeCall(cudaError_t err, char *file, int line) 
{
    if ((err) != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in file %s at line %i: %s.\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void cudaDeviceInit(int major, int minor)
{
    int devCount, device;

    cudaGetDeviceCount(&devCount);
	
	// Aucun GPU pouvant utiliser CUDA n'a été détecté
    if (devCount == 0)  
    {
        printf("No CUDA capable devices detected.\n");
        exit(EXIT_FAILURE);
    }
	
	// On regarde les propriétés du GPU
    for (device=0; device < devCount; device++)
    {
        struct cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device);

        /* If a device of compute capability >= major.minor is found, use it */
        if (props.major > major || (props.major == major && props.minor >= minor)) break;
    }
	
	// Si les propriétés sont inférieures à ce qui est demandé, on quitte le code, sinon on poursuit le code
    if (device == devCount)  
    {
        printf("No device with %d.%d compute capability or above is detected.\n", major, minor);
        exit(EXIT_FAILURE);
    }
    else
        cudaSetDevice(device);
}

