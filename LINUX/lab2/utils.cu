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
	
	if (devCount == 0) {
		printf("No CUDA capable devices detected. \n");
		exit(EXIT_FAILURE);
	}
	
	for (device=0; device < devCount; device++){
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, device);
		
		if (props.major > 1 || (props.major == major && props.minor >= minor)) break;
	}
	
	if (device == devCount) {
		printf("No device above 1.2 compute capability detected. \n");
		exit(EXIT_FAILURE);
	}
	else cudaSetDevice(device);
}

void cudaDeviceCount()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	int device;
	for(device=0; device<deviceCount; device++)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, device);
		printf("Device %d has compute capability %d.%d \n", device, deviceProp.major, deviceProp.minor);
	}
	cudaDeviceInit();
}