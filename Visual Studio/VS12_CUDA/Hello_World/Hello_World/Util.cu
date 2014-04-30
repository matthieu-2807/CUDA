#include "Util.h"


void __cudaSafeCall(cudaError_t err, char *file, int line) 
{
	if ((err) != cudaSuccess)
	{
		fprintf(stderr, "CUDA error in file %s at line %i: %s.\n", file, line, cudaGetErrorString(err));
		getchar();
		exit(EXIT_FAILURE);
	}
}

void cudaDeviceInit(int major, int minor)
{
	int devCount, device;

	cudaGetDeviceCount(&devCount);

	if (devCount == 0)
	{
		printf("No CUDA capable devices detected.\n");
		exit(EXIT_FAILURE);
	}

	for (device = 0; device < devCount; device++)
	{
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, device);

		// if device compute capability >= "major.minor"
		if (props.major > major || (props.major == major && props.minor == minor))
		{
			printf("CUDA capable device detected : %d.%d compute capability detected", props.major, props.minor);
			break;
		}

		if (device == devCount)
		{
			printf("No evice above %d.%d compute capability detected.\n", major, minor);
			exit(EXIT_FAILURE);
		}
		else
			cudaSetDevice(device);
	}
}