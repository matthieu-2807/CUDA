#include "Util.h"

__global__ void sum_kernel(double* v)
{
	double sum = 0;

	if (0 == threadIdx.x)
	{
		sum = 0;
		for (int i = 0; i < blockDim.x; i++)
			sum += v[i];
	}
}