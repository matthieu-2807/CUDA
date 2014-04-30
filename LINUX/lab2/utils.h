#ifndef __MY_CU_UTILS__
#define __MY_CU_UTILS__

#include <cuda.h>
#include <cuda_runtime_api.h>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)

void __cudaSafeCall(cudaError_t err, char *file, int line);

void cudaDeviceInit(int major, int minor);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif

