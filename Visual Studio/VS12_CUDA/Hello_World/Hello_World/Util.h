#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>

#if defined(__cplusplus) extern "C" { 
#endif

const int THREADS_PER_BLOCK = 512;

#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)

void __cudaSafeCall(cudaError_t err, char *file, int line) ;

void cudaDeviceInit(int major, int minor);

#if defined(__cplusplus) }
#endif