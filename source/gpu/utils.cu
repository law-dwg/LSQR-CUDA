#include "cublas_v2.h"
#include "utils.cuh"
#include <cuda.h>
#include <iostream>
#define MODE CUBLAS_POINTER_MODE_HOST

cublasStatus_t stat;
cudaError_t cudaStat;
cublasHandle_t handle;
cudaStream_t stream;
cublasStatus_t statCreateHandle;
cudaError_t cudaStatCreateStream;
cublasStatus_t statSetStream;
cublasStatus_t statSetPointerMode;

void cublasStart() {
  statCreateHandle = cublasCreate(&handle);
  std::cout << cublasGetErrorString(statCreateHandle) << std::endl;
  if (statCreateHandle != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS create handle failed\n");
    // return EXIT_FAILURE;
  }
  cudaStatCreateStream = cudaStreamCreate(&stream);
  if (cudaStatCreateStream != cudaSuccess) {
    printf("CUBLAS create stream failed\n");
    // return EXIT_FAILURE;
  }
  statSetStream = cublasSetStream(handle, stream);
  if (statSetStream != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS set pointer failed\n");
    // return EXIT_FAILURE;
  }
  statSetPointerMode = cublasSetPointerMode(handle, MODE);
  if (statSetPointerMode != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS set pointer failed\n");
    // return EXIT_FAILURE;
  }
};
void cublasStop() { cublasDestroy(handle); };
int checkDevice() {
  // Check Cuda Capabale Device
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  int device;
  if (deviceCount > 0) {
    for (device = 0; device < deviceCount; ++device) {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, device);
      printf("Device %s has compute capability %d.%d.\n", deviceProp.name, deviceProp.major, deviceProp.minor);
      printf("Number of multiprocessors: %d\n", deviceProp.multiProcessorCount);
      printf("Clock rate: %d Hz\n", deviceProp.clockRate);
      printf("Total amount of global memory: %d KB\n", deviceProp.totalGlobalMem / 1024);
      printf("Total amount of constant memory: %d KB\n", deviceProp.totalConstMem / 1024);
      printf("Total amount of shared memory per block: %d KB\n", deviceProp.sharedMemPerBlock / 1024);
      printf("Total amount of shared memory per SM: %d KB\n", 64);
      printf("Warp size: %d\n", deviceProp.warpSize);
      printf("Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
      printf("Maximum number of blocks per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor / deviceProp.maxThreadsPerBlock);
      printf("Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
      printf("Maximum number of warps per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor / 32);
      printf("Maximum Grid size: (%d,%d,%d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
      printf("Maximum block dimension: (%d,%d,%d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    }
  } else {
    printf("NO CUDA DEVICE AVAILABLE");
  }
  return deviceCount;
};
const char *cublasGetErrorString(cublasStatus_t status) {
  switch (status) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  return "unknown error";
};