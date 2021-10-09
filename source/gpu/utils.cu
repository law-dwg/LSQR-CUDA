#include "cublas_v2.h"
#include "utils.cuh"
#include <cuda.h>
#include <iostream>
#define MODE CUBLAS_POINTER_MODE_HOST

cublasStatus_t stat;
cudaError_t cudaStat;
cublasHandle_t handle;
cudaStream_t stream;
cublasStatus_t statCreateHandle = cublasCreate(&handle);
cudaError_t cudaStatCreateStream = cudaStreamCreate(&stream);
cublasStatus_t statSetStream = cublasSetStream(handle, stream);
cublasStatus_t statSetPointerMode = cublasSetPointerMode(handle, MODE);

void cublasReset() {
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