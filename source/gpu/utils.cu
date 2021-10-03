#include "cublas_v2.h"
#include "utils.cuh"
#include <cuda.h>

cudaError_t cudaStat;
cublasHandle_t handle;
cudaStream_t stream;
cublasStatus_t stat1 = cublasCreate(&handle);
cudaError_t cudaStat1 = cudaStreamCreate(&stream);
cublasStatus_t stat = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
cublasStatus_t stat2 = cublasSetStream(handle, stream);
void cublasReset() {
  stat1 = cublasCreate(&handle);
  cudaStat1 = cudaStreamCreate(&stream);
  stat = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  stat2 = cublasSetStream(handle, stream);
}