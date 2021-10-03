#include "cublas_v2.h"
#include <stdio.h> //NULL, printf
#ifndef gpuErrchk
#define gpuErrchk(ans)                                                                                                                               \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}
#endif
extern cudaError_t cudaStat;
extern cublasStatus_t stat;
extern cublasHandle_t handle;
extern cudaStream_t stream;
extern cublasStatus_t stat1;
extern cudaError_t cudaStat1;
extern cublasStatus_t stat2;
void extern cublasReset();