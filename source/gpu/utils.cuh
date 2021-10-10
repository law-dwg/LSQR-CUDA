#pragma once
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
#ifndef IDX2C
#define IDX2C(i, j, ld) (((j) * (ld)) + (i))
// static __inline__ void modify (cublasHandle_t handle, float *m, int ldm, int n, int p, int q, float alpha, float beta){
//    cublasSscal (handle, n-q, &alpha, &m[IDX2C(p,q,ldm)], ldm);
//    cublasSscal (handle, ldm-p, &beta, &m[IDX2C(p,q,ldm)], 1);
//}
#endif
extern cublasStatus_t stat;
extern cudaError_t cudaStat;
extern cublasHandle_t handle;
extern cublasStatus_t statCreateHandle;
// extern cudaStream_t stream;
// extern cudaError_t cudaStatCreateStream;
// extern cublasStatus_t statSetStream;
extern const double ONE;
extern const double ZERO;
extern const double NEGONE;

void cublasStart();
void cublasStop();
int checkDevice();
const char *cublasGetErrorString(cublasStatus_t status);