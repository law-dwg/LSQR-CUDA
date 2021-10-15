#pragma once
#include "cublas_v2.h"
#include "cusolverSp.h"
#include "cusparse.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <stdio.h> //NULL, printf

#ifndef IDX2C
#define IDX2C(i, j, ld) (((j) * (ld)) + (i))
#endif
extern const double ONE;
extern const double ZERO;
extern const double NEGONE;
extern const double TOL;

/** CUDA */
int checkDevice();
void cudaLastErrCheck();
#ifndef cudaErrCheck
#define cudaErrCheck(ans)                                                                                                                            \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA_assert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}
#endif

/** cuBLAS */
extern cublasStatus_t stat;
extern cudaError_t cudaStat;
extern cublasHandle_t handle;
// extern cudaStream_t stream;
void cublasStart();
void cublasStop();
#ifndef cublasErrCheck
#define cublasErrCheck(ans)                                                                                                                          \
  { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t code, const char *file, int line, bool abort = true) {
  const char *errstr = "CUBLAS UNKNOWN ERROR";
  if (code != CUBLAS_STATUS_SUCCESS) {
    switch (code) {
    case CUBLAS_STATUS_SUCCESS:
      errstr = "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      errstr = "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      errstr = "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      errstr = "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      errstr = "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      errstr = "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      errstr = "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      errstr = "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    fprintf(stderr, "cuBLAS_assert: %s %s %d\n", errstr, file, line);
    if (abort)
      exit(code);
  }
}
#endif

/** cuSPARSE */
extern cusparseStatus_t spStat;
extern cusparseHandle_t spHandle;
void cusparseStart();
void cusparseStop();
#ifndef cusparseErrCheck
#define cusparseErrCheck(ans)                                                                                                                        \
  { cusparseAssert((ans), __FILE__, __LINE__); }
inline void cusparseAssert(cusparseStatus_t code, const char *file, int line, bool abort = true) {
  if (code != CUSPARSE_STATUS_SUCCESS) {
    fprintf(stderr, "cuSPARSE_assert: %s %s %d\n", cusparseGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}
#endif

/** cuSolver */
extern cusolverStatus_t solStat;
extern cusolverSpHandle_t solHandle;
void cusolverStart();
void cusolverStop();
#ifndef cusolverErrCheck
#define cusolverErrCheck(ans)                                                                                                                        \
  { cusolverAssert((ans), __FILE__, __LINE__); }
inline void cusolverAssert(cusolverStatus_t code, const char *file, int line, bool abort = true) {
  const char *errstr = "CUSOLVER UNKNOWN ERROR";
  if (code != CUSOLVER_STATUS_SUCCESS) {
    switch (code) {
    case CUSOLVER_STATUS_SUCCESS:
      errstr = "CUSOLVER_STATUS_SUCCESS";
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      errstr = "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:
      errstr = "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:
      errstr = "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      errstr = "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      errstr = "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      errstr = "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      errstr = "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    }
    fprintf(stderr, "CUSOLVER_assert: %s %s %d\n", errstr, file, line);
    if (abort)
      exit(code);
  }
}
#endif